I have always wondered how Intel's AMX support can be used to multiply matrices. I researched around and couldn't find anything that starts with 2 matrices and uses AMX to multiply them[^1].

In this post, I will take two 1024×1024 matrices, **A** and **B**, with elements in **bfloat16** format. These matrices will be divided into tiles of a predefined size. Using AMX instructions, I'll explain tiled matrix multiplication to produce the resulting matrix **C**. **C** will have elements in **FP32** format[^2].

[^1]: AMX (Advanced Matrix Extension) is Intel's ISA extension capability that adds matrix multiplication instructions directly to the hardware. [More info on Wikichip](https://en.wikichip.org/wiki/x86/amx)

[^2]: The AMX `tdpbf16ps` instruction, which we plan to use, takes two tiles of bfloat16 data, performs a matrix multiply, and produces the result in "float" format. This approach strikes a good balance: bfloat16(2 bytes in size) reduces memory bandwidth usage, while the hardware automatically converts the result to float, providing higher precision without additional cost.


## Core Idea

For bfloat16 matrix multiplication, I'll use the below intrinsic:

```c
__tile_dpbf16ps (__tile1024i* dst, __tile1024i src0, __tile1024i src1)
```

## Intrinsic Explanation

Here is the explanation for this intrinsic:

> Compute dot-product of BF16 (16-bit) floating-point pairs in tiles src0 and src1, accumulating the intermediate single-precision (32-bit) floating-point elements with elements in dst, and store the 32-bit result back to tile dst. The shape of tile is specified in the struct of __tile1024i. The register of the tile is allocated by compiler.

In simple terms, this intrinsic (which translates to `tdpbf16ps` instruction) will perform matrix multiplication between `src0` with `src1`, storing the result in `dst`. `src0`, `src1` and `dst` are registers similar to standard x86 registers, with the key difference being that these are 2-dimensional registers.

One thing to emphasize in the quote above is the word "pairs", whose usage will be explained below.

## Example

I will start with a simple example of multiplying 2 bfloat16 matrices of size 2x4 and 4x2. The result will be a 2x2 matrix.

[

![](https://substackcdn.com/image/fetch/$s_!XWGC!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F5559c943-0ec6-457a-9a36-5d7efe98b488_1800x716.png)

](https://substackcdn.com/image/fetch/$s_!XWGC!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F5559c943-0ec6-457a-9a36-5d7efe98b488_1800x716.png)

In order to keep the explanation simple, I am going to concentrate on computing only **c00.**

**Case 1**: According to standard way of matrix multiplication, **to compute c00**, we multiply first row of matrix A with the first column of matrix B, summing the products, and storing the result in the first element of the destination matrix. This is shown below.

[

![](https://substackcdn.com/image/fetch/$s_!z1eV!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F08ec83c3-c02c-40e5-bf69-997804b11446_2858x912.png)

](https://substackcdn.com/image/fetch/$s_!z1eV!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F08ec83c3-c02c-40e5-bf69-997804b11446_2858x912.png)

**Case 2**: According to the Intel intrinsic discussed above, **to compute c00**, we take a row wise pair of elements from A and a row-wise pair from B, multiply them together, add the products, and store the result as the first element in dst. This is shown below.

Note: We will only consider partial result of **c00** not the entire dot product.

[

![](https://substackcdn.com/image/fetch/$s_!cI69!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F44cdaaa2-2b81-4682-b7e7-0d271e7108d7_2858x1035.png)

](https://substackcdn.com/image/fetch/$s_!cI69!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F44cdaaa2-2b81-4682-b7e7-0d271e7108d7_2858x1035.png)

In computing **c00**, case 2's way of multiplying pair of numbers is incorrect. Case 2 considers A's row 0(which is right) and B's first 2 elements of row 0 for c00. This is clearly a wrong way of doing standard matrix multiplication. But as the Intel Intrinsic explains, this is how the hardware does things. **In order to match the hardware, we need to transform B before using AMX matrix multiply instructions.**

More specifically every element in "odd" row should be put adjacent to corresponding element in "even" row. For e.g., in B matrix element 2(row 1 and column 0) should move to row 0 and column 1, similarly 4(row 1 and column 1) should move row 0 and column 3.

[

![](https://substackcdn.com/image/fetch/$s_!ZVar!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F05d499b6-9432-4fc2-993c-8ac440ab8332_1845x690.png)

](https://substackcdn.com/image/fetch/$s_!ZVar!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F05d499b6-9432-4fc2-993c-8ac440ab8332_1845x690.png)

B matrix has been transformed from 4x2 to 2x4 matrix. Now, A is a 2x4 matrix, B is also a 2x4 matrix, these can be loaded to src0 and src1 registers respectively and multiplied to get the result. As shown in the below picture, **c00** is obtained by computing "**pair"** dot product of A's row 0 and B's column 0. This obeys the rule of standard matrix multiplication.

[

![](https://substackcdn.com/image/fetch/$s_!sv2K!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Ffa3b85a1-f5d4-4cce-9dfb-df724d52dd56_2604x1105.png)

](https://substackcdn.com/image/fetch/$s_!sv2K!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Ffa3b85a1-f5d4-4cce-9dfb-df724d52dd56_2604x1105.png)

Multiply 1024x1024 matrices
---------------------------

Now that we have grasped the core idea, we will multiply 1024x1024 bfloat matrices, A and B. To do that, we need to transform B matrix to the form as discussed above.

[

![](https://substackcdn.com/image/fetch/$s_!WvnZ!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fc3712a99-064c-4342-85a4-ccaf0db14a93_2182x595.png)

](https://substackcdn.com/image/fetch/$s_!WvnZ!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fc3712a99-064c-4342-85a4-ccaf0db14a93_2182x595.png)

Following the transformation, transformed B is 512×2048 matrix.

AMX supports eight 2-dimensional registers where each register's maximum size is 1KB each. Since each register is 2-dimensional, it can hold maximum of 16 rows and 64 bytes per row(16x64 = 1KB)[3](https://thoughtsorganized.substack.com/p/using-intels-amxadvanced-matrix-extensions#footnote-3-163968298).

[

![](https://substackcdn.com/image/fetch/$s_!IRFH!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F2ce1a30d-a858-426d-9bd2-15812ffb2f9e_1650x626.png)

](https://substackcdn.com/image/fetch/$s_!IRFH!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F2ce1a30d-a858-426d-9bd2-15812ffb2f9e_1650x626.png)

I'll choose the maximum tile size to be 1KB and tile dimensions to be 16x64. Each tile has 16 rows and each row has 64 bytes. For bfloat16 datatype, tile dimensions are 16x32(bfloat is 2 bytes). There are 16 rows and 32 bfloat16 elements/row. For 1024x1024 matrix A, there are 1024 rows/(16 rows/tile) = 64 tiles in row direction and 1024 columns/(32 bfloat16s per row) = 32 tiles in column direction. Similarly, for 512x2048 matrix B, there are 512/16 = 32 tiles in row direction and 2048/32 = 64 tiles in column direction.

Considering number of tiles, A will be 64x32 matrix and B will be 32x64 matrix. The resulting matrix C will be a 64x64 matrix. All that needs to be done is iterating our rows, columns with these dimensions, compute pointer arithmetic to load appropriate tiles to tile registers, use __tile_dpbf16ps to multiply them and store the result to C matrix.

Below figure explains the discussed matrix multiplication.

[

![](https://substackcdn.com/image/fetch/$s_!ay0P!,w_2400,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Ff007a319-ec4c-47d8-bd97-0d10cea34b75_3345x1455.png)

](https://substackcdn.com/image/fetch/$s_!ay0P!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Ff007a319-ec4c-47d8-bd97-0d10cea34b75_3345x1455.png)

Below is the relevant part of the code.

[

![](https://substackcdn.com/image/fetch/$s_!fbux!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F433d27b7-f423-4132-809f-c484ed3090a7_918x445.png)

](https://substackcdn.com/image/fetch/$s_!fbux!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F433d27b7-f423-4132-809f-c484ed3090a7_918x445.png)

Only the inner loop is shown here. Single tile of C matrix is loaded to `tmm0 `register(line: 91). Loop over TILES_K, whose value is 32, load each tile of A into `tmm1`(line: 97) and corresponding tile of B into `tmm2(line:101)`, and finally matrix multiply them using `_tile_dpbf16ps`(line:104) intrinsic. Store the result in `tmm0 `back to corresponding C entry which was loaded from(line:107).

The entire source code can be found [here](https://github.com/dilipshivaraju/code-samples/tree/main/AMX).

Reminder that this code runs only on Intel processors that has AMX support.


[2](https://thoughtsorganized.substack.com/p/using-intels-amxadvanced-matrix-extensions#footnote-anchor-2-163968298)

The AMX `tdpbf16ps` instruction, which we plan to use, takes two tiles of bfloat16 data, performs a matrix multiply, and produces the result in "float" format. This approach strikes a good balance: bfloat16(2 bytes in size) reduces memory bandwidth usage, while the hardware automatically converts the result to float, providing higher precision without additional cost.

[3](https://thoughtsorganized.substack.com/p/using-intels-amxadvanced-matrix-extensions#footnote-anchor-3-163968298)

Of course, the programmer can configure these registers in anyway he likes it. But we will not get into the complexities of that here.