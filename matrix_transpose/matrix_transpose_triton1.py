import torch
import triton.language as tl
import triton

@triton.jit
def matrix_transpose_kernel(input_ptr, out_ptr, rows, cols):
    pid = tl.program_id(0)
    # 说明：kernel 假设输入/输出张量在内存中按行连续（contiguous）排布
    # pid 对应输入矩阵的线性索引
    cur = tl.load(input_ptr + pid)
    # 计算在原矩阵中的行列位置 (row_index, col_index)
    row_index = pid // cols
    col_index = pid % cols
    # 转置后：原来的 (row, col) 变成 (col, row)
    # 新矩阵形状是 (cols, rows)，线性索引 = col_index * rows + row_index
    transposed_index = col_index * rows + row_index
    tl.store(out_ptr + transposed_index, cur)

def matrix_transpose_triton(input, output, rows, cols):
    grid = (rows * cols,)
    matrix_transpose_kernel[grid](input, output, rows, cols)

if __name__ == '__main__':
    input = torch.randn((3, 2), device='cuda:0')
    # 注意：不要用 empty_like(input.t()) 来创建输出！
    # 原因：input.t() 是视图（view），其 stride 会被转置为 (1, 2)（以 (2,3) 为例），
    # empty_like(input.t()) 会继承这种非连续内存布局（非 contiguous）。
    # 但我们的 kernel 用线性索引写入，默认假设输出为连续内存（例如 (2,3) 的连续 stride 应为 (3,1)）。
    # 如果输出不是连续的，写入地址与预期不符，导致转置结果错位。
    # 正确做法：显式按目标形状创建一个连续内存的张量。
    output = torch.empty((input.shape[1], input.shape[0]), device=input.device)

    output_torch = input.t()
    matrix_transpose_triton(input, output, *(input.shape))

    if torch.allclose(output, output_torch):
        print(True)
    else:
        print(False)
