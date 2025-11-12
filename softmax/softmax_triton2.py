import triton
import triton.language as tl
import torch

@triton.jit
def softmax_kernel(input_ptr, output_ptr, 
                input_row_stride, 
                input_col_stride,
                output_row_stride,
                output_col_stride,
                n_rows, n_cols, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr):
    # input_ptr = input_ptr.to(tl.pointer_type(tl.float32))
    # output_ptr = output_ptr.to(tl.pointer_type(tl.float32))

    pid = tl.program_id(0)
    row_offsets = pid * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    col_offsets = tl.arange(0, BLOCK_SIZE_N)
    mask = (row_offsets[:, None] < n_rows) & (col_offsets[None, :] < n_cols)

    row = tl.load(input_ptr + row_offsets[:, None] * input_row_stride + col_offsets[None, :] * input_col_stride, mask=mask, other=float('-inf')).to(tl.float32)

    row_minus_max = row - tl.max(row, axis=-1, keep_dims=True)
    numerator = tl.exp(row_minus_max)
    denominator = tl.sum(numerator, axis=-1, keep_dims=True)
    softmax_output = numerator / denominator

    tl.store(output_ptr + row_offsets[:, None] * output_row_stride + col_offsets[None, :] * output_col_stride, softmax_output.to(tl.float16), mask=mask)

def softmax(x):
    n_rows, n_cols = x.shape
    y = torch.empty_like(x)
    BLOCK_SIZE_M = 16
    BLOCK_SIZE_N = triton.next_power_of_2(n_cols)  # 块大小取2的幂

    # 配置每个块的warp数量
    num_warps = 4
    if BLOCK_SIZE_N >= 2048:
        num_warps = 8
    if BLOCK_SIZE_N >= 4096:
        num_warps = 16

    softmax_kernel[(triton.cdiv(n_rows, BLOCK_SIZE_M),)]( 
        x, y,
        *x.stride(), *y.stride(),
        n_rows,
        n_cols,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        num_warps=num_warps  # 补充num_warps参数（之前遗漏了）
    )

    return y

if __name__ == '__main__':
    x = torch.randn((4096, 1024), device='cuda', dtype=torch.float16)
    torch_output = torch.softmax(x, dim=-1)
    triton_output = softmax(x)
    print(torch_output)
    print(triton_output)
    if torch.allclose(triton_output, torch_output, atol=1e-5, rtol=1e-4):
        print(True)
    else:
        print(False)
