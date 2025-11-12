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
    tiles = (n_cols + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    mask_row = row_offsets < n_rows
    m = tl.full([BLOCK_SIZE_M, 1], float('-inf'), tl.float32)
    d = tl.zeros([BLOCK_SIZE_M, 1], tl.float32)

    for tile in range(tiles):
        tile_offsets = tile * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        mask_col = tile_offsets < n_cols
        x = tl.load(input_ptr + row_offsets[:, None] * input_row_stride + tile_offsets[None, :] * input_col_stride, mask=mask_row[:, None] & mask_col[None, :], other=float('-inf')).to(tl.float32)
        x_max = tl.max(x, axis=-1, keep_dims=True)
        m_new = tl.maximum(m, x_max)
        d = d * tl.exp(m - m_new) + tl.sum(tl.exp(x - m_new), axis=-1, keep_dims=True)
        m = m_new

    for tile in range(tiles):
        tile_offsets = tile * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        mask_col = tile_offsets < n_cols
        x = tl.load(input_ptr + row_offsets[:, None] * input_row_stride + tile_offsets[None, :] * input_col_stride, mask=mask_row[:, None] & mask_col[None, :], other=float('-inf')).to(tl.float32)
        x_sub_max = x - m
        exp_x_sub_max = tl.exp(x_sub_max)
        softmax_output = exp_x_sub_max / d
        tl.store(output_ptr + row_offsets[:, None] * output_row_stride + tile_offsets[None, :] * output_col_stride, softmax_output.to(tl.float16), mask=mask_row[:, None] & mask_col[None, :])
        

def softmax(x):
    n_rows, n_cols = x.shape
    y = torch.empty_like(x)
    BLOCK_SIZE_M = 16
    BLOCK_SIZE_N = 128

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
    x = torch.randn((1024, 128), device='cuda', dtype=torch.float16)
    torch_output = torch.softmax(x, dim=-1)
    triton_output = softmax(x)
    print(torch_output)
    print(triton_output)
    if torch.allclose(triton_output, torch_output, atol=1e-4, rtol=1e-4):
        print(True)
    else:
        print(False)
