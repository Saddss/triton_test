import triton
import triton.language as tl
import torch

@triton.jit
def softmax_kernel(input_ptr, output_ptr, input_row_stride, 
                output_row_stride, n_cols, BLOCK_SIZE: tl.constexpr):
    
    row_idx = tl.program_id(0) # 一个块处理一行元素，idx 表示第几行，每行之间的处理是并行的
    row_start_ptr = input_ptr + row_idx * input_row_stride # # 步幅表示我们需要增加指针多少才能前进 1 行
    col_offsets = tl.arange(0 , BLOCK_SIZE) # 块大小是大于 n_cols 的下一个 2 的幂，因此我们可以将每一行放在一个块中
    input_ptrs = row_start_ptr + col_offsets 

    row = tl.load(input_ptrs, mask=col_offsets < n_cols)# using a mask since BLOCK_SIZE may be > than n_cols

    row_minus_max = row - tl.max(row, axis=0)
    numerator = tl.exp(row_minus_max)
    denominator = tl.sum(numerator, axis=0)
    softmax_output = numerator / denominator

    # 将结果行数据写入到指定地址范围中
    out_row_ptr = output_ptr + row_idx * output_row_stride
    output_ptrs = out_row_ptr + col_offsets
    tl.store(output_ptrs, softmax_output, mask=col_offsets < n_cols)

def softmax(x):
    n_rows, n_cols = x.shape
    y = torch.empty_like(x)
    BLOCK_SIZE = triton.next_power_of_2(n_cols)  # 块大小取2的幂

    # 配置每个块的warp数量
    num_warps = 4
    if BLOCK_SIZE >= 2048:
        num_warps = 8
    if BLOCK_SIZE >= 4096:
        num_warps = 16

    # 方式1：直接用元组 (n_rows,) 表示1D网格（n_rows个线程块）
    softmax_kernel[(n_rows,)]( 
        x, y,
        x.stride(0), y.stride(0),
        n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps  # 补充num_warps参数（之前遗漏了）
    )

    # 方式2（等价）：使用之前定义的grid函数
    # grid = lambda meta: (n_rows,)  # 1D网格，n_rows个线程块
    # softmax_kernel[grid](x, y, x.stride(0), y.stride(0), n_cols, BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps)

    return y

x = torch.randn((5, 5), device='cuda', dtype=torch.float32)
print(x)
print(softmax(x))
