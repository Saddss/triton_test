import torch
import triton
import triton.language as tl


@triton.jit
def softmax_kernel(input_ptr, output_ptr, input_row_stride, output_row_stride, n_cols, BLOCK_SIZE: tl.constexpr):
    row_id = tl.program_id(0)
    offset = tl.arange(0, BLOCK_SIZE)
    input_ptrs = input_ptr + row_id * input_row_stride + offset
    mask = offset < n_cols

    row = tl.load(input_ptrs, mask=mask, other=-float('inf'))

    row_max = tl.max(row, axis=0)
    row_sub_max = row - row_max
    numerator = tl.exp(row_sub_max)
    denominator = tl.sum(numerator, axis=0)
    softmax_output = numerator / denominator

    output_ptrs = output_ptr + row_id * output_row_stride + offset
    tl.store(output_ptrs, softmax_output, mask=mask)


def softmax(x):
    row_num, col_num = x.shape
    BLOCK_SIZE = triton.next_power_of_2(col_num)
    y = torch.empty_like(x)

    num_warps = 4
    if BLOCK_SIZE >= 2048:
        num_warps = 8
    if BLOCK_SIZE >= 4096:
        num_warps = 16
    
    softmax_kernel[(row_num,)](
        x, y, x.stride(0), y.stride(0), col_num, BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps
    )

    return y


x = torch.tensor([[1, 1, 1, 1, 1], [2, 4, 6, 8, 10]], device='cuda', dtype=torch.float32)
print(x)
print(softmax(x))