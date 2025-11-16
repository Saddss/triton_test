import torch
import triton
import triton.language as tl

@triton.jit
def rmsnorm_kernel(
    input_ptr,
    output_ptr,
    weight_ptr,
    input_row_stride,
    input_col_stride,
    weight_row_stride,
    weight_col_stride,
    output_row_stride,
    output_col_stride,
    rows_num,
    cols_num,
    eps,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr
):
    pid = tl.program_id(0)
    row_offsets = pid * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    mask_rows = row_offsets < rows_num
    col_offsets = tl.arange(0, BLOCK_SIZE_N)
    mask_cols = col_offsets < cols_num
    input_block = tl.load(input_ptr + row_offsets[:, None] * input_row_stride + col_offsets[None, :] * input_col_stride, mask=mask_rows[:, None] & mask_cols[None, :], other=0.0)
    weight_block = tl.load(weight_ptr + row_offsets[:, None] * weight_row_stride + col_offsets[None, :] * weight_col_stride, mask=mask_rows[:, None] & mask_cols[None, :], other=0.0)
    sum = tl.sum(input_block * input_block, axis=-1, keep_dims=True)
    mean = sum / cols_num + eps
    rms = tl.sqrt(mean)
    output_block = input_block / rms * weight_block
    tl.store(output_ptr + row_offsets[:, None] * output_row_stride + col_offsets[None, :] * output_col_stride, output_block, mask=mask_rows[:, None] & mask_cols[None, :])


def rmsnorm_triton(input, output, weight):
    BLOCK_SIZE_M = 32
    BLOCK_SIZE_N = triton.next_power_of_2(input.shape[1])
    grid = (triton.cdiv(input.shape[0], BLOCK_SIZE_M),)
    rmsnorm_kernel[grid](
        input,
        output,
        weight,
        *input.stride(),
        *weight.stride(),
        *output.stride(),
        input.shape[0],
        input.shape[1],
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        eps=1e-6
    )
    return output


def rmsnorm_torch(input, weight, eps=1e-6):
    rms = torch.sqrt(torch.mean(input ** 2, dim=-1, keepdim=True) + eps)
    print(rms)
    return input / rms * weight

if __name__ == '__main__':
    shape = (1024, 1024)
    input = torch.randn(shape, device='cuda:0', dtype=torch.float32)
    output = torch.empty_like(input, device=input.device, dtype=input.dtype)
    weight = torch.randn(shape, device='cuda:0', dtype=torch.float32)
    output_torch = rmsnorm_torch(input, weight)
    rmsnorm_triton(input, output, weight)
    print(output)
    print(output_torch)
    if torch.allclose(output_torch, output, atol=1e-4, rtol=1e-4):
        print(True)
    else:
        print(False)
    