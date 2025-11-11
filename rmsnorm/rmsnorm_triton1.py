import torch
import triton
import triton.language as tl

@triton.jit
def rmsnorm_kernel(
    input_ptr,
    output_ptr,
    weight_ptr,
    input_row_stride,
    weight_row_stride,
    output_row_stride,
    cols_num,
    eps,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    input_start_ptr = input_ptr + pid * input_row_stride
    weight_start_ptr = weight_ptr + pid * weight_row_stride
    output_start_ptr = output_ptr + pid * output_row_stride
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < cols_num
    input_block = tl.load(input_start_ptr + offsets, mask=mask, other=0.0)
    weight_block = tl.load(weight_start_ptr + offsets, mask=mask, other=0.0)
    sum = tl.sum(input_block * input_block, axis=-1)
    mean = sum / cols_num + eps
    rms = tl.sqrt(mean)
    output_block = input_block / rms * weight_block
    tl.store(output_start_ptr + offsets, output_block, mask=mask)


def rmsnorm_triton(input, output, weight):
    BLOCK_SIZE = input.shape[1]
    grid = (input.shape[0],)
    rmsnorm_kernel[grid](
        input,
        output,
        weight,
        input.stride(0),
        weight.stride(0),
        output.stride(0),
        input.shape[1],
        BLOCK_SIZE=BLOCK_SIZE,
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
    