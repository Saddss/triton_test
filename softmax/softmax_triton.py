import torch
import triton
import triton.language as tl

@triton.jit
def softmax_kernel(input_ptr, output_ptr, stride_input_row, stride_output_row, BLOCK_SIZE: tl.constexpr):
    input_ptr = input_ptr.to(tl.pointer_type(tl.float32))
    # output_ptr = output_ptr.to(tl.pointer_type(tl.float32))

    pid = tl.program_id(0)
    block_start = pid * stride_input_row
    tiles = (stride_input_row + BLOCK_SIZE - 1) // BLOCK_SIZE

    m = tl.full([1], float('-inf'), tl.float32)
    d = tl.zeros([1], tl.float32)

    for i in range(tiles):
        tile_start = i * BLOCK_SIZE
        offsets = tile_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < stride_input_row
        x = tl.load(input_ptr + block_start + offsets, mask=mask, other=float('-inf'))
        tile_max = tl.max(x, axis=-1)
        m_new = tl.maximum(m, tile_max)
        d = d * tl.exp(m - m_new) + tl.sum(tl.exp(x - m_new), axis=-1)
        m = m_new

    for i in range(tiles):
        tile_start = i * BLOCK_SIZE
        offsets = tile_start + tl.arange(0, BLOCK_SIZE)
        input_mask = offsets < stride_input_row
        x = tl.load(input_ptr + block_start + offsets, mask=input_mask, other=float('-inf'))
        exp_x = tl.exp(x - m) / d
        output_mask = offsets < stride_output_row
        tl.store(output_ptr + block_start + offsets, exp_x.to(tl.float16), mask=output_mask)

def softmax_triton(input, output, stride_input_row, stride_output_row):
    BLOCK_SIZE = 2048
    grid = (input.shape[0],)
    softmax_kernel[grid](input, output, stride_input_row, stride_output_row, BLOCK_SIZE=BLOCK_SIZE)

if __name__ == '__main__':
    input = torch.randn((4096, 8192), device='cuda:0', dtype=torch.float16)
    output = torch.empty_like(input, device=input.device, dtype=input.dtype)
    output_torch = torch.softmax(input, dim=-1)
    softmax_triton(input, output, input.stride(0), output.stride(0))
    if torch.allclose(output, output_torch, rtol=1e-5):
        print(True)
    else:
        print(False)