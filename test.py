import torch
import triton
import triton.language as tl

@triton.jit
def softmax_kernel(
    input, output,
    N,
    BLOCK_SIZE: tl.constexpr
):
    input = input.to(tl.pointer_type(tl.float32))
    output = output.to(tl.pointer_type(tl.float32))

    m = tl.full([1], float('-inf'), tl.float32)
    d = tl.zeros([1], tl.float32)

    offsets = tl.arange(0, BLOCK_SIZE)
    tiles = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    for i in range(tiles):
        idx = i * BLOCK_SIZE + offsets
        mask = idx < N
        val = tl.load(input + idx, mask=mask, other=-float('inf'))
        tile_max = tl.max(val)
        m_new = tl.maximum(m, tile_max)
        d = d * tl.exp(m - m_new) + tl.sum(tl.exp(val - m_new))
        m = m_new

    for i in range(tiles):
        idx = i * BLOCK_SIZE + offsets
        mask = idx < N
        val = tl.load(input + idx, mask=mask, other=-float('inf'))
        e = tl.exp(val - m) / d
        tl.store(output + idx, e, mask=mask)

# input, output are tensors on the GPU
def solve(input: torch.Tensor, output: torch.Tensor, N: int):
    BLOCK_SIZE = 1024
    grid = (1,)
    softmax_kernel[grid](input, output, N, BLOCK_SIZE=BLOCK_SIZE)