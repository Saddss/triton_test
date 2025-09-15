import triton
import triton.language as tl
import torch

@triton.jit
def vector_add_kernel(X_ptr, Y_ptr, Z_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offset = tl.arange(0, BLOCK_SIZE)
    idx = block_start + offset
    mask = idx < N

    x = tl.load(X_ptr + idx, mask=mask)
    y = tl.load(Y_ptr + idx, mask=mask)

    z = x + y

    tl.store(Z_ptr + idx, z, mask=mask)

def vector_add(X, Y):
    assert X.shape == Y.shape
    N = X.numel()
    Z = torch.zeros_like(X)
    grid = lambda META: (triton.cdiv(N, META['BLOCK_SIZE']),)
    vector_add_kernel[grid](X, Y, Z, N, BLOCK_SIZE=1024)
    return Z

if __name__ == '__main__':
    N = 10
    X = torch.randn(N, device='cuda', dtype=torch.float32)
    Y = torch.randn(N, device='cuda', dtype=torch.float32)

    Z = vector_add(X, Y)