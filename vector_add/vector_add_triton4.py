import torch
import triton
import triton.language as tl

@triton.jit
def vector_add_kernel(a_ptr, b_ptr, c_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE) + pid * 16
    mask = offsets < N
    a = tl.load(a_ptr + offsets)
    b = tl.load(b_ptr + offsets)
    c = a + b
    tl.store(c_ptr + offsets, c)

def vector_add_triton(a, b, c, N):
    BLOCK_SIZE = 16
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    vector_add_kernel[grid](a, b, c, N, BLOCK_SIZE=BLOCK_SIZE)
    return c


if __name__ == '__main__':
    N = 12345
    a = torch.randn(N, device='cuda:0')
    b = torch.randn(N, device='cuda:0')
    torch_output = a + b
    c = torch.empty_like(a)
    triton_output = vector_add_triton(a, b, c, N)
    if torch.allclose(torch_output, triton_output):
        print(True)
    else:
        print(False)    