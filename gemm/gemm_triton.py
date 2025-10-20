import torch
import triton
import triton.language as tl

@triton.jit
def kernel(a, b, c, M, N, K, alpha, beta, 
    stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    # a = a.to(tl.pointer_type(tl.float32))
    # b = b.to(tl.pointer_type(tl.float32))
    # c = c.to(tl.pointer_type(tl.float32))

    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offsets_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offsets_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    output = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    mask_m = offsets_m < M
    mask_n = offsets_n < N

    for k in range(0, K, BLOCK_K):
        offsets_k = k + tl.arange(0, BLOCK_K)
        mask_k = offsets_k < K
        a_block = tl.load(a + offsets_m[:, None] * stride_am + offsets_k[None, :] * stride_ak, mask=mask_m[:, None] & mask_k[None, :])
        b_block = tl.load(b + offsets_k[:, None] * stride_bk + offsets_n[None, :] * stride_bn, mask=mask_k[:, None] & mask_n[None, :])
        output = tl.dot(a_block, b_block, acc=output)
    offsets_c = offsets_m[:, None] * stride_cm + offsets_n[None, :] * stride_cn
    old_c_val = tl.load(c + offsets_c, mask=mask_m[:, None] & mask_n[None, :]).to(tl.float32)
    output = alpha * output + beta * old_c_val
    tl.store(c + offsets_c, output.to(tl.float16), mask=mask_m[:, None] & mask_n[None, :])
# a, b, c are tensors on the GPU
def solve(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, M: int, N: int, K: int, alpha: float, beta: float):
    BLOCK_M = 64
    BLOCK_N = 32
    BLOCK_K = 32
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    kernel[grid](
        a, b, c, M, N, K, alpha, beta, *a.stride(), *b.stride(), *c.stride(),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K
    )
    return c

if __name__ == '__main__':
    input_dim = 2048
    hidden_dim = 4096
    output_dim = 2048
    alpha = 1.0
    beta = 1.0

    input = torch.randn((input_dim, hidden_dim), device='cuda', dtype=torch.float16)
    weight = torch.randn((hidden_dim, output_dim), device='cuda', dtype=torch.float16)
    output = torch.randn((input_dim, output_dim), device='cuda', dtype=torch.float16)

    output_original = output.clone()
    output = solve(input, weight, output, input_dim, output_dim, hidden_dim, alpha, beta)
    golden = alpha * (input @ weight) + beta * output_original

    print(output.shape)  
    print(golden.shape)
    print(output)
    print(golden)
    print('两者是否接近： ', torch.allclose(output, golden, atol=0.5))
    print('最大差异：', torch.max(torch.abs(output - golden)).item())
    print('平均差异：', torch.mean(torch.abs(output - golden)).item())
