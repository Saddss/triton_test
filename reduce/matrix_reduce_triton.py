import time

import torch
import triton
import triton.language as tl


@triton.jit
def matrix_reduce_sum_kernel(
    input_ptr,
    output_ptr,
    M,
    N,
    stride_am,
    stride_an,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    row_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    col_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    mask_rows = row_offsets < M
    mask_cols = col_offsets < N

    rows = row_offsets[:, None]
    cols = col_offsets[None, :]
    ptrs = input_ptr + rows * stride_am + cols * stride_an

    block = tl.load(ptrs, mask=mask_rows[:, None] & mask_cols[None, :], other=0.0)
    block_sum = tl.sum(block, axis=1)
    block_sum = tl.sum(block_sum, axis=0)

    tl.atomic_add(output_ptr, block_sum)


def matrix_reduce_sum_triton(x: torch.Tensor):
    assert x.is_cuda, "输入张量必须在 GPU 上"
    assert x.dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64), "暂仅支持浮点类型"

    M, N = x.shape
    result = torch.zeros((), device=x.device, dtype=x.dtype)

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]),
        triton.cdiv(N, META["BLOCK_N"]),
    )

    matrix_reduce_sum_kernel[grid](
        x,
        result,
        M,
        N,
        x.stride(0),
        x.stride(1),
        BLOCK_M=128,
        BLOCK_N=128,
    )

    return result


if __name__ == "__main__":
    torch.manual_seed(0)
    M, N = 4096, 3072
    x = torch.randn((M, N), device="cuda", dtype=torch.float32)

    for _ in range(5):
        matrix_reduce_sum_triton(x)

    torch.cuda.synchronize()
    start = time.time()
    triton_res = matrix_reduce_sum_triton(x)
    torch.cuda.synchronize()
    triton_time = time.time() - start

    torch.cuda.synchronize()
    start = time.time()
    torch_res = torch.sum(x)
    torch.cuda.synchronize()
    torch_time = time.time() - start

    if torch.allclose(triton_res, torch_res):
        print("Triton 规约求和成功！")
    else:
        print("Triton 规约求和失败！")

    print(f"Triton 规约时间: {triton_time * 1000:.3f} ms")
    print(f"PyTorch 规约时间: {torch_time * 1000:.3f} ms")

