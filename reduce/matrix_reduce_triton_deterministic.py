import time

import torch
import triton
import triton.language as tl


@triton.jit
def matrix_reduce_partial_kernel(
    input_ptr,
    partials_ptr,
    M,
    N,
    stride_am,
    stride_an,
    grid_n,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    ACC_DTYPE: tl.constexpr,
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
    block_sum = block_sum.to(ACC_DTYPE)

    linear_idx = pid_m * grid_n + pid_n
    tl.store(partials_ptr + linear_idx, block_sum)


@triton.jit
def reduce_stage_kernel(
    input_ptr,
    output_ptr,
    num_values,
    BLOCK_SIZE: tl.constexpr,
    ACC_DTYPE: tl.constexpr,
):
    pid = tl.program_id(0)
    start = pid * BLOCK_SIZE
    offsets = start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_values

    vals = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    vals = vals.to(ACC_DTYPE)
    partial = tl.sum(vals, axis=0)

    tl.store(output_ptr + pid, partial)


def matrix_reduce_sum_triton_deterministic(
    x: torch.Tensor,
    block_m: int = 128,
    block_n: int = 128,
    stage_block_size: int = 1024,
):
    assert x.is_cuda, "输入张量必须在 GPU 上"
    assert x.ndim == 2, "仅支持二维矩阵规约"
    assert x.dtype in (
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
    ), "暂仅支持浮点类型"

    M, N = x.shape
    grid_m = triton.cdiv(M, block_m)
    grid_n = triton.cdiv(N, block_n)

    acc_dtype = torch.float64 if x.dtype == torch.float64 else torch.float32

    partials = torch.empty(grid_m * grid_n, device=x.device, dtype=acc_dtype)

    grid = (grid_m, grid_n)
    matrix_reduce_partial_kernel[grid](
        x,
        partials,
        M,
        N,
        x.stride(0),
        x.stride(1),
        grid_n,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        ACC_DTYPE=tl.float64 if acc_dtype == torch.float64 else tl.float32,
    )

    current = partials
    num_values = current.numel()

    while num_values > 1:
        next_num = triton.cdiv(num_values, stage_block_size)
        next_buffer = torch.empty(next_num, device=x.device, dtype=acc_dtype)

        reduce_stage_kernel[(next_num,)](
            current,
            next_buffer,
            num_values,
            BLOCK_SIZE=stage_block_size,
            ACC_DTYPE=tl.float64 if acc_dtype == torch.float64 else tl.float32,
        )

        current = next_buffer
        num_values = current.numel()

    result = current.reshape(())
    if result.dtype != x.dtype:
        result = result.to(dtype=x.dtype)

    return result


if __name__ == "__main__":
    torch.manual_seed(0)
    M, N = 4096, 3072
    x = torch.randn((M, N), device="cuda", dtype=torch.float32)

    for _ in range(5):
        matrix_reduce_sum_triton_deterministic(x)

    torch.cuda.synchronize()
    start = time.time()
    triton_res = matrix_reduce_sum_triton_deterministic(x)
    torch.cuda.synchronize()
    triton_time = time.time() - start

    torch.cuda.synchronize()
    start = time.time()
    torch_res = torch.sum(x)
    torch.cuda.synchronize()
    torch_time = time.time() - start

    if torch.allclose(triton_res, torch_res):
        print("Triton 确定性规约成功！")
    else:
        print("Triton 确定性规约失败！")

    print(f"Triton 规约时间: {triton_time * 1000:.3f} ms")
    print(f"PyTorch 规约时间: {torch_time * 1000:.3f} ms")

