import torch
import triton
import triton.language as tl


@triton.jit
def rmsnorm_kernel(
    x_ptr,          # 输入张量指针
    output_ptr,     # 输出张量指针
    weight_ptr,     # 权重参数指针
    x_row_stride,   # 输入张量行步长
    output_row_stride,  # 输出张量行步长
    n_cols,         # 列数（归一化维度）
    eps,            # 防止除零的小常数
    BLOCK_SIZE: tl.constexpr,  # 块大小
):
    """
    RMSNorm的Triton内核实现
    
    每个program处理一行数据（最后一个维度）
    """
    row_idx = tl.program_id(0)  # 当前处理的行索引
    row_start_ptr = x_ptr + row_idx * x_row_stride  # 当前行的起始指针
    
    # 列偏移量
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = row_start_ptr + col_offsets
    
    # 加载输入数据（使用mask处理边界情况）
    mask = col_offsets < n_cols
    x = tl.load(input_ptrs, mask=mask, other=0.0)
    
    # 计算RMS（均方根）
    # RMS = sqrt(mean(x^2) + eps)
    x_squared = x * x
    # 计算均值
    x_squared_sum = tl.sum(x_squared, axis=0)
    mean_x_squared = x_squared_sum / n_cols
    # 计算RMS
    rms = tl.sqrt(mean_x_squared + eps)
    
    # 归一化：x / rms
    normalized = x / rms
    
    # 加载权重并应用
    weight_ptrs = weight_ptr + col_offsets
    weight = tl.load(weight_ptrs, mask=mask, other=1.0)
    
    # 应用权重
    output = normalized * weight
    
    # 存储结果
    output_row_ptr = output_ptr + row_idx * output_row_stride
    output_ptrs = output_row_ptr + col_offsets
    tl.store(output_ptrs, output, mask=mask)


def rmsnorm_triton(x, weight, eps=1e-6):
    """
    RMSNorm的Triton实现
    
    Args:
        x: 输入张量，形状为 (..., d)，将被reshape为 (n_rows, d)
        weight: 可学习的缩放参数，形状为 (d,)
        eps: 防止除零的小常数，默认1e-6
    
    Returns:
        归一化后的张量，形状与输入相同
    """
    # 保存原始形状
    original_shape = x.shape
    # 计算归一化维度
    dim = x.shape[-1]
    
    # 将输入reshape为2D: (n_rows, dim)
    n_rows = x.numel() // dim
    x_2d = x.reshape(n_rows, dim)
    
    # 创建输出张量
    output = torch.empty_like(x_2d)
    
    # 确定块大小（取大于等于dim的最小2的幂）
    BLOCK_SIZE = triton.next_power_of_2(dim)
    # 限制最大块大小以避免资源问题
    if BLOCK_SIZE > 4096:
        BLOCK_SIZE = 4096
    
    # 配置warp数量
    num_warps = 4
    if BLOCK_SIZE >= 2048:
        num_warps = 8
    if BLOCK_SIZE >= 4096:
        num_warps = 16
    
    # 启动内核
    rmsnorm_kernel[(n_rows,)](
        x_2d,
        output,
        weight,
        x_2d.stride(0),
        output.stride(0),
        dim,
        eps,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )
    
    # 恢复原始形状
    output = output.reshape(original_shape)
    return output


if __name__ == '__main__':
    # 测试Triton实现
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device != 'cuda':
        print("警告: 未检测到CUDA，Triton实现需要在GPU上运行")
    
    batch_size = 4
    seq_len = 128
    dim = 512
    
    x = torch.randn(batch_size, seq_len, dim, device=device, dtype=torch.float32)
    weight = torch.ones(dim, device=device, dtype=torch.float32)
    
    # 测试Triton实现
    if device == 'cuda':
        output_triton = rmsnorm_triton(x, weight)
        print(f"Triton实现输出形状: {output_triton.shape}")
        
        # 与PyTorch实现对比
        from rmsnorm_torch import rmsnorm_torch
        output_torch = rmsnorm_torch(x, weight)
        
        # 验证结果一致性
        if torch.allclose(output_triton, output_torch, atol=1e-4, rtol=1e-4):
            print("✓ Triton实现和PyTorch实现结果一致")
        else:
            print("✗ Triton实现和PyTorch实现结果不一致")
            max_diff = (output_triton - output_torch).abs().max()
            print(f"  最大差异: {max_diff.item():.6f}")
        
        # 验证归一化效果
        # RMSNorm的作用是将输入的RMS归一化到接近1
        # output = (x / RMS) * weight，所以 (output / weight) 的RMS应该接近1.0
        rms_check = torch.sqrt(torch.mean((output_triton / weight) ** 2, dim=-1))
        print(f"归一化后的RMS范围（应该接近1.0）: [{rms_check.min().item():.6f}, {rms_check.max().item():.6f}]")
    else:
        print("跳过测试（需要CUDA支持）")

