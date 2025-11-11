import torch
import torch.nn as nn

def rmsnorm_torch(x, weight, eps=1e-6):
    """
    RMSNorm的PyTorch实现
    
    Args:
        x: 输入张量，形状为 (..., d)
        weight: 可学习的缩放参数，形状为 (d,)
        eps: 防止除零的小常数，默认1e-6
    
    Returns:
        归一化后的张量
    """
    # 计算RMS（均方根）
    # 对最后一个维度求均方根
    rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + eps)
    # 归一化并应用权重
    output = (x / rms) * weight
    return output


class RMSNorm(nn.Module):
    """
    RMSNorm的PyTorch模块实现
    """
    def __init__(self, dim, eps=1e-6):
        """
        Args:
            dim: 归一化的维度大小
            eps: 防止除零的小常数，默认1e-6
        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x):
        """
        Args:
            x: 输入张量，形状为 (..., d)
        
        Returns:
            归一化后的张量
        """
        # 计算RMS
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        # 归一化并应用权重
        output = (x / rms) * self.weight
        return output


if __name__ == '__main__':
    # 测试函数实现
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = 4
    seq_len = 128
    dim = 512
    
    x = torch.randn(batch_size, seq_len, dim, device=device, dtype=torch.float32)
    weight = torch.ones(dim, device=device, dtype=torch.float32)
    
    # 测试函数版本
    output_func = rmsnorm_torch(x, weight)
    print(f"函数实现输出形状: {output_func.shape}")
    
    # 测试模块版本
    rms_norm = RMSNorm(dim).to(device)
    output_module = rms_norm(x)
    print(f"模块实现输出形状: {output_module.shape}")
    
    # 验证两种实现结果一致
    if torch.allclose(output_func, output_module, atol=1e-5):
        print("✓ 函数实现和模块实现结果一致")
    else:
        print("✗ 函数实现和模块实现结果不一致")
    
    # 验证归一化效果（每行的RMS应该接近1）
    rms_check = torch.sqrt(torch.mean((output_func / weight) ** 2, dim=-1))
    print(f"归一化后的RMS范围: [{rms_check.min().item():.6f}, {rms_check.max().item():.6f}]")

