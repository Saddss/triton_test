import torch
import triton
from matmul_triton_block3 import block_matmul_triton as block_matmul_triton_block
from matmul_triton_group import block_matmul_triton as block_matmul_triton_group
@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['M', 'N', 'K'],  # 用作图表x轴的参数名
        x_vals=[128 * i for i in range(2, 64)],  # `x_name`的不同可能值
        line_arg='provider',  # 对应于图表中不同线条的参数名
        line_vals=['cublas', 'triton'],  # `line_arg`的可能值
        line_names=["cuBLAS", "Triton"],  # 线条的标签名
        styles=[('green', '-'), ('blue', '-')],  # 线条样式
        ylabel="TFLOPS",  # y轴的标签名
        plot_name="matmul-performance",  # 图表的名称，也用作保存图表的文件名
        args={},  # 其他参数
    )
)
def benchmark(M, N, K, provider):
    a = torch.randn((M, K), device='cuda', dtype=torch.float16)
    b = torch.randn((K, N), device='cuda', dtype=torch.float16)
    quantiles = [0.5, 0.2, 0.8]
    if provider == 'cublas':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.matmul(a, b), quantiles=quantiles)
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: block_matmul_triton_group(a, b), quantiles=quantiles)
    perf = lambda ms: 2 * M * N * K * 1e-12 / (ms * 1e-3)
    return perf(ms), perf(max_ms), perf(min_ms)

benchmark.run(show_plots=True, print_data=True, save_path='./output')