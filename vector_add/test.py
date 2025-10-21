from numpy import False_
import torch
import triton
import triton.language as tl
from vector_add_triton import vector_add_triton

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['size'],  # 用作图表x轴的参数名。
        x_vals=[2**i for i in range(12, 28, 1)],  # `x_name`的不同可能值。
        x_log=True,  # x轴是对数的。取False会让左侧挤在一起，无法分辨
        line_arg='provider',  # 其值对应图表中不同线条的参数名。
        line_vals=['triton', 'torch'],  # `line_arg`的可能值。
        line_names=['Triton', 'Torch'],  # 线条的标签名。
        styles=[('blue', '-'), ('green', '-')],  # 线条样式。
        ylabel='GB/s',  # y轴的标签名。
        plot_name='vector-add-performance',  # 图表的名称。也用作保存图表的文件名。
        args={},  # 不在`x_names`和`y_name`中的函数参数值。
    ))
def benchmark(size, provider):
    x = torch.rand(size, device='cuda', dtype=torch.float32)
    y = torch.rand(size, device='cuda', dtype=torch.float32)
    quantiles = [0.5, 0.2, 0.8]
    if provider == 'torch':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: x + y, quantiles=quantiles)
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: vector_add_triton(x, y), quantiles=quantiles)
    gbps = lambda ms: 12 * size / ms * 1e-6
    return gbps(ms), gbps(max_ms), gbps(min_ms)

if __name__ == '__main__':
    benchmark.run(print_data=True, show_plots=True, save_path='./output')
