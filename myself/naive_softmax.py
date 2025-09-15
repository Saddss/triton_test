import torch

def naive_softmax(x):
    x_max = torch.max(x, dim=1)[0]
    z = x - x_max[:, None]
    numerator = torch.exp(z)
    denominator = torch.sum(x, dim=1)
    result = numerator / denominator[:, None]
    return result
