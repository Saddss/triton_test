import torch

def naive_softmax(input):
    input = input.to(torch.float32)
    # output = torch.softmax(input, dim=-1)
    input_max = torch.max(input, dim=-1)[0]
    input_sub_max = input - input_max[:, None]
    exp_input_sub_max = torch.exp(input_sub_max)
    output = exp_input_sub_max / torch.sum(exp_input_sub_max, dim=-1)[:, None]
    return output

if __name__ == '__main__':
    input = torch.tensor([[1, 1, 1], [2, 2, 2]])
    print(naive_softmax(input))