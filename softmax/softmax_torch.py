import torch

def naive_softmax(input):
    # output = torch.softmax(input, dim=-1)
    # torch.max(input, dim=-1, keepdim=True)[1]是对应的索引
    input_max = torch.max(input, dim=-1, keepdim=True)[0]
    input_sub_max = input - input_max
    exp_input_sub_max = torch.exp(input_sub_max)
    output = exp_input_sub_max / torch.sum(exp_input_sub_max, dim=-1, keepdim=True)
    return output

if __name__ == '__main__':
    input = torch.tensor([[1, 2, 1], [2, 1, 2]]).to(torch.float32)
    print(naive_softmax(input))
    print(torch.softmax(input, dim=-1))