import torch

def orthogonal_(weight, gain):
    with torch.no_grad():
        weight.copy_(torch.nn.init.orthogonal_(torch.empty_like(weight, dtype=torch.float), gain))
