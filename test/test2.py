import torch
from torch.autograd.functional import jacobian
import torch.nn as nn


class FCN(nn.Module):
    "define a fully connected network"

    def __init__(self, N_INPUT, N_OUTPUT, N_HIDDEN, N_LAYERS):
        super(FCN, self).__init__()
        activation = nn.Tanh
        self.fcs = nn.Sequential(*[nn.Linear(N_INPUT, N_HIDDEN), activation()])
        self.fch = nn.Sequential(
            *[
                nn.Sequential(*[nn.Linear(N_HIDDEN, N_HIDDEN), activation()])
                for _ in range(N_LAYERS)
            ]
        )
        self.fce = nn.Linear(N_HIDDEN, N_OUTPUT)

    def forward(self, x):
        x = self.fcs(x)
        x = self.fch(x)
        x = self.fce(x)
        return x


def exp_reducer(x):
    return x.exp()


f = FCN(1, 2, 16, 2)

inputs = torch.rand(10, 1)

print(inputs, f(inputs), jacobian(f, inputs), sep="\n")

from torch.func import jacrev, vmap

x = torch.randn(64, 5)
jacobian = vmap(jacrev(torch.sin))(x)
print(x, torch.sin(x), jacobian, sep="\n")
assert jacobian.shape == (64, 5, 5)

x = inputs
jacobian = vmap(jacrev(f))(x)
print(x, f(x), jacobian, sep="\n")
assert jacobian.shape == (10, 2, 1)
