# source: https://github.com/RAI-SCC/jigsaw/issues/224
# reshard_parameters assigns fresh nn.Parameters after the optimizer was built, so the optimizer
# keeps stepping the old tensors and the layer stops training.
# expect: nothing (fixed version)
import torch, torch.nn as nn
torch.manual_seed(0)

class ShardedLinear(nn.Module):
    def __init__(self, i, o):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(o, i) * 0.1)
        self.bias = nn.Parameter(torch.zeros(o))
    def reshard_parameters(self):
        self.weight = nn.Parameter(self.weight.detach().clone())
        self.bias = nn.Parameter(self.bias.detach().clone())
    def forward(self, x):
        return x @ self.weight.t() + self.bias

m = nn.Sequential(ShardedLinear(4, 8), nn.ReLU(), nn.Linear(8, 1))
opt = torch.optim.SGD(m.parameters(), 0.1)
m[0].reshard_parameters()
opt = torch.optim.SGD(m.parameters(), 0.1)
for _ in range(2):
    loss = m(torch.randn(16, 4)).pow(2).mean()
    opt.zero_grad(); loss.backward(); opt.step()
