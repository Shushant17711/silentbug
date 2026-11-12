# source: https://github.com/pytorch/torchtitan/issues/4802
# The sparse-attention indexer scores keys and returns only the int64 top-k indices; the loss
# uses values[indices], so idx_q / idx_k / idx_w never get a gradient.
# expect: dead-param idx_q
import torch, torch.nn as nn
torch.manual_seed(0)

class Indexer(nn.Module):
    def __init__(self, d=8):
        super().__init__()
        self.idx_q, self.idx_k, self.idx_w = nn.Linear(d, d), nn.Linear(d, d), nn.Linear(d, 1)
    def select(self, x, topk=2):
        scores = (self.idx_q(x) @ self.idx_k(x).transpose(-1, -2)) * self.idx_w(x)
        return scores.topk(topk, dim=-1).indices  # BUG

class Model(nn.Module):
    def __init__(self, d=8):
        super().__init__()
        self.indexer = Indexer(d)
        self.values = nn.Linear(d, d)
    def forward(self, x):
        v = self.values(x)
        idx = self.indexer.select(x)
        return torch.gather(v, 1, idx[..., :1].expand(-1, -1, v.shape[-1]))

m = Model(); opt = torch.optim.Adam(m.parameters())
for _ in range(2):
    loss = m(torch.randn(2, 16, 8)).square().sum()
    opt.zero_grad(); loss.backward(); opt.step()
