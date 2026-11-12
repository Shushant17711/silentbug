# source: https://github.com/pytorch/pytorch/issues/14467
# The base class builds the optimizer in __init__; the subclass adds its GRU after
# super().__init__(), so the GRU is never in the optimizer.
# expect: not-in-optimizer gru
import torch, torch.nn as nn
torch.manual_seed(0)

class Base(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(20, 8)
        self.out = nn.Linear(8, 20)
        self.optimizer = torch.optim.Adadelta(self.parameters())

class Seq(Base):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(8, 8, batch_first=True)
    def forward(self, t):
        h, _ = self.gru(self.embed(t))
        return self.out(h)

m = Seq()
for _ in range(2):
    t = torch.randint(0, 20, (4, 6))
    loss = nn.functional.cross_entropy(m(t).flatten(0, 1), t.flatten())
    m.optimizer.zero_grad(); loss.backward(); m.optimizer.step()
