# source: https://github.com/Lightning-Universe/lightning-bolts/issues/654
# LogisticRegression: F.nll_loss(F.log_softmax(F.softmax(self.linear(x))), y)
# expect: nothing (fixed version)
import torch, torch.nn as nn, torch.nn.functional as F
torch.manual_seed(0)
linear = nn.Linear(20, 3); opt = torch.optim.SGD(linear.parameters(), 0.1)
for _ in range(2):
    x, y = torch.randn(16, 20), torch.randint(0, 3, (16,))
    loss = F.nll_loss(F.log_softmax(linear(x), dim=-1), y)
    opt.zero_grad(); loss.backward(); opt.step()
