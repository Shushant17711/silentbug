# source: https://github.com/pytorch/pytorch/issues/16045
# Linear regression: model output (n,1) vs target (n,) in nn.MSELoss broadcasts to (n,n).
# expect: nothing (fixed version)
import torch, torch.nn as nn
torch.manual_seed(0)
X = torch.randn(100, 2)
y = X @ torch.tensor([2.0, -3.0]) + 1.0          # shape (100,)
model = nn.Linear(2, 1)
opt = torch.optim.SGD(model.parameters(), lr=0.1)
crit = nn.MSELoss()
for _ in range(3):
    yhat = model(X)                              # shape (100, 1)
    loss = crit(yhat.squeeze(-1), y)
    opt.zero_grad(); loss.backward(); opt.step()
