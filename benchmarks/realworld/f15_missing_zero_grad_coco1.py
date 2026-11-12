# source: https://github.com/juweipku/CoCo/issues/1
# train.py: loss.backward(); optimizer.step() with no optimizer.zero_grad().
# expect: nothing (fixed version)
import torch, torch.nn as nn
torch.manual_seed(0)
model = nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 1))
optimizer = torch.optim.Adam(model.parameters(), 1e-3)
for epoch in range(3):
    loss = model(torch.randn(16, 10)).pow(2).mean()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
