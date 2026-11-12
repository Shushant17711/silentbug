# source: https://github.com/OpenTabular/DeepTab/issues/441
# The datamodule unsqueezes y without flattening it first: a (n,1) target becomes (B,1,1)
# per batch while predictions are (B,1); MSELoss broadcasts to (B,B,1).
# expect: broadcast
import torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
torch.manual_seed(0)
X = torch.randn(64, 5)
y = (X.sum(1, keepdim=True))                    # user passes a (n,1) column vector
labels = y.unsqueeze(dim=1)                     # datamodule.setup(): (n,1) -> (n,1,1)
dl = DataLoader(TensorDataset(X, labels), batch_size=16)
model = nn.Sequential(nn.Linear(5, 16), nn.ReLU(), nn.Linear(16, 1))
opt = torch.optim.Adam(model.parameters(), 1e-3)
loss_fn = nn.MSELoss()
for xb, yb in dl:
    preds = model(xb)                           # (B,1)
    loss = loss_fn(preds, yb)  # BUG
    opt.zero_grad(); loss.backward(); opt.step()
