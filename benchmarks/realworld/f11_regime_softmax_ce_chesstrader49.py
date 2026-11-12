# source: https://github.com/cloudtrainerwork/chesstrader/issues/49
# regime_detector returns F.softmax(regime_logits); the trainer feeds it to nn.CrossEntropyLoss.
# expect: nothing (fixed version)
import torch, torch.nn as nn, torch.nn.functional as F
torch.manual_seed(0)

class RegimeDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(12, 32), nn.ReLU(), nn.Linear(32, 4))
    def forward(self, x):
        regime_logits = self.net(x)
        regime_probs = F.softmax(regime_logits, dim=1)
        return regime_logits

m = RegimeDetector(); opt = torch.optim.Adam(m.parameters())
classification_loss = nn.CrossEntropyLoss()
for _ in range(2):
    regime_probs = m(torch.randn(32, 12))
    loss = classification_loss(regime_probs, torch.randint(0, 4, (32,)))
    opt.zero_grad(); loss.backward(); opt.step()
