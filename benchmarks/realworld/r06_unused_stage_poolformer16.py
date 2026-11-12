# source: https://github.com/sail-sg/poolformer/issues/16
# With fork_feat the backbone adds norm0 on stage 0, but the detector's FPN starts at level 1,
# so outs[0] is thrown away and norm0 never gets a gradient.
# expect: dead-param norm0
import torch, torch.nn as nn
torch.manual_seed(0)

class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.stages = nn.ModuleList([nn.Conv2d(3 if i == 0 else 8, 8, 3, padding=1, stride=2) for i in range(4)])
        self.norm0, self.norm1 = nn.GroupNorm(1, 8), nn.GroupNorm(1, 8)
        self.norm2, self.norm3 = nn.GroupNorm(1, 8), nn.GroupNorm(1, 8)
    def forward(self, x):
        outs = []
        for i, st in enumerate(self.stages):
            x = st(x)
            outs.append(getattr(self, f"norm{i}")(x))  # BUG
        return outs

class Detector(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = Backbone()
        self.fpn = nn.ModuleList([nn.Conv2d(8, 4, 1) for _ in range(3)])
        self.fpn_start_level = 1
    def forward(self, x):
        feats = self.backbone(x)[self.fpn_start_level:]
        return sum(f(o).mean() for f, o in zip(self.fpn, feats))

m = Detector(); opt = torch.optim.SGD(m.parameters(), 0.01)
for _ in range(2):
    loss = m(torch.randn(2, 3, 32, 32))
    opt.zero_grad(); loss.backward(); opt.step()
