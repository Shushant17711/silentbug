# source: https://github.com/kornia/kornia/issues/6157
# MS-SSIM-style loss only checks ndim, so a batch-1 target is broadcast across the batch
# inside the library's internal F.l1_loss.
# expect: nothing (fixed version)
import torch, torch.nn as nn, torch.nn.functional as F
torch.manual_seed(0)

def ms_ssim_like_loss(img1, img2):
    if img1.shape != img2.shape:                # the only check the library did
        raise ValueError("dims differ")
    l1 = F.l1_loss(img1, img2, reduction="none")
    return l1.mean()

net = nn.Conv2d(3, 3, 3, padding=1)
opt = torch.optim.SGD(net.parameters(), 0.01)
x = torch.rand(2, 3, 32, 32)
ref = torch.rand(2, 3, 32, 32)
for _ in range(2):
    loss = ms_ssim_like_loss(net(x), ref)
    opt.zero_grad(); loss.backward(); opt.step()
