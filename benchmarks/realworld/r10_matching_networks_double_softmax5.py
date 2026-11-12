# source: https://github.com/gitabcworld/MatchingNetworks/issues/5
# AttentionalClassify softmaxes the similarities, mixes one-hot support labels with bmm,
# and the result goes to F.cross_entropy, which softmaxes again.
# expect: double-activation
import torch, torch.nn as nn, torch.nn.functional as F
torch.manual_seed(0)
enc = nn.Linear(10, 16); opt = torch.optim.Adam(enc.parameters())
for _ in range(2):
    support, target = torch.randn(4, 5, 10), torch.randn(4, 10)
    support_y = F.one_hot(torch.arange(5).repeat(4, 1), 5).float()      # (B, K, C)
    sims = torch.einsum("bkd,bd->bk", enc(support), enc(target))
    softmax_similarities = nn.Softmax(dim=1)(sims)
    preds = softmax_similarities.unsqueeze(1).bmm(support_y).squeeze()  # (B, C)
    loss = F.cross_entropy(preds, torch.randint(0, 5, (4,)))  # BUG
    opt.zero_grad(); loss.backward(); opt.step()
