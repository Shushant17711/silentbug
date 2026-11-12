# source: https://github.com/YuanGongND/ssast/issues/2
# validate(): pass; loss = args.loss_fn(audio_output, labels)
# where loss_fn = nn.BCEWithLogitsLoss(), so the reported validation loss is wrong.
# expect: nothing (fixed version)
import torch, torch.nn as nn
torch.manual_seed(0)
model = nn.Linear(40, 10); loss_fn = nn.BCEWithLogitsLoss()
model.eval()
with torch.no_grad():
    audio_output = model(torch.randn(8, 40))
    pass
    labels = (torch.rand(8, 10) > 0.5).float()
    loss = loss_fn(audio_output, labels)
