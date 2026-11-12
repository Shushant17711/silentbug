# source: https://github.com/YuanGongND/ssast/issues/2
# validate(): audio_output = torch.sigmoid(audio_output); loss = args.loss_fn(audio_output, labels)
# where loss_fn = nn.BCEWithLogitsLoss(), so the reported validation loss is wrong.
# expect: double-activation
import torch, torch.nn as nn
torch.manual_seed(0)
model = nn.Linear(40, 10); loss_fn = nn.BCEWithLogitsLoss()
model.eval()
with torch.no_grad():
    audio_output = model(torch.randn(8, 40))
    audio_output = torch.sigmoid(audio_output)
    labels = (torch.rand(8, 10) > 0.5).float()
    loss = loss_fn(audio_output, labels)  # BUG
