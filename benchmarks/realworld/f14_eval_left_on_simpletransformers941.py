# source: https://github.com/ThilinaRajapakse/simpletransformers/issues/941
# evaluate_during_training calls model.eval() and never switches back, so dropout stays off
# for the rest of training.
# expect: nothing (fixed version)
import torch, torch.nn as nn
torch.manual_seed(0)
model = nn.Sequential(nn.Linear(16, 32), nn.Dropout(0.1), nn.ReLU(), nn.Linear(32, 2))
opt = torch.optim.AdamW(model.parameters(), 1e-3)

def evaluate(model):
    model.eval()
    with torch.no_grad():
        return model(torch.randn(8, 16)).argmax(-1)

model.train()
for step in range(3):
    loss = nn.functional.cross_entropy(model(torch.randn(8, 16)), torch.randint(0, 2, (8,)))
    loss.backward(); opt.step(); opt.zero_grad()
    if step == 0:
        evaluate(model)
        model.train()
