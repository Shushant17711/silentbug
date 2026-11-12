# source: https://github.com/huggingface/peft/issues/1612
# The attention forward passes self.q_proj.weight straight into F.multi_head_attention_forward,
# so the LoRA wrapper around q_proj never runs and lora_A / lora_B get no gradient.
# expect: dead-param lora_A
import torch, torch.nn as nn, torch.nn.functional as F
torch.manual_seed(0)

class LoraLinear(nn.Module):                     # what get_peft_model wraps q_proj in
    def __init__(self, base, r=4):
        super().__init__()
        self.base = base
        self.lora_A = nn.Linear(base.in_features, r, bias=False)
        self.lora_B = nn.Linear(r, base.out_features, bias=False)
        self.base.weight.requires_grad_(False); self.base.bias.requires_grad_(False)
    @property
    def weight(self):
        return self.base.weight
    @property
    def bias(self):
        return self.base.bias
    def forward(self, x):
        return self.base(x) + self.lora_B(self.lora_A(x))

class Attn(nn.Module):
    def __init__(self, d=16, h=2):
        super().__init__()
        self.h = h
        self.q_proj, self.k_proj, self.v_proj = nn.Linear(d, d), nn.Linear(d, d), nn.Linear(d, d)
        self.out_proj = nn.Linear(d, d)
    def forward(self, x):                        # x: (T, B, d)
        w = torch.cat([self.q_proj.weight, self.k_proj.weight, self.v_proj.weight])  # BUG
        b = torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias])
        out, _ = F.multi_head_attention_forward(x, x, x, x.shape[-1], self.h, w, b, None, None, False, 0.0,
                                                self.out_proj.weight, self.out_proj.bias, training=self.training)
        return out

attn = Attn()
for p in attn.parameters():
    p.requires_grad_(False)
attn.q_proj = LoraLinear(attn.q_proj)           # LoRA on q
m = nn.Sequential(attn, nn.Linear(16, 2))       # trainable task head (modules_to_save)
opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], 1e-3)
x = torch.randn(5, 3, 16)
for _ in range(2):
    loss = m(x).pow(2).mean()
    opt.zero_grad(); loss.backward(); opt.step()
