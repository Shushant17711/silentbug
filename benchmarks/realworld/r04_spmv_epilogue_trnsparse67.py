# source: https://github.com/trnsci/trnsparse/issues/67
# spmv(A, x, y=y, beta=1): A (12,10) @ x (10,) is (12,); y passed as (12,1) makes the result (12,12).
# expect: broadcast
import torch
torch.manual_seed(0)

def spmv(A, x, y=None, alpha=1.0, beta=0.0):
    out = alpha * (A @ x)
    if y is not None:
        out = out + beta * y  # BUG
    return out

A = torch.randn(12, 10).to_sparse()
x = torch.randn(10, requires_grad=True)
y = torch.randn(12, 1)
r = spmv(A, x, y=y, beta=1.0)
r.sum().backward()
