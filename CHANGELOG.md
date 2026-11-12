# Changelog

## v0.1.0 (2026-11-12)

First release.

**Finds**

- `broadcast`: shapes that silently broadcast, such as an `(N,1)` output against an `(N,)` target,
  in arithmetic and in loss functions.
- `dead-param`: parameters that never get a gradient, with the cause and line: `.detach()`, `.data`,
  `.item()` / NumPy round trips, `torch.no_grad()`, argmax/topk, a module that is never called
  (such as a bypassed LoRA wrapper), or an output that never reaches the loss.
- `zero-grad`: gradients that exist but are always zero (`round`, `sign`, `floor`, ...).
- `not-in-optimizer` and `stale-optimizer`: layers created after the optimizer, or parameters
  replaced after it was built.
- `double-activation`: softmax or sigmoid fed into a loss that applies it again.
- `eval-mode`, `grad-not-cleared` (also when weights are updated by hand) and `batchnorm-batch1`.

**Runs on**

- CPU and CUDA, fp16/bf16 autocast with `GradScaler`, `torch.compile`, and DDP (`torchrun`; only
  rank 0 prints).
- `silentbug run train.py` with no code changes, or `with silentbug.check():` in code and tests.

**Tested**

- 15 reproducers of silent bugs from public issues, each found at the right line, plus 15 fixed
  versions that report nothing (`benchmarks/realworld/`).
- No false alarms on karpathy/nanoGPT and pytorch/examples `word_language_model`.
