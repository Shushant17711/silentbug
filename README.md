# silentbug

Your model trains, the loss goes down a little, and nothing errors. But one tensor is
broadcast against another, half your parameters never get a gradient, or softmax runs
twice. silentbug watches a few training steps and tells you **which** silent bug it is,
**on which line**, **why** it matters and **how to fix it**.

```
$ silentbug run train.py
silentbug: checked 3 backward pass(es), 3 optimizer step(s) on Net
  2 likely bug(s), 0 thing(s) to check

[BUG  ] 1. mse_loss got input (100, 1) and target (100,)
    at train.py:13: loss = crit(yhat, y)
    why: they broadcast to (100, 100), so each prediction is compared with many targets
         instead of its own. The loss still goes down a little, which hides the bug.
    fix: reshape one side so input and target have the same shape (often `.squeeze(-1)`)

[BUG  ] 2. 2 parameters never get a gradient: cut by detach
    at train.py:21: g = self.gate(h).detach()
    parameters: gate.bias, gate.weight
    fix: remove `.detach()` if these parameters should learn
```

## Install

```bash
pip install git+https://github.com/Shushant17711/silentbug@v0.1.0   # Python 3.10+, PyTorch 2.1+
```

From a clone, for development:

```bash
git clone https://github.com/Shushant17711/silentbug && cd silentbug
pip install -e ".[test]" && pytest tests
```

## Use it

On a script, with no code changes. It runs the first N optimizer steps (default 3), then stops:

```bash
silentbug run train.py --lr 3e-4         # your script's own arguments go after it
silentbug run --steps 10 train.py
silentbug run --json - train.py          # JSON report on stdout
silentbug run -m mypackage.train         # like python -m
```

Exit codes: 0 nothing found, 1 likely bugs found, 2 only things to check.

Or around a block of code, for example in a test:

```python
import silentbug

with silentbug.check() as report:
    for step in range(3):
        loss = loss_fn(model(x), y)
        loss.backward()
        opt.step(); opt.zero_grad()

assert not report.bugs, str(report)
```

Models and optimizers are found on their own. Pass them in (`silentbug.check(model,
optimizer=opt)`) only when the model is never called inside the block.

## What it finds

| Kind | Example | Level |
|---|---|---|
| `broadcast` | `(N,1) - (N,)` gives an `(N,N)` loss, or `mse_loss(input (B,1), target (B,))` | BUG |
| `dead-param` | parameters with no gradient, with the cause: `.detach()`, `.data`, `.item()`/NumPy round trip, `torch.no_grad()`, argmax/topk indices, a module never called (LoRA bypassed), an output that never reaches the loss | BUG |
| `zero-grad` | gradients that exist but are always 0 (`round`, `sign`, `floor`, ...) | BUG |
| `not-in-optimizer` | layers created after the optimizer was built | BUG |
| `stale-optimizer` | the optimizer still updates parameters that were replaced | BUG |
| `double-activation` | softmax fed to `cross_entropy`/`nll_loss`, sigmoid fed to `BCEWithLogits`, softmax then `log_softmax` | BUG |
| `eval-mode` | the model is still in `eval()` while it trains | BUG |
| `grad-not-cleared` | `optimizer.step()` uses gradients left over from the previous step | BUG |
| `batchnorm-batch1` | batch norm in training mode with a batch of 1 | CHECK |

BUG means almost certainly wrong. CHECK means often wrong but sometimes intended (for example an
outer product written as a broadcast).

How it works: a `TorchFunctionMode` sees every torch call. It checks the shapes going into
broadcasting ops and losses, and tags each tensor with a bitmask of the trainable parameters it
depends on. When a tensor built from parameters comes out of an op without a gradient, that line is
recorded as a cut. Gradient hooks on every parameter record which ones received a gradient, and
global module and optimizer hooks find the models and optimizers. After the run, each parameter
that never got a gradient is matched to the cut or missing call that explains it.

## Tested on real bugs

`benchmarks/realworld/` has 15 reproducers of silent bugs from public issues (PyTorch, PEFT,
torchtitan, kornia, DeepTab, poolformer, Lightning Bolts, simpletransformers and others). Each one
is a short script that keeps the buggy code pattern from the issue. Each also has a fixed version
(`f*.py`) that must produce no findings.

```bash
cd benchmarks/realworld && python score.py $(which silentbug)
```

| Case | Issue | Bug | Found at the right line |
|---|---|---|---|
| r01 | pytorch/pytorch#16045 | `MSELoss` on `(N,1)` vs `(N,)` | yes |
| r02 | OpenTabular/DeepTab#441 | `(n,1)` target unsqueezed to `(B,1,1)` | yes |
| r03 | kornia/kornia#6157 | SSIM loss broadcasts a batch-1 target | yes |
| r04 | trnsci/trnsparse#67 | `y` broadcasts the spmv output to `(12,12)` | yes |
| r05 | huggingface/peft#1612 | weights read directly, LoRA wrapper never runs | yes |
| r06 | sail-sg/poolformer#16 | stage-0 output dropped, `norm0` never trains | yes |
| r07 | pytorch/torchtitan#4802 | top-k indices cut the indexer's gradient | yes |
| r08 | pytorch/pytorch#14467 | GRU created after the optimizer | yes |
| r09 | RAI-SCC/jigsaw#224 | reshard replaces parameters, optimizer goes stale | yes |
| r10 | gitabcworld/MatchingNetworks#5 | softmax, then `cross_entropy` | yes |
| r11 | cloudtrainerwork/chesstrader#49 | softmax output fed to `CrossEntropyLoss` | yes |
| r12 | Lightning-Universe/lightning-bolts#654 | `log_softmax(softmax(x))` | yes |
| r13 | YuanGongND/ssast#2 | sigmoid fed to `BCEWithLogitsLoss` | yes |
| r14 | ThilinaRajapakse/simpletransformers#941 | `model.eval()` never undone | yes |
| r15 | juweipku/CoCo#1 | no `optimizer.zero_grad()` | yes |

Result: 15 of 15 found at the right line, 15 of 15 fixed versions report nothing.

## Overhead

6-layer MLP, width 1024, batch 256, AdamW, CPU: 30.9 ms/step plain, 33.8 ms/step under
silentbug (1.09x). It is meant to run for a few steps, not for a whole training run.

## Limits

## Tested setups

* PyTorch 2.15 on CPU and PyTorch 2.14 with CUDA 13.0 on an RTX 4060: unit tests and all 15 benchmark
  cases plus their 15 fixed versions, on both devices.
* Mixed precision (`torch.autocast` with fp16 + `GradScaler`, and bf16): bugs found, clean code stays quiet.
* `torch.compile` (a compiled model, and a compiled loss function): findings name the original module.
* DDP with 2 processes (`torchrun`, gloo): DDP's own internal detach is not reported. Only rank 0 prints.
* Real code, unchanged: karpathy/nanoGPT (CUDA, with and without `compile`, with gradient accumulation)
  and pytorch/examples `word_language_model` (manual SGD with no optimizer). Neither gave a false alarm.
  When `model.zero_grad()` was deleted from the second one, silentbug reported it at the update line.

## Limits

* Not tested with FSDP, tensor or pipeline parallelism, or on multiple GPUs at once.
* `torch.compile`: while silentbug watches, the compiler stops at silentbug's hooks and runs the
  compiled code op by op. That is why findings inside a compiled function point at the exact line
  there. It also means bugs that only happen in compiled graphs (compiler bugs) are not seen, and
  the watched steps run at eager speed. Compilation goes back to normal once the check ends.
* It only sees what runs during the steps it watches. A branch that runs only every 100 steps is
  not checked.
* `broadcast` checks the first three tensor arguments of each op.

## Related tools

* [nantrace](https://github.com/Optimus2007/nantrace) finds where NaN and Inf start. silentbug
  finds the bugs that never produce a NaN.
* [jaxtyping](https://github.com/patrick-kidger/jaxtyping) checks shapes if you annotate them.
  silentbug needs no annotations.
* `TORCH_DISTRIBUTED_DEBUG=DETAIL` names unused parameters under DDP, but not why they are unused.

## License

MIT. See [LICENSE](LICENSE).
