"""Plain-language findings: what went wrong, why it matters, how to fix it."""

from __future__ import annotations

import math

from ._where import Where
from .report import BUG, CHECK, Finding


def _s(shape) -> str:
    return "(" + ", ".join(str(x) for x in shape) + ("," if len(shape) == 1 else "") + ")"


# ---------------------------------------------------------------- shapes

def broadcast(op: str, sym: str | None, sa, sb, out, same: bool, w: Where | None, c: Where | None) -> Finding:
    expr = f"a {sym} b" if sym else f"{op}(a, b)"
    big = math.prod(out)
    small = max(math.prod(sa), math.prod(sb))
    lo, hi = (sa, sb) if len(sa) < len(sb) else (sb, sa)
    title = f"{expr} broadcasts {_s(sa)} with {_s(sb)} into {_s(out)}"
    if same:
        why = (f"the shapes have different ranks, so PyTorch puts a new leading axis on {_s(lo)} and stretches both "
               f"tensors: you get {big} values instead of {small}. This is the classic pred (N,1) vs target (N,) "
               "mistake, where every row is paired with every other row. Nothing errors; the loss is just wrong.")
        level = BUG
    else:
        why = (f"both tensors get stretched ({_s(lo)} gets a new leading axis), giving {big} values from inputs of at "
               f"most {small}. That is right for an outer product, and wrong anywhere you meant to combine matching "
               "elements.")
        level = CHECK
    fix = (f"make the shapes match, e.g. `.squeeze(-1)` on the {_s(hi)} tensor or `.unsqueeze(-1)` on the {_s(lo)} one. "
           "If you do want all pairs, write the new axis yourself (`x[:, None]`, `unsqueeze`) so the intent is visible; "
           "silentbug then stays quiet.")
    return Finding("broadcast", level, title, why, fix, w, c, detail={"a": list(sa), "b": list(sb), "out": list(out)})


def loss_shape(op: str, sx, sy, so, w, c) -> Finding:
    grows = math.prod(so) > max(math.prod(sx), math.prod(sy))
    title = f"{op} got input {_s(sx)} and target {_s(sy)}"
    if grows:
        why = (f"they broadcast to {_s(so)}, so each prediction is compared with many targets instead of its own. "
               "The loss still goes down a little, which hides the bug. PyTorch only prints a UserWarning.")
    else:
        why = (f"they broadcast to {_s(so)}: one tensor is reused across a dimension of the other "
               "(for example a batch of 1 compared with every sample).")
    fix = "reshape one side so input and target have the same shape (often `.squeeze(-1)` or `.view_as(input)`)"
    return Finding("broadcast", BUG, title, why, fix, w, c, detail={"input": list(sx), "target": list(sy), "out": list(so)})


# ---------------------------------------------------------------- activations into losses

def activation_into_loss(loss: str, act: str, act_w: Where | None, w: Where | None, c: Where | None) -> Finding | None:
    at = f" (made at {act_w.short()})" if act_w else ""
    if loss == "cross_entropy" and act in ("softmax", "softmax-mix"):
        what = "softmax probabilities" if act == "softmax" else "values built from softmax probabilities"
        return Finding("double-activation", BUG, f"cross_entropy got {what}{at}",
                       "cross_entropy applies log_softmax itself, so softmax runs twice. Outputs squash toward uniform, "
                       "gradients shrink, and training stalls at a poor accuracy without any error.",
                       "pass raw logits to the loss; apply softmax only for predictions and metrics", w, c)
    if loss == "cross_entropy" and act == "sigmoid":
        return Finding("double-activation", BUG, f"cross_entropy got sigmoid outputs{at}",
                       "cross_entropy expects raw logits; squashing them into (0, 1) first caps how confident the "
                       "model can get and weakens the gradient.",
                       "pass the logits before sigmoid", w, c)
    if loss == "nll_loss" and act in ("softmax", "sigmoid", "softmax-mix"):
        return Finding("double-activation", BUG, f"nll_loss got {act} probabilities{at}, not log-probabilities",
                       "nll_loss just picks -input[target]. With probabilities the loss becomes -p instead of -log p, "
                       "which goes negative and gives tiny gradients.",
                       "use log_softmax instead of softmax, or call cross_entropy on the logits", w, c)
    if loss == "binary_cross_entropy_with_logits" and act == "sigmoid":
        return Finding("double-activation", BUG, f"binary_cross_entropy_with_logits got sigmoid outputs{at}",
                       "the loss applies sigmoid itself, so it runs twice: predictions are squeezed into (0.5, 0.73) "
                       "and the loss can never reach zero.",
                       "pass the logits, or switch to binary_cross_entropy if you must pass probabilities", w, c)
    if loss == "kl_div" and act in ("softmax", "sigmoid"):
        return Finding("double-activation", BUG, f"kl_div got {act} probabilities{at} as input",
                       "F.kl_div expects log-probabilities as its first argument; with probabilities the value "
                       "and its gradient are wrong.",
                       "use log_softmax for the input (the target stays probabilities unless log_target=True)", w, c)
    if loss in ("softmax", "log_softmax") and act == "softmax":
        return Finding("double-activation", BUG, f"{loss} applied to an output that is already softmax{at}",
                       "softmax runs twice, which squashes the distribution toward uniform and shrinks gradients.",
                       "drop the first softmax and work from the logits", w, c)
    return None


# ---------------------------------------------------------------- gradients

_CUT_TEXT = {
    "detach": ("`.detach()` here cuts the graph, so no gradient flows back to these parameters",
               "remove `.detach()` if these parameters should learn; keep it only for targets, logging, or a frozen branch"),
    "data": ("`.data` gives a tensor with no gradient history, so everything computed from it is cut off",
             "use the tensor itself (or `.detach()` where you really mean to stop gradients)"),
    "value": ("`.{op}()` turns the tensor into a Python number or NumPy array; anything rebuilt from it has no gradient",
              "keep the math in torch ops on tensors instead of going through Python numbers or NumPy"),
    "new_tensor": ("`{op}(...)` makes a brand-new tensor with no gradient history",
                   "use the existing tensor (or torch.stack/torch.cat to combine tensors) instead of building a new one"),
    "no_grad": ("this line runs with gradients turned off (inside torch.no_grad() or a @torch.no_grad() function)",
                "move this computation out of the no_grad block if its parameters should learn"),
    "inference_mode": ("this line runs inside torch.inference_mode(), which records no gradients",
                       "move this computation out of inference_mode if its parameters should learn"),
    "requires_grad_false": ("`requires_grad_(False)` is called on a tensor computed from these parameters",
                            "remove the requires_grad_(False) call"),
    "int": ("`{op}` returns integer or boolean values (indices, comparisons, casts), which carry no gradient",
            "use a differentiable path: return the scores as well as the indices and add them to the loss, use soft "
            "weights (softmax) instead of argmax/topk indices, or a straight-through estimator"),
    "other": ("`{op}` returns a tensor with no gradient history",
              "check whether this op is differentiable; if not, route the loss through a differentiable alternative"),
}


def cut(params: list[str], ev) -> Finding:
    why, fix = _CUT_TEXT.get(ev.kind, _CUT_TEXT["other"])
    op = ev.op.replace("attr:", "")
    why = why.format(op=op)
    n = len(params)
    title = f"{n} parameter{'s' if n != 1 else ''} never get a gradient: cut by {op}"
    return Finding("dead-param", BUG, title, why + ". They will never train, and nothing errors (unless you use DDP).",
                   fix, ev.where, ev.caller, sorted(params), {"cause": ev.kind, "op": op})


def unused(params: list[str], module: str, called: bool, has_opt: bool,
           w: Where | None = None, c: Where | None = None) -> Finding:
    n = len(params)
    mod = f"`{module}`" if module else "the model"
    if module and not called:
        title = f"{n} parameter{'s' if n != 1 else ''} never get a gradient: {mod} is never called"
        if w is not None:
            why = (f"{mod} is defined, but forward() never calls it: its weights are read directly at the line "
                   "shown, which skips its forward() and anything it wraps (such as a LoRA adapter)")
        else:
            why = (f"{mod} is defined, but forward() never calls it (it may read its weights directly, like "
                   "F.linear(x, layer.weight), which skips any wrapper such as a LoRA adapter)")
        fix = f"call {mod} in forward(), or delete it / set requires_grad=False so it stops looking trainable"
    else:
        title = f"{n} parameter{'s' if n != 1 else ''} never get a gradient: never used in forward"
        why = f"{mod} runs, but no op ever reads these parameters"
        fix = "use them in forward(), or delete them / set requires_grad=False"
    level = BUG if has_opt else CHECK
    why += ". They sit in the model and the optimizer but never learn; DDP would fail with 'did not receive grad'."
    return Finding("dead-param", level, title, why, fix, w, c, sorted(params), {"cause": "unused", "module": module})


def orphan(params: list[str], w: Where | None, c: Where | None) -> Finding:
    n = len(params)
    title = f"{n} parameter{'s' if n != 1 else ''} never get a gradient: their output never reaches the loss"
    why = ("they are used (first at the line shown), but nothing computed from them ends up in the value you call "
           "backward() on: for example an extra output that is returned but never put into the loss, or a "
           "feature map the next stage skips")
    fix = "add that output to the loss, or remove the unused branch (or set requires_grad=False on it)"
    return Finding("dead-param", BUG, title, why, fix, w, c, sorted(params), {"cause": "orphan"})


def zero_grad(params: list[str], ev) -> Finding:
    n = len(params)
    if ev is not None:
        title = f"{n} parameter{'s' if n != 1 else ''} only ever get a zero gradient: through {ev.op}()"
        why = f"{ev.op}() has a zero derivative almost everywhere, so the gradient that reaches these parameters is exactly 0"
        fix = f"use a straight-through estimator, e.g. `x + (x.{ev.op}() - x).detach()`"
        return Finding("zero-grad", BUG, title, why, fix, ev.where, ev.caller, sorted(params), {"op": ev.op})
    title = f"{n} parameter{'s' if n != 1 else ''} only ever get a zero gradient"
    why = "every gradient element was exactly 0 in every step: they are masked out, behind a dead ReLU, or multiplied by zero"
    fix = "check masks and activations on the path from these parameters to the loss"
    return Finding("zero-grad", CHECK, title, why, fix, None, None, sorted(params))


# ---------------------------------------------------------------- optimizer and modes

def not_in_optimizer(params: list[str], numel: int) -> Finding:
    n = len(params)
    return Finding(
        "not-in-optimizer", BUG, f"{n} trainable parameter{'s' if n != 1 else ''} ({numel:,} values) are in no optimizer",
        "they get gradients but optimizer.step() never updates them. This usually means a layer was created or "
        "replaced after the optimizer was built (for example in a subclass __init__ after super().__init__()).",
        "build the optimizer after the whole model exists, or add these parameters with optimizer.add_param_group()",
        params=sorted(params))


def stale_optimizer(count: int, numel: int, replacements: list[str] | None = None) -> Finding:
    why = ("these tensors are not part of any model that ran: a layer or parameter was probably replaced after the "
           "optimizer was built, so the optimizer keeps updating the old copy.")
    if replacements:
        why += " The model now uses new parameters that no optimizer updates, so they never train."
    return Finding(
        "stale-optimizer", BUG, f"the optimizer updates {count} tensor{'s' if count != 1 else ''} ({numel:,} values) that the model no longer uses",
        why, "rebuild the optimizer after replacing layers, or update parameters in place (param.data.copy_(new))",
        params=sorted(replacements or []))


def eval_mode(mods: list[str], whole: bool, w, c) -> Finding:
    kinds = sorted(set(mods))
    if whole:
        title = "the model is in eval() mode while it trains"
        why = (f"gradients flow, but the model's training flag is off, so {', '.join(kinds)} behave as at inference "
               "(dropout does nothing, batch norm uses frozen running stats). Usually model.eval() was called for "
               "validation and model.train() was never called again.")
        level = BUG
    else:
        title = f"{len(mods)} layer{'s' if len(mods) != 1 else ''} ({', '.join(kinds)}) run in eval mode during training"
        why = "part of the model is in eval mode while gradients flow. This is fine if you froze them on purpose."
        level = CHECK
    return Finding("eval-mode", level, title, why, "call model.train() before the training step (e.g. at the top of each epoch)", w, c)


def batchnorm_batch1(shape, w, c) -> Finding:
    return Finding("batchnorm-batch1", CHECK, f"batch norm in training mode got a batch of 1 {_s(shape)}",
                   "with one sample, batch statistics are just that sample's own values, which makes the running "
                   "stats noisy and the output nearly constant per channel",
                   "use a larger batch, drop_last=True in the DataLoader, or GroupNorm/LayerNorm", w, c)


def grad_not_cleared(w, c, manual: bool = False) -> Finding:
    if manual:
        return Finding("grad-not-cleared", BUG, "gradients from the previous step were not cleared",
                       "this weight update used gradients that still held the previous step's values: backward() adds onto "
                       ".grad, so every update uses the sum of all past gradients and the effective learning rate keeps growing",
                       "clear the gradients once per step, e.g. model.zero_grad() or p.grad = None for each parameter, "
                       "before backward()", w, c)
    return Finding("grad-not-cleared", BUG, "gradients from the previous step were not cleared",
                   "this optimizer.step() used gradients that still held the previous step's values: backward() adds onto .grad, so every step uses the "
                   "sum of all past gradients and the effective learning rate keeps growing",
                   "call optimizer.zero_grad() once per step (before backward() or after step()). For gradient "
                   "accumulation, zero right after each optimizer.step().", w, c)
