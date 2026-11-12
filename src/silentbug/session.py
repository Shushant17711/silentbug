"""The tracker: one TorchFunctionMode plus a few global hooks.

How it works, in short:

* Every torch call made by user code passes through ``_Mode.__torch_function__``.
  There we look at the shapes going into broadcasting ops and losses, and we
  carry a bitmask on every tensor saying which trainable parameters it was
  computed from (bit i = parameter i). When a tensor that came from parameters
  goes into an op and comes out without a gradient (``.detach()``, ``.data``,
  ``.item()``, ``torch.no_grad()``, ``argmax`` ...) we record that cut, with the
  line that did it.
* Hooks on every parameter record which ones receive a gradient, and whether it
  is ever non-zero.
* Global module and optimizer hooks find the models and optimizers on their own,
  so the user only wraps their code.

After the run, parameters that never got a gradient are matched to the cut (or
the missing call) that explains it.
"""

from __future__ import annotations

import weakref
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.overrides import TorchFunctionMode
from torch.utils.weak import WeakTensorKeyDictionary

try:
    from torch.utils._pytree import tree_leaves
except Exception:  # pragma: no cover
    def tree_leaves(x):
        if isinstance(x, (list, tuple)):
            return [z for y in x for z in tree_leaves(y)]
        if isinstance(x, dict):
            return [z for y in x.values() for z in tree_leaves(y)]
        return [x]

from . import explain
from ._where import Where, torch_caller, where
from .report import BUG, CHECK, Finding, Report

# ---------------------------------------------------------------- op tables

_REFLECTED = {"radd", "rsub", "rmul", "rtruediv", "rdiv", "rfloordiv", "rpow", "rmod", "rand", "ror", "rxor", "rmatmul"}
_ALIASES = {
    "truediv": "div", "true_divide": "div", "divide": "div", "multiply": "mul", "subtract": "sub",
    "floordiv": "floor_divide", "mod": "remainder", "and": "logical_and", "or": "logical_or",
    "xor": "logical_xor", "greater": "gt", "less": "lt", "greater_equal": "ge", "less_equal": "le",
    "not_equal": "ne",
}
BROADCAST_OPS = {
    "add", "sub", "mul", "div", "floor_divide", "pow", "rsub", "maximum", "minimum", "fmax", "fmin",
    "atan2", "hypot", "remainder", "fmod", "eq", "ne", "lt", "le", "gt", "ge", "logical_and",
    "logical_or", "logical_xor", "bitwise_and", "bitwise_or", "where", "copysign", "xlogy", "lerp",
    "addcmul", "addcdiv",
}
SYMBOL = {"add": "+", "sub": "-", "rsub": "-", "mul": "*", "div": "/", "pow": "**", "eq": "==", "ne": "!=",
          "lt": "<", "le": "<=", "gt": ">", "ge": ">=", "floor_divide": "//"}
LOSS_SHAPE = {"mse_loss", "l1_loss", "smooth_l1_loss", "huber_loss", "soft_margin_loss", "poisson_nll_loss", "kl_div"}
ACTIVATIONS = {"softmax": "softmax", "sigmoid": "sigmoid", "log_softmax": "log_softmax"}
ACT_LOSSES = {"cross_entropy", "nll_loss", "binary_cross_entropy_with_logits", "kl_div"}
# Ops that keep the values (so a softmax stays a softmax through them).
PASS_THROUGH = {
    "view", "reshape", "squeeze", "unsqueeze", "flatten", "contiguous", "float", "double", "half", "bfloat16",
    "to", "clone", "transpose", "permute", "t", "getitem", "view_as", "reshape_as", "type", "attr:T", "cuda", "cpu",
}
MIXING = {"matmul", "bmm", "mm", "einsum"}
# Ops that show the author shaped the tensor on purpose for broadcasting.
SHAPING = {"unsqueeze", "expand", "expand_as", "broadcast_to", "atleast_2d", "atleast_3d"}
# Ops whose derivative is zero almost everywhere.
ZERO_DERIV = {"round", "floor", "ceil", "trunc", "sign", "sgn", "fix", "heaviside"}
# Calls that turn a tensor into a Python value or NumPy array.
VALUE_FUNCS = {"item", "tolist", "numpy", "__float__", "__int__", "__index__", "__array__", "__complex__"}
NEW_TENSOR = {"tensor", "as_tensor", "from_numpy", "new_tensor", "Tensor"}

EVAL_SENSITIVE = (nn.modules.dropout._DropoutNd, nn.modules.batchnorm._BatchNorm)

_CUT_PRIORITY = ["detach", "data", "value", "new_tensor", "no_grad", "inference_mode", "requires_grad_false", "int", "other"]


def _name(func) -> str:
    n = getattr(func, "__name__", None)
    if n is None:
        return str(func)
    if n == "__get__":
        owner = getattr(func, "__self__", None)
        return "attr:" + str(getattr(owner, "__name__", "?"))
    return n


def _base(name: str) -> str:
    """Map __radd__, __iadd__, add_, true_divide ... onto one op name."""
    n = name
    if n.startswith("__") and n.endswith("__"):
        n = n[2:-2]
        if n in _REFLECTED:
            n = n[1:]
        elif n.startswith("i") and (n[1:] in BROADCAST_OPS or n[1:] in _ALIASES):
            n = n[1:]
    elif n.endswith("_") and not n.endswith("__"):
        n = n[:-1]
    return _ALIASES.get(n, n)


def _is_inplace(name: str) -> bool:
    if name.startswith("__"):
        return name in ("__iadd__", "__isub__", "__imul__", "__itruediv__")
    return name.endswith("_") and not name.startswith("attr:")


def _shape(t: torch.Tensor) -> tuple[int, ...]:
    return tuple(int(s) for s in t.shape)


def crossed_broadcast(sa: tuple[int, ...], sb: tuple[int, ...]):
    """Return (out_shape, same_size) when both operands get expanded and their ranks differ, else None.

    (N,1) with (N,) -> (N,N) is the classic case: a size-N axis of one operand is
    crossed with a size-N axis of the other. same_size is True when that happens.
    """
    if len(sa) == len(sb):
        return None
    n = max(len(sa), len(sb))
    a = (1,) * (n - len(sa)) + sa
    b = (1,) * (n - len(sb)) + sb
    a_exp, b_exp = [], []  # sizes each operand gets stretched to
    for x, y in zip(a, b):
        if x == 1 and y > 1:
            a_exp.append(y)
        elif y == 1 and x > 1:
            b_exp.append(x)
        elif x != y:
            return None  # would raise anyway
    if not a_exp or not b_exp:
        return None
    out = tuple(max(x, y) for x, y in zip(a, b))
    same = bool(set(a_exp) & set(b_exp))
    return out, same


# ---------------------------------------------------------------- records

@dataclass
class _Cut:
    mask: int
    kind: str
    op: str
    where: Where | None
    caller: Where | None
    order: int


class _Done(BaseException):
    """Raised to end a `silentbug run` after the requested number of steps."""


class _Mode(TorchFunctionMode):
    def __init__(self, session: "Session"):
        super().__init__()
        self.s = session

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        out = func(*args, **kwargs)
        s = self.s
        if not s._busy:
            s._busy = True
            try:
                s._on_op(func, args, kwargs, out)
            except Exception as e:  # never break the user's program
                s._internal_errors.append(repr(e))
            finally:
                s._busy = False
        return out


class Session:
    """Watches a stretch of training code. Use `silentbug.check()` rather than this directly."""

    def __init__(self, *models: nn.Module, optimizer=None, max_steps: int | None = None, stop_at_limit: bool = False):
        self.report = Report()
        self._explicit_models = list(models)
        self._optimizers: list[weakref.ref] = []
        if optimizer is not None:
            for o in optimizer if isinstance(optimizer, (list, tuple)) else [optimizer]:
                self._add_opt(o)
        self.max_steps = max_steps
        self.stop_at_limit = stop_at_limit

        self._busy = False
        self._active = False
        self._done = False
        self._internal_errors: list[str] = []
        self._order = 0

        # taint
        self._masks = WeakTensorKeyDictionary()  # tensor -> int bitmask of parameters it depends on (with grad)
        self._pbit = WeakTensorKeyDictionary()  # leaf tensor -> bit
        self._bit_param: list[weakref.ref] = []
        self._first_use: dict[int, tuple[Where | None, Where | None]] = {}
        self._first_mask = 0  # bits that already have a first use
        self._frozen_use = WeakTensorKeyDictionary()  # frozen nn.Parameter -> first line that read it
        self._used_mask = 0
        self._cuts: list[_Cut] = []
        self._cut_cover: dict[str, int] = {}
        self._zero_ops: list[_Cut] = []
        self._zero_cover = 0

        # tags
        self._act = WeakTensorKeyDictionary()  # tensor -> (activation, Where)
        self._shaped = WeakTensorKeyDictionary()  # tensor -> True

        # gradient bookkeeping, by id(param)
        self._hooked: dict[int, object] = {}
        self._got_grad: set[int] = set()
        self._nonzero_grad: set[int] = set()
        self._stepped: set[int] = set()
        self._dirty: set[int] = set()  # holds a gradient carried over from before the last step
        self._data_of = WeakTensorKeyDictionary()  # p.data tensor -> weakref to the parameter p
        self._param_refs: dict[int, weakref.ref] = {}

        # modules
        self._called: dict[int, weakref.ref] = {}
        self._eval_mods: list[tuple[weakref.ref, Where | None, Where | None]] = []
        self._eval_seen: set[int] = set()

        self._seen_findings: set = set()
        self._handles: list = []
        self._mode = _Mode(self)

    # ------------------------------------------------------------ lifecycle

    def start(self) -> "Session":
        if self._active:
            return self
        self._active = True
        from torch.nn.modules.module import register_module_forward_pre_hook
        from torch.optim.optimizer import register_optimizer_step_post_hook, register_optimizer_step_pre_hook

        self._handles.append(register_module_forward_pre_hook(self._module_pre))
        self._handles.append(register_optimizer_step_pre_hook(self._opt_pre))
        self._handles.append(register_optimizer_step_post_hook(self._opt_post))
        for m in self._explicit_models:
            self._hook_params(m)
        self._mode.__enter__()
        return self

    def stop(self) -> Report:
        if not self._active:
            return self.report
        self._active = False
        try:
            self._mode.__exit__(None, None, None)
        except Exception:
            pass
        for h in self._handles:
            h.remove()
        self._handles.clear()
        self._busy = True
        try:
            self._finalize()
        finally:
            for h in self._hooked.values():
                try:
                    h.remove()
                except Exception:
                    pass
            self._hooked.clear()
            self._busy = False
        return self.report

    def __enter__(self) -> Report:
        self.start()
        return self.report

    def __exit__(self, *exc) -> bool:
        self.stop()
        return False

    # ------------------------------------------------------------ helpers

    def _add(self, f: Finding) -> None:
        k = f.key()
        if k in self._seen_findings:
            return
        self._seen_findings.add(k)
        self.report.findings.append(f)

    def _add_opt(self, opt) -> None:
        if not any(r() is opt for r in self._optimizers):
            self._optimizers.append(weakref.ref(opt))

    def _hook_param(self, p: torch.Tensor) -> None:
        pid = id(p)
        if pid in self._hooked or not p.requires_grad:
            return
        self._param_refs[pid] = weakref.ref(p)

        def hook(grad, pid=pid, pref=weakref.ref(p)):
            if self._busy or not self._active or grad is None:
                return None
            self._busy = True
            try:
                self._got_grad.add(pid)
                if pid not in self._nonzero_grad and bool(grad.detach().ne(0).any()):
                    self._nonzero_grad.add(pid)
                p = pref()
                old = None if p is None else p.grad
                old_nonzero = old is not None and bool(old.detach().ne(0).any())
                if pid in self._stepped:
                    # first backward since this parameter was stepped: old gradient should be gone
                    self._stepped.discard(pid)
                    if old_nonzero:
                        self._dirty.add(pid)
                elif not old_nonzero:
                    self._dirty.discard(pid)
            except Exception as e:
                self._internal_errors.append(repr(e))
            finally:
                self._busy = False
            return None

        self._hooked[pid] = p.register_hook(hook)

    def _hook_params(self, m: nn.Module) -> None:
        for p in m.parameters():
            self._hook_param(p)

    def _mask_of(self, t: torch.Tensor) -> int:
        m = self._masks.get(t)
        if m is not None:
            return m
        b = self._pbit.get(t)
        if b is None:
            if t.grad_fn is not None or not t.is_leaf:
                return 0
            b = len(self._bit_param)
            self._pbit[t] = b
            self._bit_param.append(weakref.ref(t))
            self._hook_param(t)
        return 1 << b

    # ------------------------------------------------------------ the op hook

    def _on_op(self, func, args, kwargs, out) -> None:
        name = _name(func)
        if name == "backward":
            self.report.backwards += 1
            return
        if name.startswith("attr:") and name != "attr:data" and name != "attr:T":
            return
        base = _base(name)
        leaves = tree_leaves((args, kwargs))
        ins = [t for t in leaves if isinstance(t, torch.Tensor)]
        if not ins:
            return
        self._order += 1
        if name == "attr:data" or _is_inplace(name):
            self._manual_update(name, args, out)

        if base in BROADCAST_OPS and len(ins) >= 2:
            self._check_broadcast(base, ins)
        if base in LOSS_SHAPE:
            self._check_loss_shape(base, args, kwargs)
        if base in ACT_LOSSES or base in ("softmax", "log_softmax"):
            self._check_act_loss(base, args, kwargs)

        outs = [o for o in tree_leaves(out) if isinstance(o, torch.Tensor)] if out is not None else []

        # gradient taint
        live = 0
        for t in ins:
            if t.requires_grad:
                live |= self._mask_of(t)
            elif (isinstance(t, nn.Parameter) and outs and not name.endswith("_") and torch.is_grad_enabled()
                  and t not in self._frozen_use):  # a real read, not requires_grad_(False) or an in-place init
                self._frozen_use[t] = where(3)
        if live:
            self._used_mask |= live
            # first real use: an op that builds a gradient-carrying tensor from the parameter
            # (not nn.init reading .dim() / .size() or filling it under no_grad)
            fresh = live & ~self._first_mask
            if fresh and any(o.requires_grad and o not in self._pbit for o in outs):
                w = where(3)
                self._first_mask |= fresh
                while fresh:
                    low = fresh & -fresh
                    self._first_use[low.bit_length() - 1] = w
                    fresh ^= low
            if not outs:
                if base in VALUE_FUNCS or name in VALUE_FUNCS:
                    self._cut(live, "value", name)
            else:
                for o in outs:
                    if o.requires_grad:
                        if o in self._pbit:
                            continue
                        prev = self._masks.get(o)
                        self._masks[o] = live if prev is None else (prev | live)
                    else:
                        self._cut(live, self._cut_kind(name, base, o), name)
                if base in ZERO_DERIV and any(o.requires_grad for o in outs):
                    if live & ~self._zero_cover:
                        w, c = where(2)
                        self._zero_ops.append(_Cut(live, "zero", base, w, c, self._order))
                        self._zero_cover |= live

        # tags for the next ops
        if outs:
            o = outs[0]
            if base in ACTIVATIONS:
                self._act[o] = (ACTIVATIONS[base], where(2)[0])
            elif base in PASS_THROUGH and ins[0] in self._act and o is not ins[0]:
                self._act[o] = self._act[ins[0]]
            elif base in MIXING and len(ins) >= 2:
                # probabilities @ constant one-hot labels is still a probability mix (matching networks);
                # attention (probabilities @ trainable values) is not, so require the other side to be constant
                a, b = ins[0], ins[1]
                for p, q in ((a, b), (b, a)):
                    tag = self._act.get(p)
                    if tag and tag[0] in ("softmax", "softmax-mix") and not q.requires_grad:
                        self._act[o] = ("softmax-mix", tag[1])
                        break
            shaped = base in SHAPING or (
                base == "getitem" and len(args) > 1 and (args[1] is None or (isinstance(args[1], tuple) and any(i is None for i in args[1])))
            )
            if shaped:
                self._shaped[o] = True

    @staticmethod
    def _cut_kind(name: str, base: str, o: torch.Tensor) -> str:
        if torch.is_inference_mode_enabled():
            return "inference_mode"
        if not torch.is_grad_enabled():
            return "no_grad"
        if base in ("detach",):
            return "detach"
        if name == "attr:data":
            return "data"
        if base == "requires_grad":
            return "requires_grad_false"
        if base in NEW_TENSOR:
            return "new_tensor"
        if not (o.is_floating_point() or o.is_complex()):
            return "int"
        return "other"

    def _cut(self, mask: int, kind: str, op: str) -> None:
        # torch's own bookkeeping (DDP syncing parameters, optimizers, clip_grad_norm_ ...)
        # detaches on purpose; only cuts made by the user's code (or other libraries) count.
        # no_grad / inference_mode are the user's choice when the op is torch's layer code
        # (F.linear inside nn.Linear), but not inside DDP, optimizers or nn.utils.
        by = torch_caller()
        if by == "infra" or (by == "torch" and kind not in ("no_grad", "inference_mode")):
            return
        cover = self._cut_cover.get(kind, 0)
        if not mask & ~cover:
            return
        self._cut_cover[kind] = cover | mask
        w, c = where(3)
        self._cuts.append(_Cut(mask, kind, op, w, c, self._order))

    # ------------------------------------------------------------ shape and loss checks

    def _check_broadcast(self, base: str, ins: list[torch.Tensor]) -> None:
        ts = [t for t in ins[:3] if t.dim() > 0 and t.numel() > 1]
        for i in range(len(ts)):
            for j in range(i + 1, len(ts)):
                a, b = ts[i], ts[j]
                if a in self._shaped or b in self._shaped:
                    continue
                sa, sb = _shape(a), _shape(b)
                r = crossed_broadcast(sa, sb)
                if r is None:
                    continue
                outs, same = r
                w, c = where(3)
                self._add(explain.broadcast(base, SYMBOL.get(base), sa, sb, outs, same, w, c))

    def _check_loss_shape(self, base: str, args, kwargs) -> None:
        x = args[0] if len(args) > 0 else kwargs.get("input")
        y = args[1] if len(args) > 1 else kwargs.get("target")
        if not (isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor)):
            return
        sx, sy = _shape(x), _shape(y)
        if sx == sy:
            return
        try:
            so = tuple(torch.broadcast_shapes(sx, sy))
        except RuntimeError:
            return
        w, c = where(3)
        self._add(explain.loss_shape(base, sx, sy, so, w, c))

    def _check_act_loss(self, base: str, args, kwargs) -> None:
        x = args[0] if len(args) > 0 else kwargs.get("input")
        if not isinstance(x, torch.Tensor):
            return
        tag = self._act.get(x)
        if tag is None:
            return
        act, act_where = tag
        f = explain.activation_into_loss(base, act, act_where, *where(3))
        if f is not None:
            self._add(f)

    # ------------------------------------------------------------ module / optimizer hooks

    def _module_pre(self, module: nn.Module, inputs) -> None:
        if self._busy or not self._active:
            return
        # no optimizer (weights updated by hand): count backward passes as steps instead
        if (self.max_steps is not None and self.report.steps == 0 and not self._optimizers
                and self.report.backwards >= self.max_steps):
            self.stop()
            if self.stop_at_limit:
                raise _Done()
            return
        self._busy = True
        try:
            mid = id(module)
            if mid not in self._called:
                self._called[mid] = weakref.ref(module)
                self._hook_params(module)
            if torch.is_grad_enabled() and isinstance(module, EVAL_SENSITIVE):
                if not module.training and mid not in self._eval_seen:
                    self._eval_seen.add(mid)
                    self._eval_mods.append((weakref.ref(module), *where(2)))
                if module.training and isinstance(module, nn.modules.batchnorm._BatchNorm):
                    x = inputs[0] if inputs else None
                    if isinstance(x, torch.Tensor) and x.dim() > 2 and x.shape[0] == 1:
                        self._add(explain.batchnorm_batch1(_shape(x), *where(2)))
        except Exception as e:
            self._internal_errors.append(repr(e))
        finally:
            self._busy = False

    def _param_of(self, t) -> torch.Tensor | None:
        if not isinstance(t, torch.Tensor):
            return None
        r = self._param_refs.get(id(t))
        if r is not None and r() is t:
            return t
        r = self._data_of.get(t)
        return r() if r is not None else None

    def _manual_update(self, name: str, args, out) -> None:
        """Weights updated by hand (p.data.add_(...), p -= lr * p.grad under no_grad)."""
        p = self._param_of(args[0]) if args else None
        if p is None:
            return
        if name == "attr:data":
            if isinstance(out, torch.Tensor):
                self._data_of[out] = weakref.ref(p)
            return
        if p.grad is None:  # initialisation, not an update
            return
        pid = id(p)
        if pid in self._dirty:
            self._dirty.discard(pid)
            self._add(explain.grad_not_cleared(*where(4), manual=True))
        self._stepped.add(pid)

    def _opt_pre(self, opt, args, kwargs) -> None:
        if not self._active:
            return
        self._add_opt(opt)
        stale = [p for g in opt.param_groups for p in g["params"] if id(p) in self._dirty and p.grad is not None]
        if stale:
            for p in stale:
                self._dirty.discard(id(p))
            self._add(explain.grad_not_cleared(*where(2)))
        self._busy = True  # do not trace the update itself

    def _opt_post(self, opt, args, kwargs) -> None:
        if not self._active:
            return
        self._busy = False
        self.report.steps += 1
        for g in opt.param_groups:
            for p in g["params"]:
                self._stepped.add(id(p))
        if self.max_steps is not None and self.report.steps >= self.max_steps:
            self.stop()
            if self.stop_at_limit:
                raise _Done()

    # ------------------------------------------------------------ analysis

    def _models(self) -> list[nn.Module]:
        mods = [r() for r in self._called.values()]
        mods = [m for m in mods if m is not None] + [m for m in self._explicit_models if m is not None]
        # torch.compile wraps the model in an OptimizedModule; report the original module's names
        mods = [getattr(m, "_orig_mod", m) if isinstance(getattr(m, "_orig_mod", None), nn.Module) else m
                for m in mods]
        uniq: dict[int, nn.Module] = {}
        for m in mods:
            uniq.setdefault(id(m), m)
        inner: set[int] = set()
        for m in uniq.values():
            for c in m.modules():
                if c is not m:
                    inner.add(id(c))
        roots = [m for mid, m in uniq.items() if mid not in inner and any(True for _ in m.parameters())]
        return roots

    def _finalize(self) -> None:
        rep = self.report
        models = self._models()
        names: dict[int, str] = {}
        owner: dict[int, nn.Module] = {}
        counts: dict[str, int] = {}
        for m in models:
            counts[type(m).__name__] = counts.get(type(m).__name__, 0) + 1
        seen_cls: dict[str, int] = {}
        for m in models:
            cls = type(m).__name__
            label = cls
            if counts[cls] > 1:
                seen_cls[cls] = seen_cls.get(cls, 0) + 1
                label = f"{cls}#{seen_cls[cls]}"
            rep.models.append(label)
            prefix = f"{label}." if len(models) > 1 else ""
            for n, p in m.named_parameters(remove_duplicate=True):
                names.setdefault(id(p), prefix + n)
                owner.setdefault(id(p), m)

        opts = [o for o in (r() for r in self._optimizers) if o is not None]
        in_opt: set[int] = set()
        opt_params: list[torch.Tensor] = []
        for o in opts:
            for gi, g in enumerate(o.param_groups):
                for i, p in enumerate(g["params"]):
                    in_opt.add(id(p))
                    opt_params.append(p)
                    names.setdefault(id(p), f"optimizer group {gi} param {i} {tuple(p.shape)}")

        # Which models are being trained?
        trained: list[nn.Module] = []
        for m in models:
            ps = [p for p in m.parameters() if p.requires_grad]
            if not ps:
                continue
            if opts:
                if any(id(p) in in_opt for p in ps):
                    trained.append(m)
            elif any(id(p) in self._got_grad for p in ps):
                trained.append(m)

        # optimizer entries that no model holds and that never got a gradient: replaced parameters
        model_ids = {id(p) for m in models for p in m.parameters()}
        stale = ([p for p in opt_params if id(p) not in model_ids and id(p) not in self._got_grad]
                 if opts and models else [])
        stale_ids = {id(p) for p in stale}

        candidates: dict[int, torch.Tensor] = {}
        for m in trained:
            for p in m.parameters():
                if p.requires_grad:
                    candidates[id(p)] = p
        for p in opt_params:
            if p.requires_grad and id(p) not in stale_ids:
                candidates[id(p)] = p

        # parameters left out of every optimizer
        if opts:
            missing = [p for m in trained for p in m.parameters() if p.requires_grad and id(p) not in in_opt]
            uniq = list({id(p): p for p in missing}.values())
            if stale:
                # the parameters the model now uses are the replacements of the stale ones
                rep.findings.append(explain.stale_optimizer(len(stale), sum(p.numel() for p in stale),
                                                            [names.get(id(p), "?") for p in uniq]))
            elif uniq:
                rep.findings.append(explain.not_in_optimizer([names.get(id(p), "?") for p in uniq],
                                                             sum(p.numel() for p in uniq)))

        # dead and zero-gradient parameters
        if rep.backwards == 0 and self._got_grad:
            rep.backwards = 1
        if rep.backwards > 0 and self._got_grad:
            groups: dict[tuple, list[str]] = {}
            group_info: dict[tuple, tuple] = {}
            for pid, p in candidates.items():
                if pid in self._nonzero_grad:
                    continue
                nm = names.get(pid, f"tensor {tuple(p.shape)}")
                bit = self._pbit.get(p)
                if pid in self._got_grad:
                    ev = self._first_event(self._zero_ops, bit)
                    key = ("zero", ev.op if ev else None, ev.where if ev else None)
                    group_info[key] = ("zero", ev, None)
                elif bit is None or (bit not in self._first_use and self._first_event(self._cuts, bit) is None):
                    mod_name = nm.rsplit(".", 1)[0] if "." in nm else ""
                    sub = self._submodule(owner.get(pid), nm, names)
                    called = sub is not None and id(sub) in self._called
                    # name the outermost module that never ran (the skipped wrapper, not its pieces)
                    parts, bypass = nm.split(".")[:-1], None
                    for k in range(1, len(parts) + 1):
                        sm = self._submodule(owner.get(pid), ".".join(parts[:k]) + ".x", names)
                        if sm is not None and id(sm) not in self._called:
                            mod_name, called = ".".join(parts[:k]), False
                            bypass = self._bypass_use(sm)
                            break
                    key = ("unused", mod_name, called)
                    group_info[key] = ("unused", mod_name, called, bypass)
                else:
                    ev = self._first_event(self._cuts, bit)
                    if ev is not None:
                        key = ("cut", ev.kind, ev.where, ev.op)
                        group_info[key] = ("cut", ev, None)
                    else:
                        fu = self._first_use.get(bit, (None, None))
                        key = ("orphan", fu[0])
                        group_info[key] = ("orphan", fu, None)
                groups.setdefault(key, []).append(nm)
            for key, plist in groups.items():
                kind, a, b, *rest = group_info[key]
                trained_by_opt = bool(opts)
                if kind == "zero":
                    rep.findings.append(explain.zero_grad(plist, a))
                elif kind == "unused":
                    w, c = rest[0] if rest and rest[0] else (None, None)
                    rep.findings.append(explain.unused(plist, a, b, trained_by_opt, w, c))
                elif kind == "cut":
                    rep.findings.append(explain.cut(plist, a))
                else:
                    rep.findings.append(explain.orphan(plist, *a))

        # eval mode during training
        if self._eval_mods and trained:
            trained_ids = {id(c) for m in trained for c in m.modules()}
            hits = [(r(), w, c) for r, w, c in self._eval_mods if r() is not None and id(r()) in trained_ids]
            if hits:
                whole = any(not m.training for m in trained)
                rep.findings.append(explain.eval_mode([type(m).__name__ for m, _, _ in hits], whole, hits[0][1], hits[0][2]))

        if self._internal_errors:
            rep.findings.append(Finding("internal", CHECK, "silentbug hit an internal error and may have missed things",
                                        "; ".join(sorted(set(self._internal_errors))[:3]),
                                        "please report this at the project issue tracker"))

    def _bypass_use(self, module: nn.Module) -> tuple[Where | None, Where | None] | None:
        """Where a parameter of a module that never ran was read directly (skipping its forward)."""
        for p in module.parameters():
            if p in self._frozen_use:
                return self._frozen_use[p]
            b = self._pbit.get(p)
            if b is not None and b in self._first_use:
                return self._first_use[b]
        return None

    @staticmethod
    def _submodule(root: nn.Module | None, pname: str, names: dict) -> nn.Module | None:
        if root is None or "." not in pname:
            return root
        path = pname.split(".")[:-1]
        if len(path) and path[0] == f"{type(root).__name__}" or (path and "#" in path[0]):
            path = path[1:]
        m = root
        for part in path:
            m = getattr(m, part, None)
            if m is None:
                return None
        return m if isinstance(m, nn.Module) else None

    @staticmethod
    def _first_event(events: list[_Cut], bit: int | None) -> _Cut | None:
        if bit is None:
            return None
        hits = [e for e in events if e.mask >> bit & 1]
        if not hits:
            return None
        rank = {k: i for i, k in enumerate(_CUT_PRIORITY)}
        hits.sort(key=lambda e: (rank.get(e.kind, 99), e.order))
        return hits[0]


def check(*models: nn.Module, optimizer=None) -> Session:
    """Watch the code inside the `with` block for silent training bugs.

        with silentbug.check() as report:
            for step in range(3):
                loss = loss_fn(model(x), y)
                loss.backward()
                opt.step(); opt.zero_grad()
        print(report)

    Models and optimizers are found on their own. Pass them in only if the model
    is never called inside the block.
    """
    return Session(*models, optimizer=optimizer)
