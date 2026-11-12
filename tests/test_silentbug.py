import json
import os
import subprocess
import sys
import warnings

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

import silentbug

warnings.simplefilter("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.join(HERE, "..", "benchmarks", "realworld")


def kinds(rep):
    return sorted(f.kind for f in rep.bugs)


def train(model, step, n=2, opt=None):
    opt = opt or torch.optim.SGD(model.parameters(), 0.1)
    with silentbug.check() as rep:
        for _ in range(n):
            loss = step(model)
            opt.zero_grad()
            loss.backward()
            opt.step()
    return rep


# ----------------------------------------------------------------- clean code


def test_clean_regression_reports_nothing():
    m = nn.Linear(4, 1)
    x, y = torch.randn(16, 4), torch.randn(16, 1)
    rep = train(m, lambda m: F.mse_loss(m(x), y))
    assert not rep.findings, str(rep)
    assert rep.steps == 2 and rep.backwards == 2


def test_clean_classifier_reports_nothing():
    m = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Dropout(0.1), nn.Linear(8, 3))
    x, y = torch.randn(16, 4), torch.randint(0, 3, (16,))
    rep = train(m, lambda m: F.cross_entropy(m(x), y))
    assert not rep.findings, str(rep)


def test_intended_outer_product_is_not_a_bug():
    m = nn.Linear(4, 1)
    x = torch.randn(8, 4)
    rep = train(m, lambda m: (m(x)[:, None] * torch.randn(1, 4)).pow(2).mean())
    assert not rep.bugs, str(rep)


def test_gan_style_detach_and_two_optimizers_is_clean():
    G, D = nn.Linear(4, 4), nn.Linear(4, 1)
    og, od = torch.optim.Adam(G.parameters()), torch.optim.Adam(D.parameters())
    with silentbug.check() as rep:
        for _ in range(2):
            fake = G(torch.randn(16, 4))
            dl = F.binary_cross_entropy_with_logits(D(fake.detach()), torch.zeros(16, 1))
            od.zero_grad(); dl.backward(); od.step()
            gl = F.binary_cross_entropy_with_logits(D(fake), torch.ones(16, 1))
            og.zero_grad(); gl.backward(); og.step()
    assert not rep.bugs, str(rep)


# ----------------------------------------------------------------- broadcasting


def test_broadcast_n1_vs_n():
    m = nn.Linear(4, 1)
    x, y = torch.randn(16, 4), torch.randn(16)
    rep = train(m, lambda m: ((m(x) - y) ** 2).mean())
    assert "broadcast" in kinds(rep)
    f = next(f for f in rep.bugs if f.kind == "broadcast")
    assert f.where.file.endswith("test_silentbug.py")


def test_loss_shape_mismatch():
    m = nn.Linear(4, 1)
    x, y = torch.randn(16, 4), torch.randn(16)
    rep = train(m, lambda m: F.mse_loss(m(x), y))
    assert "broadcast" in kinds(rep)


# ----------------------------------------------------------------- gradients


class Net(nn.Module):
    def __init__(self, mode):
        super().__init__()
        self.mode = mode
        self.a, self.b = nn.Linear(4, 4), nn.Linear(4, 1)

    def forward(self, x):
        h = self.a(x)
        if self.mode == "detach":
            h = h.detach()
        elif self.mode == "item":
            h = h * torch.tensor(h.sum().item())
        elif self.mode == "round":
            h = h.round()
        elif self.mode == "no_grad":
            with torch.no_grad():
                h = self.a(x)
        return self.b(h)


@pytest.mark.parametrize("mode,cause", [("detach", "detach"), ("round", None), ("no_grad", "no_grad")])
def test_dead_params_name_the_cause(mode, cause):
    m = Net(mode)
    x = torch.randn(8, 4)
    rep = train(m, lambda m: m(x).pow(2).mean())
    dead = [f for f in rep.bugs if f.kind in ("dead-param", "zero-grad")]
    assert dead, str(rep)
    assert all(p.startswith("a.") for f in dead for p in f.params)
    if cause:
        assert dead[0].detail.get("cause") == cause


def test_unused_module():
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.used, self.spare = nn.Linear(4, 1), nn.Linear(4, 1)

        def forward(self, x):
            return self.used(x)

    rep = train(M(), lambda m: m(torch.randn(8, 4)).pow(2).mean())
    f = [f for f in rep.findings if f.kind == "dead-param"]
    assert f and f[0].params == ["spare.bias", "spare.weight"]


def test_orphan_output():
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.a, self.aux = nn.Linear(4, 1), nn.Linear(4, 2)

        def forward(self, x):
            return self.a(x), self.aux(x)

    rep = train(M(), lambda m: m(torch.randn(8, 4))[0].pow(2).mean())
    assert any(f.kind == "dead-param" and f.detail.get("cause") == "orphan" for f in rep.bugs), str(rep)


# ----------------------------------------------------------------- optimizer and modes


def test_layer_added_after_optimizer():
    m = nn.Sequential(nn.Linear(4, 4))
    opt = torch.optim.SGD(m.parameters(), 0.1)
    m.append(nn.Linear(4, 1))
    rep = train(m, lambda m: m(torch.randn(8, 4)).pow(2).mean(), opt=opt)
    assert "not-in-optimizer" in kinds(rep)


def test_missing_zero_grad():
    m = nn.Linear(4, 1)
    opt = torch.optim.SGD(m.parameters(), 0.1)
    with silentbug.check() as rep:
        for _ in range(3):
            m(torch.randn(8, 4)).pow(2).mean().backward()
            opt.step()
    assert "grad-not-cleared" in kinds(rep)


def test_gradient_accumulation_is_clean():
    m = nn.Linear(4, 1)
    opt = torch.optim.SGD(m.parameters(), 0.1)
    with silentbug.check() as rep:
        for _ in range(2):
            for _ in range(4):
                m(torch.randn(8, 4)).pow(2).mean().backward()
            opt.step()
            opt.zero_grad()
    assert not rep.bugs, str(rep)


def test_eval_mode_while_training():
    m = nn.Sequential(nn.Linear(4, 8), nn.Dropout(0.5), nn.Linear(8, 1))
    m.eval()
    rep = train(m, lambda m: m(torch.randn(8, 4)).pow(2).mean())
    assert "eval-mode" in kinds(rep)


# ----------------------------------------------------------------- activations into losses


@pytest.mark.parametrize("loss_fn", [
    lambda z, y: F.cross_entropy(F.softmax(z, -1), y),
    lambda z, y: F.nll_loss(F.softmax(z, -1), y),
    lambda z, y: F.nll_loss(F.log_softmax(F.softmax(z, -1), -1), y),
])
def test_softmax_into_loss(loss_fn):
    m = nn.Linear(4, 3)
    x, y = torch.randn(8, 4), torch.randint(0, 3, (8,))
    rep = train(m, lambda m: loss_fn(m(x), y))
    assert "double-activation" in kinds(rep), str(rep)


def test_sigmoid_into_bce_with_logits():
    m = nn.Linear(4, 1)
    x, y = torch.randn(8, 4), torch.rand(8, 1).round()
    rep = train(m, lambda m: F.binary_cross_entropy_with_logits(torch.sigmoid(m(x)), y))
    assert "double-activation" in kinds(rep)


def test_sigmoid_into_bce_is_clean():
    m = nn.Linear(4, 1)
    x, y = torch.randn(8, 4), torch.rand(8, 1).round()
    rep = train(m, lambda m: F.binary_cross_entropy(torch.sigmoid(m(x)), y))
    assert not rep.findings, str(rep)


# ----------------------------------------------------------------- plumbing


def test_report_json_and_session_restores_state():
    m = nn.Linear(4, 1)
    rep = train(m, lambda m: F.mse_loss(m(torch.randn(8, 4)), torch.randn(8)))
    data = rep.to_dict()
    assert data["findings"][0]["kind"] == "broadcast"
    assert data["findings"][0]["level"] == silentbug.BUG
    # hooks are gone: training again outside the session adds nothing
    n = len(rep.findings)
    F.mse_loss(m(torch.randn(8, 4)), torch.randn(8)).backward()
    assert len(rep.findings) == n


def test_cli_exit_codes(tmp_path):
    bad = tmp_path / "bad.py"
    bad.write_text(
        "import torch, torch.nn as nn\n"
        "m = nn.Linear(4, 1); opt = torch.optim.SGD(m.parameters(), 0.1)\n"
        "for _ in range(3):\n"
        "    loss = ((m(torch.randn(8, 4)) - torch.randn(8)) ** 2).mean()\n"
        "    opt.zero_grad(); loss.backward(); opt.step()\n")
    good = tmp_path / "good.py"
    good.write_text(bad.read_text().replace("torch.randn(8))", "torch.randn(8, 1))"))
    run = lambda p: subprocess.run([sys.executable, "-m", "silentbug", "run", str(p)],
                                   capture_output=True, text=True).returncode
    assert run(bad) == 1
    assert run(good) == 0


def test_benchmark():
    if not os.path.isdir(BENCH):
        pytest.skip("benchmark not present")
    exe = os.path.join(os.path.dirname(sys.executable), "silentbug")
    r = subprocess.run([sys.executable, os.path.join(BENCH, "score.py"), exe],
                       capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stdout[-2000:]


# ---------------------------------------------------------------- torch.compile and DDP

def test_torch_compile_model():
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.a, self.g, self.b = nn.Linear(4, 8), nn.Linear(8, 8), nn.Linear(8, 1)

        def forward(self, x):
            h = torch.relu(self.a(x))
            return self.b(h * self.g(h).detach())

    m = M()
    cm = torch.compile(m, backend="eager")
    x, y = torch.randn(16, 4), torch.randn(16)
    rep = train(m, lambda _: F.mse_loss(cm(x), y))
    assert "broadcast" in kinds(rep), str(rep)
    dead = [f for f in rep.bugs if f.kind == "dead-param"]
    assert dead and dead[0].params == ["g.bias", "g.weight"], str(rep)


DDP_SCRIPT = """
import os, torch, torch.nn as nn, torch.nn.functional as F, torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
dist.init_process_group("gloo"); torch.manual_seed(0)
buggy = os.environ.get("BUGGY") == "1"
class Net(nn.Module):
    def __init__(s):
        super().__init__(); s.a = nn.Linear(8, 16); s.aux = nn.Linear(16, 4); s.b = nn.Linear(16, 1)
    def forward(s, x):
        h = torch.relu(s.a(x)); z = s.aux(h); return s.b(h), z
m = DDP(Net(), find_unused_parameters=True)
opt = torch.optim.SGD(m.parameters(), 0.1)
x = torch.randn(16, 8); y = torch.randn(16) if buggy else torch.randn(16, 1)
for _ in range(4):
    out, z = m(x)
    loss = F.mse_loss(out, y) + (0 if buggy else z.pow(2).mean())
    opt.zero_grad(); loss.backward(); opt.step()
dist.destroy_process_group()
"""


@pytest.mark.parametrize("buggy", [True, False])
def test_ddp_two_ranks(tmp_path, buggy):
    if not torch.distributed.is_available():
        pytest.skip("torch.distributed not available")
    script = tmp_path / "ddp.py"
    script.write_text(DDP_SCRIPT)
    env = dict(os.environ, BUGGY="1" if buggy else "0", PYTHONWARNINGS="ignore")
    r = subprocess.run(
        [sys.executable, "-m", "torch.distributed.run", "--nproc-per-node", "2", "--no-python",
         os.path.join(os.path.dirname(sys.executable), "silentbug"), "run", "--json", "-", str(script)],
        capture_output=True, text=True, timeout=300, env=env, cwd=tmp_path)
    if buggy:
        assert r.returncode != 0
        rep = json.loads(r.stdout[r.stdout.index("{"):])
        got = {(f["kind"], f["detail"].get("cause")) for f in rep["findings"]}
        assert got == {("broadcast", None), ("dead-param", "orphan")}, r.stdout
    else:
        assert r.returncode == 0, r.stdout + r.stderr[-2000:]


# ---------------------------------------------------------------- manual updates (no optimizer)

def _manual(clear: bool, style: str):
    m = nn.Linear(4, 1)
    with silentbug.check() as rep:
        for _ in range(3):
            if clear:
                m.zero_grad()
            m(torch.randn(8, 4)).pow(2).mean().backward()
            for p in m.parameters():
                if style == "data":
                    p.data.add_(p.grad, alpha=-0.1)
                else:
                    with torch.no_grad():
                        p -= 0.1 * p.grad
    return rep


@pytest.mark.parametrize("style", ["data", "no_grad"])
def test_manual_update_missing_zero_grad(style):
    rep = _manual(False, style)
    assert "grad-not-cleared" in kinds(rep), str(rep)


@pytest.mark.parametrize("style", ["data", "no_grad"])
def test_manual_update_clean(style):
    rep = _manual(True, style)
    assert not rep.bugs, str(rep)
