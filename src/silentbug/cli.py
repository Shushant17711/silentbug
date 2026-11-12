"""Command line: `silentbug run train.py [args...]`."""

from __future__ import annotations

import argparse
import os
import runpy
import sys

from .session import Session, _Done

USAGE = """silentbug run [--steps N] [--keep-going] [--json FILE] (script.py | -m module) [args...]

Runs your training script with silentbug watching the first N optimizer steps
(default 3), prints what it found, and stops the script. Exit code: 0 nothing
found, 1 likely bugs found, 2 only things to check."""


def _parse(argv: list[str]):
    ap = argparse.ArgumentParser(prog="silentbug run", usage=USAGE)
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--keep-going", action="store_true", help="let the script keep running after the report")
    ap.add_argument("--json", metavar="FILE", help="also write the report as JSON ('-' for stdout only JSON)")
    ap.add_argument("-m", dest="module", action="store_true", help="run a module, like python -m")
    ap.add_argument("target")
    ap.add_argument("args", nargs=argparse.REMAINDER)
    return ap.parse_args(argv)


def run(argv: list[str]) -> int:
    a = _parse(argv)
    s = Session(max_steps=a.steps, stop_at_limit=not a.keep_going)
    old_argv = sys.argv[:]
    sys.argv = [a.target, *a.args]
    if not a.module:
        sys.path.insert(0, os.path.dirname(os.path.abspath(a.target)))
    error = None
    s.start()
    try:
        if a.module:
            runpy.run_module(a.target, run_name="__main__", alter_sys=True)
        else:
            runpy.run_path(a.target, run_name="__main__")
    except _Done:
        pass
    except SystemExit:
        pass
    except BaseException as e:  # the script crashed: still report what we saw
        error = e
    finally:
        sys.argv = old_argv
        rep = s.stop()
    rank = os.environ.get("RANK", "0")  # under torchrun every rank sees the same code
    if rank != "0":
        pass
    elif a.json == "-":
        print(rep.to_json())
    else:
        print(rep, file=sys.stderr if a.keep_going else sys.stdout)
        if a.json:
            with open(a.json, "w") as f:
                f.write(rep.to_json())
    if error is not None:
        print(f"\n(your script raised {type(error).__name__}: {error})", file=sys.stderr)
    return 1 if rep.bugs else (2 if rep.findings else 0)


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] in ("-h", "--help"):
        print(USAGE)
        return 0
    if argv[0] == "run":
        return run(argv[1:])
    if argv[0] in ("--version", "version"):
        from . import __version__
        print(__version__)
        return 0
    print(f"unknown command {argv[0]!r}\n\n{USAGE}", file=sys.stderr)
    return 64
