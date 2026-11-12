"""Find the line of user code that called the current torch op."""

from __future__ import annotations

import linecache
import os
import sys
import sysconfig
from dataclasses import dataclass

import torch

_TORCH_DIR = os.path.dirname(os.path.realpath(torch.__file__)) + os.sep
_OWN_DIR = os.path.dirname(os.path.realpath(__file__)) + os.sep
_LIB_DIRS = tuple(
    {os.path.realpath(p) + os.sep for p in (sysconfig.get_paths().get("purelib"), sysconfig.get_paths().get("platlib")) if p}
)
_STDLIB = os.path.realpath(sysconfig.get_paths()["stdlib"]) + os.sep

_real_cache: dict[str, str] = {}


def _real(path: str) -> str:
    r = _real_cache.get(path)
    if r is None:
        r = os.path.realpath(path) if not path.startswith("<") else path
        _real_cache[path] = r
    return r


def _skip(path: str) -> bool:
    if path.startswith("<"):
        return True
    p = _real(path)
    return p.startswith(_TORCH_DIR) or p.startswith(_OWN_DIR) or p.startswith(_STDLIB) and not any(
        p.startswith(d) for d in _LIB_DIRS
    )


def _is_lib(path: str) -> bool:
    p = _real(path)
    return any(p.startswith(d) for d in _LIB_DIRS)


@dataclass(frozen=True)
class Where:
    file: str
    line: int
    func: str
    code: str
    lib: bool = False  # True if the line is inside an installed package

    def short(self) -> str:
        f = self.file
        try:
            rel = os.path.relpath(f)
            if not rel.startswith(".."):
                f = rel
        except ValueError:
            pass
        return f"{f}:{self.line}"

    def to_dict(self) -> dict:
        return {"file": self.file, "line": self.line, "function": self.func, "code": self.code, "library": self.lib}


def where(skip: int = 2) -> tuple[Where | None, Where | None]:
    """Return (first frame outside torch, first frame outside installed packages)."""
    f = sys._getframe(skip)
    first = None
    while f is not None:
        path = f.f_code.co_filename
        if not _skip(path):
            lib = _is_lib(path)
            w = Where(path, f.f_lineno, f.f_code.co_name, linecache.getline(path, f.f_lineno).strip(), lib)
            if first is None:
                first = w
            if not lib:
                return first, (w if w != first else None)
        f = f.f_back
    return first, None
