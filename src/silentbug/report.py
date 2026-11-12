"""Findings and the printed report."""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from ._where import Where

# How sure we are that a finding is a real bug.
BUG = "bug"  # almost certainly wrong
CHECK = "check"  # often wrong, sometimes intended


@dataclass
class Finding:
    kind: str  # broadcast | dead-param | zero-grad | not-in-optimizer | stale-optimizer | double-activation | eval-mode | batchnorm-batch1 | grad-not-cleared
    level: str  # BUG or CHECK
    title: str
    why: str
    fix: str
    where: Where | None = None
    caller: Where | None = None  # your own line, when `where` is inside a library
    params: list[str] = field(default_factory=list)
    detail: dict = field(default_factory=dict)

    def key(self) -> tuple:
        loc = (self.where.file, self.where.line) if self.where else None
        return (self.kind, loc, self.title)

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "level": self.level,
            "title": self.title,
            "why": self.why,
            "fix": self.fix,
            "where": self.where.to_dict() if self.where else None,
            "caller": self.caller.to_dict() if self.caller else None,
            "params": self.params,
            "detail": self.detail,
        }


def _plist(names: list[str], limit: int = 6) -> str:
    if len(names) <= limit:
        return ", ".join(names)
    return ", ".join(names[:limit]) + f" and {len(names) - limit} more"


@dataclass
class Report:
    findings: list[Finding] = field(default_factory=list)
    steps: int = 0  # optimizer steps seen
    backwards: int = 0  # backward passes seen
    models: list[str] = field(default_factory=list)

    @property
    def bugs(self) -> list[Finding]:
        return [f for f in self.findings if f.level == BUG]

    @property
    def checks(self) -> list[Finding]:
        return [f for f in self.findings if f.level == CHECK]

    def kinds(self) -> set[str]:
        return {f.kind for f in self.findings}

    def __bool__(self) -> bool:  # truthy when something was found
        return bool(self.findings)

    def to_dict(self) -> dict:
        return {
            "findings": [f.to_dict() for f in self.findings],
            "steps": self.steps,
            "backwards": self.backwards,
            "models": self.models,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    def __str__(self) -> str:
        head = f"silentbug: checked {self.backwards} backward pass(es), {self.steps} optimizer step(s)"
        if self.models:
            head += f" on {', '.join(self.models)}"
        if not self.findings:
            msg = head + "\n  no silent bugs found"
            if self.backwards == 0:
                msg += " (but no backward pass ran, so gradient checks were skipped)"
            return msg
        out = [head, f"  {len(self.bugs)} likely bug(s), {len(self.checks)} thing(s) to check", ""]
        for i, f in enumerate(sorted(self.findings, key=lambda f: f.level != BUG), 1):
            tag = "BUG  " if f.level == BUG else "CHECK"
            out.append(f"[{tag}] {i}. {f.title}")
            if f.where:
                out.append(f"    at {f.where.short()}: {f.where.code}")
            if f.caller:
                out.append(f"    called from {f.caller.short()}: {f.caller.code}")
            if f.params:
                out.append(f"    parameters: {_plist(f.params)}")
            out.append(f"    why: {f.why}")
            out.append(f"    fix: {f.fix}")
            out.append("")
        return "\n".join(out).rstrip()
