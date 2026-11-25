"""Loader for the unified JSONC experiment config (configs/*.jsonc).

JSONC = JSON + `//` and `/* */` comments (plus tolerating trailing commas,
since these files are hand-edited). We deliberately avoid adding a `json5`
dependency to requirements.txt for this -- the subset of JSONC we need
(comments + trailing commas) is small enough to strip with a tiny parser
below, kept dependency-free.

The config file's keys are a 1:1 match to scripts/train.py's argparse `dest`
names (dashes become underscores), so loading one is just:

    python scripts/train.py --config configs/racetrack_experiment_config.jsonc

Any flag also passed explicitly on the command line overrides the file
(argparse's own `set_defaults` + `parse_args` precedence -- see train.py).
A top-level key starting with "_" is a reference-only/comment section (e.g.
benchmark-protocol notes) and is ignored by the loader, not forwarded as an
arg default. A `null` value also means "leave train.py's own default alone"
rather than literally setting an arg to None.
"""
from __future__ import annotations

import json
import re
from typing import Any


def _strip_jsonc_comments(text: str) -> str:
    """Remove `//` line comments and `/* */` block comments, respecting
    string literals (so a `//` or `/*` inside a JSON string is left alone)."""
    out = []
    i, n = 0, len(text)
    in_string = False
    string_char = ""
    while i < n:
        c = text[i]
        if in_string:
            out.append(c)
            if c == "\\" and i + 1 < n:
                out.append(text[i + 1])
                i += 2
                continue
            if c == string_char:
                in_string = False
            i += 1
            continue
        if c in ('"', "'"):
            in_string = True
            string_char = c
            out.append(c)
            i += 1
            continue
        if c == "/" and i + 1 < n and text[i + 1] == "/":
            while i < n and text[i] != "\n":
                i += 1
            continue
        if c == "/" and i + 1 < n and text[i + 1] == "*":
            i += 2
            while i + 1 < n and not (text[i] == "*" and text[i + 1] == "/"):
                i += 1
            i += 2
            continue
        out.append(c)
        i += 1
    return "".join(out)


def load_jsonc(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()
    cleaned = _strip_jsonc_comments(raw)
    cleaned = re.sub(r",(\s*[\]}])", r"\1", cleaned)  # tolerate trailing commas
    return json.loads(cleaned)


def experiment_config_to_arg_defaults(cfg: dict[str, Any]) -> dict[str, Any]:
    """Flattens a loaded config into {arg_dest_name: value}, dropping
    reference-only sections (leading "_") and null values (meaning: don't
    override train.py's own default for that flag)."""
    return {k: v for k, v in cfg.items() if not k.startswith("_") and v is not None}
