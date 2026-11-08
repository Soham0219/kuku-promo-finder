"""Load config.yaml (with optional overrides) and compile word lists."""
from __future__ import annotations

import copy
import re
from functools import lru_cache
from pathlib import Path

DEFAULT_PATH = Path(__file__).resolve().parent.parent / "config.yaml"


def deep_merge(base: dict, override: dict) -> dict:
    """Recursive merge. For lists, a key ending in '+' adds items and '-' removes them
    (e.g. {"words+": ["gun*"]} extends base["words"]); a plain key replaces the value."""
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(k, str) and k[-1:] in "+-" and isinstance(v, list) and isinstance(out.get(k[:-1]), list):
            name = k[:-1]
            if k.endswith("+"):
                out[name] = out[name] + [x for x in v if x not in out[name]]
            else:
                out[name] = [x for x in out[name] if x not in v]
        elif isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(path: str | Path | None = None, overrides: dict | None = None) -> dict:
    import yaml   # imported here so the browser build (which passes a dict) needs no YAML library
    with open(path or DEFAULT_PATH, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return deep_merge(cfg, overrides or {})


def config_from_dict(cfg: dict, overrides: dict | None = None) -> dict:
    """Same as load_config, for a config that was already parsed (e.g. by the website)."""
    return deep_merge(cfg, overrides or {})


SUFFIXES = r"(?:s|es|ed|d|ing|er|ers|ly|ion|ions)?"


@lru_cache(maxsize=512)
def compile_terms(terms: tuple[str, ...]) -> re.Pattern:
    """Whole-word match with common inflections ('slap' -> slaps, slapped is listed
    separately when spelling changes). A trailing '*' means prefix match ('humiliat*')."""
    parts = []
    for t in sorted({t.strip().lower() for t in terms if t.strip()}, key=len, reverse=True):
        if t.endswith("*"):
            parts.append(re.escape(t[:-1]) + r"[a-z]*")
        else:
            parts.append(re.escape(t) + SUFFIXES)
    if not parts:
        return re.compile(r"(?!x)x")
    return re.compile(r"(?<![a-z])(?:" + "|".join(parts) + r")(?![a-z])", re.I)


def terms(cfg_list) -> re.Pattern:
    return compile_terms(tuple(cfg_list or ()))
