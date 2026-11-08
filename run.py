"""Command-line runner.

Rules only (free, no API key):
    python run.py --episodes data/episodes --out outputs/rules

With an AI reviewer (any provider; the key comes from an env var or --api-key):
    GEMINI_API_KEY=... python run.py --provider gemini --out outputs/gemini
    python run.py --provider openai_compatible --base-url https://host/v1 --model some-model
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

from promofinder.config import load_config
from promofinder.engine import find_promos
from promofinder.llm import make_reviewer
from promofinder.parser import load_episodes
from promofinder.report import write_all


def main(argv=None):
    ap = argparse.ArgumentParser(description="Find promos in microdrama screenplays.")
    ap.add_argument("--episodes", default="data/episodes", help="folder of .txt screenplays")
    ap.add_argument("--out", default="outputs/run", help="output folder")
    ap.add_argument("--config", default=None, help="config.yaml (default: repo config)")
    ap.add_argument("--show", default="", help="show name for the report")
    ap.add_argument("--provider", default=None,
                    help="none | gemini | groq | openai | anthropic | openai_compatible | ollama | mock")
    ap.add_argument("--model", default=None)
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--ml-model", default=None, help="a model file from train.py (learned from posted promos)")
    ap.add_argument("--genre", default=None,
                    help="auto (default) | drama | thriller | horror | romance | comedy")
    args = ap.parse_args(argv)

    files = sorted(Path(args.episodes).glob("*.txt"))
    if not files:
        sys.exit(f"No .txt screenplays in {args.episodes}")
    cfg = load_config(args.config, {"genre": {"mode": args.genre}} if args.genre else None)
    episodes = load_episodes(files)
    reviewer = make_reviewer(cfg, args.provider, args.model, args.api_key, args.base_url)
    mode = "rules only" if reviewer is None else f"rules + AI ({reviewer.p.label})"

    ml = None
    if args.ml_model:
        from promofinder.learn import MLScorer, from_json
        ml = MLScorer(from_json(Path(args.ml_model).read_text(encoding="utf-8")))
        mode += f" + learned model ({ml.model['type']})"
        print(f"Using learned model: {args.ml_model}")
    print(f"{len(episodes)} episodes, mode: {mode}")
    t = time.time()
    result = find_promos(episodes, cfg, llm=reviewer, ml=ml,
                         progress=lambda i, n, msg: print(f"  [{i}/{n}] {msg}", end="\r"))
    elapsed = time.time() - t
    g = result["genre"]
    if g.get("genre"):
        print(f"\nGenre: {g['label']} ({'detected' if g['detected'] else 'chosen'})" + " " * 20)
    meta = {
        "show": args.show or Path(args.episodes).resolve().name,
        "episodes": len(episodes),
        "mode": mode,
        "genre": g.get("label", ""),
        "seconds": round(elapsed, 1),
        "config_sha": hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:12],
        "input_files": [f.name for f in files],
    }
    if reviewer is not None:
        meta.update(llm_calls=reviewer.p.calls, llm_cache_hits=reviewer.p.cache_hits,
                    llm_errors=reviewer.errors, llm_notes=reviewer.p.notes, llm_model_used=reviewer.p.label)
    payload = write_all(result, cfg, meta, args.out)
    print(f"\nDone in {elapsed:.1f} s: {len(result['promos'])} promos -> {args.out}/")
    for row in payload["post_first"]:
        print(f"  {row['rank']:>2}. {row['segments']:<40} {row['hook'][:70]}")
    if reviewer is not None and reviewer.errors:
        print(f"  AI warnings: {len(reviewer.errors)} (see promos.json meta)")


if __name__ == "__main__":
    main()
