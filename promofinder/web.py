"""Entry point for the website. Runs inside the browser (Pyodide), using the same engine as the CLI.

    run(files_json, cfg_json, opts_json) -> JSON string

files_json : [{"name": "episode-001.txt", "text": "..."}, ...]
cfg_json   : the parsed config.yaml (the page parses the YAML and passes a dict)
opts_json  : {"overrides": {...}, "ai": {"provider", "model", "api_key", "base_url"}}
"""
from __future__ import annotations

import json
import time

from .config import config_from_dict
from .engine import find_promos
from .parser import ensure_unique_numbers, parse_text
from .report import FLAG_HELP, to_payload, write_csv


def _excerpt(promo, indexes) -> list[dict]:
    segs = []
    for s in promo.segments:
        idx = indexes[s.episode]
        lines = []
        for b in idx.beats:
            e = b.event
            if e.kind == "camera" or e.start < s.start - 0.05 or e.start >= s.end:
                continue
            mark = "hook" if e.id == promo.hook_event else ("cut" if e.id == promo.cliff_event else "")
            lines.append({"t": round(e.start, 2), "kind": e.kind, "speaker": e.speaker, "cue": e.cue,
                          "vo": e.voice_over, "text": e.text, "mark": mark, "id": e.id})
        segs.append({"episode": s.episode, "start": s.start, "end": s.end,
                     "episode_duration": round(idx.duration, 2), "lines": lines})
    return segs


def run(files_json: str, cfg_json: str, opts_json: str = "{}", progress=None) -> str:
    opts = json.loads(opts_json or "{}")
    cfg = config_from_dict(json.loads(cfg_json), opts.get("overrides"))
    files = json.loads(files_json)
    episodes = ensure_unique_numbers([e for e in (parse_text(f["text"], f["name"]) for f in files) if e.events])
    if not episodes:
        return json.dumps({"error": "None of the files looked like a timestamped screenplay. Each needs "
                                    "blocks like **CAMERA:** with a 'Start: (00:00.00) | End: (00:02.00)' line."})

    ml = None
    if opts.get("ml_model"):
        from .learn import MLScorer, from_json
        try:
            raw = opts["ml_model"]
            ml = MLScorer(from_json(raw if isinstance(raw, str) else json.dumps(raw)))
        except (ValueError, KeyError) as e:
            return json.dumps({"error": f"Could not use the model: {e}"})

    reviewer = None
    ai = opts.get("ai") or {}
    if ai.get("provider") and ai["provider"] != "none":
        from .llm import LLMError, make_reviewer
        try:
            reviewer = make_reviewer(cfg, ai["provider"], ai.get("model") or None, ai.get("api_key") or None,
                                     ai.get("base_url") or None)
        except LLMError as e:
            return json.dumps({"error": str(e)})

    t = time.time()
    result = find_promos(episodes, cfg, llm=reviewer, progress=progress, ml=ml)
    meta = {"show": "uploaded scripts", "episodes": len(episodes),
            "mode": ("rules only" if reviewer is None else f"rules + AI ({reviewer.p.label})")
                    + (f" + learned model ({ml.model['type']})" if ml else ""),
            "genre": result["genre"].get("label", ""),
            "seconds": round(time.time() - t, 2)}
    if reviewer is not None:
        meta.update(llm_calls=reviewer.p.calls, llm_errors=reviewer.errors, llm_notes=reviewer.p.notes)

    payload = to_payload(result, cfg, meta)
    for pd, pr in zip(payload["promos"], result["promos"]):
        pd["excerpt"] = _excerpt(pr, result["indexes"])
    payload["post_first_ids"] = [p.id for p in result["post_first"]]
    payload["episodes"] = [{
        "number": i.ep.number, "title": i.ep.title, "logline": i.ep.logline, "duration": round(i.duration, 2),
        "curve": [[round(b.event.start, 2), round(b.hook, 3), round(b.cliff, 3)]
                  for b in i.beats if b.event.kind != "camera"],
    } for i in result["indexes"].values()]
    payload["flag_help"] = FLAG_HELP
    write_csv(result["promos"], "/tmp/promos.csv")
    with open("/tmp/promos.csv", encoding="utf-8") as f:
        payload["csv"] = f.read()
    return json.dumps(payload, ensure_ascii=False)


def train(files_json: str, cfg_json: str, labels_csv: str, opts_json: str = "{}", progress=None) -> str:
    """Train a model on posted promos and test it on held-out episodes. Returns JSON with the
    test report and the model (to download, or to use when finding promos)."""
    from .learn import DEFAULT_BLEND, DEFAULT_MODEL, parse_labels_csv, to_json, train_and_test
    opts = json.loads(opts_json or "{}")
    cfg = config_from_dict(json.loads(cfg_json), opts.get("overrides"))
    episodes = ensure_unique_numbers([e for e in (parse_text(f["text"], f["name"])
                                                  for f in json.loads(files_json)) if e.events])
    try:
        labels = parse_labels_csv(labels_csv)
        t = time.time()
        out = train_and_test(episodes, labels, cfg, opts.get("model") or DEFAULT_MODEL,
                             test_share=float(opts.get("test_share", 0.5)),
                             blend=float(opts.get("blend", DEFAULT_BLEND)), progress=progress)
    except ValueError as e:
        return json.dumps({"error": str(e)})
    out["report"]["seconds"] = round(time.time() - t, 1)
    return json.dumps({"report": out["report"], "model_json": to_json(out["model"])})
