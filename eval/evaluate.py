"""Measure the promo finder against editor picks, and tune the score weights.

Metrics (per split):
  recall     share of editor picks that some promo covers (>= 50% of the pick's time)
  hook_hit   share of found picks where the promo's hook line lies inside the pick
  precision  share of single-clip promos that overlap an editor pick (>= 30% of the promo)
  top1       in episodes with a pick, share where the episode's best promo matches it
  p@10       share of the "post first" list that overlaps an editor pick

Tuning: grid-search the component weights and the episode-end prior on the DEV split (episodes 1-25) only,
then report the TEST split (26-50) once with the chosen weights. Tuning never sees test.

Usage:
  python eval/evaluate.py                 # evaluate current config.yaml
  python eval/evaluate.py --tune          # grid search on dev, report test, write eval/report.json
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from promofinder.config import load_config          # noqa: E402
from promofinder.engine import find_promos           # noqa: E402
from promofinder.parser import episode_of, load_episodes   # noqa: E402

DEV = range(1, 26)
TEST = range(26, 51)


PICKS = ROOT / "data" / "editor_picks.yaml"   # kept beside the input pack: its notes describe confidential scripts


def load_picks(path=PICKS):
    return yaml.safe_load(open(path, encoding="utf-8"))["picks"]


def covered(pick, seg) -> float:
    if seg.episode != pick["episode"]:
        return 0.0
    inter = max(0.0, min(pick["end"], seg.end) - max(pick["start"], seg.start))
    return inter / max(1e-6, pick["end"] - pick["start"])


def evaluate(result, picks, episodes_in_split) -> dict:
    eps = set(episodes_in_split)
    sp = [p for p in picks if p["episode"] in eps]
    promos = [pr for pr in result["promos"] if set(pr.episodes) & eps]
    singles = [pr for pr in promos if pr.kind == "single"]
    idx = result["indexes"]

    found, hook_hits, details = 0, 0, []
    for p in sp:
        best, best_pr = 0.0, None
        for pr in promos:
            for s in pr.segments:
                c = covered(p, s)
                if c > best:
                    best, best_pr = c, pr
        ok = best >= 0.5
        found += ok
        hh = False
        if ok and best_pr is not None and best_pr.hook_event:
            ev = idx[episode_of(best_pr.hook_event)].ep.get(best_pr.hook_event)
            hh = ev is not None and p["start"] - 2 <= ev.start <= p["end"]
            hook_hits += hh
        details.append({"episode": p["episode"], "coverage": round(best, 2),
                        "found": ok, "hook_inside": hh,
                        "by": best_pr.id if best_pr else None})

    def overlaps_pick(pr):
        for s in pr.segments:
            for p in sp:
                if p["episode"] != s.episode:
                    continue
                inter = max(0.0, min(p["end"], s.end) - max(p["start"], s.start))
                if inter / max(1e-6, s.duration) >= 0.3:
                    return True
        return False

    prec = sum(overlaps_pick(pr) for pr in singles) / max(1, len(singles))
    # top-1: in episodes that have an editor pick, does the episode's best promo match it?
    picked_eps = {p["episode"] for p in sp}
    best_per_ep = {}
    for pr in singles:
        e = pr.episodes[0]
        if e in picked_eps and (e not in best_per_ep or pr.total > best_per_ep[e].total):
            best_per_ep[e] = pr
    top1 = sum(overlaps_pick(pr) for pr in best_per_ep.values()) / max(1, len(picked_eps))
    top = [pr for pr in result["post_first"] if set(pr.episodes) & eps]
    p10 = sum(overlaps_pick(pr) for pr in top) / max(1, len(top))
    recall = found / max(1, len(sp))
    f1 = 2 * recall * prec / max(1e-9, recall + prec)
    return {"picks": len(sp), "found": found, "recall": round(recall, 3),
            "hook_hit": round(hook_hits / max(1, found), 3), "precision": round(prec, 3),
            "f1": round(f1, 3), "top1": round(top1, 3), "post_first_hits": round(p10, 3), "post_first_n": len(top),
            "details": details}


def run(episodes, cfg):
    return find_promos(episodes, cfg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", default=str(ROOT / "data" / "episodes"))
    ap.add_argument("--config", default=None)
    ap.add_argument("--tune", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "eval" / "report.json"))
    args = ap.parse_args()

    files = sorted(Path(args.episodes).glob("*.txt"))
    episodes = load_episodes(files)
    picks = load_picks()
    base = load_config(args.config)

    report = {"default_weights": base["weights"]}
    res = run(episodes, base)
    report["default"] = {"dev": evaluate(res, picks, DEV), "test": evaluate(res, picks, TEST)}

    if args.tune:
        dev_eps = [e for e in episodes if e.number in DEV]
        grid = {
            "hook": [0.24, 0.32, 0.40],
            "cliffhanger": [0.22, 0.30, 0.38],
            "energy": [0.08, 0.14, 0.20],
            "clarity": [0.08, 0.16, 0.24],
            "episode_end_prior": [0.45, 0.55, 0.65],
        }
        best = None
        trials = []
        for h, c, e, cl, ep in itertools.product(*grid.values()):
            w = {"hook": h, "cliffhanger": c, "energy": e, "clarity": cl, "duration": 0.08}
            cfg = load_config(args.config, {"weights": w, "promo": {"episode_end_prior": ep}})
            r = evaluate(run(dev_eps, cfg), picks, DEV)
            trials.append({"weights": w, "episode_end_prior": ep, "dev_f1": r["f1"], "dev_recall": r["recall"],
                           "dev_precision": r["precision"]})
            key = (r["f1"] + r["top1"], r["recall"])
            if best is None or key > best[0]:
                best = (key, w, ep)
        tuned = load_config(args.config, {"weights": best[1], "promo": {"episode_end_prior": best[2]}})
        res_t = run(episodes, tuned)
        report["tuned_weights"] = best[1]
        report["tuned_episode_end_prior"] = best[2]
        report["tuned"] = {"dev": evaluate(res_t, picks, DEV), "test": evaluate(res_t, picks, TEST)}
        report["grid_trials"] = sorted(trials, key=lambda t: -t["dev_f1"])[:15]

    Path(args.out).write_text(json.dumps(report, indent=2))
    for name in ("default", "tuned"):
        if name in report:
            for split in ("dev", "test"):
                r = report[name][split]
                print(f"{name:8s} {split:4s} recall {r['recall']:.2f} ({r['found']}/{r['picks']})  "
                      f"hook_hit {r['hook_hit']:.2f}  precision {r['precision']:.2f}  "
                      f"f1 {r['f1']:.2f}  top1 {r['top1']:.2f}  post-first hits {r['post_first_hits']:.2f} (n={r['post_first_n']})")
    if "tuned_weights" in report:
        print("tuned weights:", report["tuned_weights"], "episode_end_prior:", report["tuned_episode_end_prior"])


if __name__ == "__main__":
    main()
