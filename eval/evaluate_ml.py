"""Does a model learned from posted promos beat the formula? Measured honestly.

1. Choose the model type and blend on the tuning episodes only: train on 1-17, validate on 18-25.
2. Train the chosen setup on episodes 1-25 and test on 26-50 (never seen), next to the formula.
   All three model types are reported on the test episodes too, for transparency.
Writes numbers only to eval/ml_report.json.

    python eval/evaluate_ml.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "eval"))

import evaluate as ev                                              # noqa: E402
from promofinder.config import load_config                         # noqa: E402
from promofinder.engine import find_promos                         # noqa: E402
from promofinder.learn import MLScorer, train_and_test              # noqa: E402
from promofinder.parser import load_episodes                       # noqa: E402

KINDS, BLENDS = ("forest", "boosted", "logistic"), (0.25, 0.5, 0.75, 1.0)


def brief(r):
    return {k: r[k] for k in ("picks", "found", "recall", "top1", "precision", "hook_hit")}


def main():
    cfg = load_config()
    eps = load_episodes(sorted((ROOT / "data" / "episodes").glob("*.txt")))
    picks = ev.load_picks()
    labels = [{"episode": p["episode"], "start": p["start"], "end": p["end"], "label": 1} for p in picks]
    dev = [e for e in eps if e.number <= 25]
    val = [e for e in dev if e.number >= 18]
    test = [e for e in eps if e.number >= 26]
    report = {"validation": {"formula": brief(ev.evaluate(find_promos(val, cfg), picks, range(18, 26)))}}

    best = None
    for kind in KINDS:
        m = train_and_test(dev, [l for l in labels if l["episode"] <= 25], cfg, kind, test_from=18)["model"]
        for blend in BLENDS:
            r = ev.evaluate(find_promos(val, cfg, ml=MLScorer(m, blend)), picks, range(18, 26))
            report["validation"][f"{kind}@{blend}"] = brief(r)
            key = (r["f1"] + r["top1"], r["recall"])
            if best is None or key > best[0]:
                best = (key, kind, blend)
    _, kind, blend = best
    report["chosen_on_validation"] = {"model": kind, "blend": blend}

    report["test"] = {"formula": brief(ev.evaluate(find_promos(test, cfg), picks, ev.TEST))}
    for k in KINDS:
        m = train_and_test(eps, labels, cfg, k, test_from=26, blend=blend)["model"]
        report["test"][f"{k}@{blend}"] = brief(ev.evaluate(find_promos(test, cfg, ml=MLScorer(m, blend)), picks, ev.TEST))
    (ROOT / "eval" / "ml_report.json").write_text(json.dumps(report, indent=2))
    print("chosen on validation:", kind, blend)
    for name, r in report["test"].items():
        print(f"test {name:16s} found {r['found']}/{r['picks']} ({r['recall']:.2f})  top1 {r['top1']:.2f}  "
              f"precision {r['precision']:.2f}  hook inside pick {r['hook_hit']:.2f}")


if __name__ == "__main__":
    main()
