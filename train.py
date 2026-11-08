"""Train a model on promos your team already posted, test it on held-out episodes, save it.

    python train.py --episodes path/to/scripts --labels posted.csv --model forest --out model.json
    python run.py   --episodes path/to/new_show --ml-model model.json --out outputs/new_show

labels.csv columns: episode,start,end[,label]  (times in seconds or mm:ss.ss; label 1 = posted, 0 = rejected)
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from promofinder.config import load_config                      # noqa: E402
from promofinder.learn import (DEFAULT_BLEND, DEFAULT_MODEL, parse_labels_csv,  # noqa: E402
                               to_json, train_and_test)
from promofinder.parser import load_episodes                    # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description="Learn promo picks from labelled data.")
    ap.add_argument("--episodes", required=True, help="folder of .txt screenplays")
    ap.add_argument("--labels", required=True, help="CSV of posted promos")
    ap.add_argument("--model", default=DEFAULT_MODEL, choices=["forest", "boosted", "logistic"])
    ap.add_argument("--test-share", type=float, default=0.5, help="share of labelled episodes kept for testing")
    ap.add_argument("--test-from", type=int, default=None, help="test on episodes >= this number instead")
    ap.add_argument("--blend", type=float, default=DEFAULT_BLEND, help="0 = formula only, 1 = model only")
    ap.add_argument("--config", default=None)
    ap.add_argument("--out", default="model.json")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    episodes = load_episodes(sorted(Path(args.episodes).glob("*.txt")))
    labels = parse_labels_csv(Path(args.labels).read_text(encoding="utf-8"))
    out = train_and_test(episodes, labels, cfg, args.model, args.test_share, args.test_from, args.blend,
                         progress=lambda i, n, m: print(f"  [{i}/{n}] {m}"))
    Path(args.out).write_text(to_json(out["model"]), encoding="utf-8")
    r = out["report"]
    print(f"Model: {r['model']} (blend {r['blend']}), trained on episodes {r['train_episodes'][0]}-"
          f"{r['train_episodes'][-1]}, tested on {r['test_episodes'][0]}-{r['test_episodes'][-1]}")
    print(f"  posted promos found  formula {r['formula']['found']}/{r['formula']['of']}   model {r['ml']['found']}/{r['ml']['of']}")
    print(f"  best pick right      formula {r['formula']['top1']:.2f}   model {r['ml']['top1']:.2f}")
    print(f"  clip ranking (AUC)   formula {r['clip_auc']['formula']:.2f}   model {r['clip_auc']['model']:.2f}")
    print("  top signals:", ", ".join(n for n, _ in r["top_features"]))
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
