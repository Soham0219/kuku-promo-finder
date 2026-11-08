"""Learn from promos an editor actually posted.

Instead of fixed weights, train a small model on labelled data:

  1. Labels: past promos (episode, start, end; optional label 1 = posted / 0 = rejected).
  2. Every candidate clip the engine can cut in a labelled episode becomes one row of
     features (the formula's components plus what is in the hook line, the cut line and the
     clip). A clip is a positive if it starts within 3 s and ends within 5 s of a posted promo,
     so the model learns where promos start and stop rather than "longer clips overlap more".
  3. Train on some episodes and test on the others, never splitting one episode across both.
  4. When the model is used, each clip's score blends the formula with the model:
     score = (1 - blend) * formula + blend * 100 * P(good promo).

Three models, all in plain Python so they also run inside the browser with no downloads:
  - "boosted":  gradient-boosted decision trees with a logistic loss and second-order leaf
                values, the method behind XGBoost (Chen & Guestrin, KDD 2016).
  - "forest":   a random forest of regression trees on bootstrap samples (Breiman 2001).
  - "logistic": L2-regularised logistic regression on standardised features.
Trees split on histogram bins (quantiles), as LightGBM and XGBoost's "hist" method do.
"""
from __future__ import annotations

import csv
import io
import json
import math
import random
import re

FEATURES_VERSION = 1


# ------------------------------------------------------------------------------ labels
def _secs(v) -> float:
    s = str(v).strip()
    if re.fullmatch(r"\d+:\d+(\.\d+)?", s):
        m, x = s.split(":")
        return int(m) * 60 + float(x)
    return float(s)


def parse_labels_csv(text: str) -> list[dict]:
    """CSV with a header: episode,start,end[,label]. Times as seconds or mm:ss.ss.
    label: 1 = posted / good (default), 0 = rejected. A promo that joins several moments
    is written as several rows with the same `promo` id (optional column)."""
    rows = []
    reader = csv.DictReader(io.StringIO(text.strip()))
    if not reader.fieldnames:
        raise ValueError("The labels file is empty.")
    names = {n.strip().lower(): n for n in reader.fieldnames}
    for need in ("episode", "start", "end"):
        if need not in names:
            raise ValueError(f"The labels file needs columns episode, start, end (missing '{need}').")
    for i, r in enumerate(reader, 2):
        try:
            ep = int(float(r[names["episode"]]))
            start, end = _secs(r[names["start"]]), _secs(r[names["end"]])
            lab = r.get(names.get("label", ""), "") if "label" in names else ""
            label = 1 if str(lab).strip() in ("", "1", "yes", "true", "posted", "good") else 0
        except (TypeError, ValueError):
            raise ValueError(f"Labels file, line {i}: could not read episode/start/end.")
        if end <= start:
            raise ValueError(f"Labels file, line {i}: end must be after start.")
        rows.append({"episode": ep, "start": start, "end": end, "label": label})
    if not rows:
        raise ValueError("The labels file has no rows.")
    return rows


def covered(pick: dict, ep: int, t0: float, t1: float) -> float:
    """Share of the labelled promo covered by the clip."""
    if ep != pick["episode"]:
        return 0.0
    inter = max(0.0, min(pick["end"], t1) - max(pick["start"], t0))
    return inter / max(1e-6, pick["end"] - pick["start"])


START_TOL, END_TOL = 3.0, 5.0   # a clip "is" the posted promo if it starts and ends this close to it


def matches(pick: dict, ep: int, t0: float, t1: float) -> bool:
    """Training label: the clip starts within 3 s and ends within 5 s of a posted promo, so the
    model learns where good promos start (the hook) and stop (the cut), not just that longer
    clips overlap more."""
    return (ep == pick["episode"] and abs(t0 - pick["start"]) <= START_TOL
            and abs(t1 - pick["end"]) <= END_TOL)


# ---------------------------------------------------------------------------- features
def feature_names(cfg: dict) -> list[str]:
    cats = list(cfg["stakes"].keys())
    return (["hook", "cliffhanger", "energy", "clarity", "duration_fit", "formula",
             "seconds", "start_pos", "end_pos", "ends_episode",
             "hook_intensity", "hook_stakes", "hook_is_dialogue", "hook_offset",
             "cut_open", "cut_stakes", "cut_intensity", "cut_resolution", "cut_is_dialogue",
             "cut_question", "cut_trailing", "cut_threat", "cut_sting",
             "end_resolution", "max_gap", "talk_share", "speakers", "lines", "scenes",
             "voice_over_share", "mean_intensity", "max_stakes", "mean_hook", "mean_cliff"]
            + [f"has_{c}" for c in cats] + [f"cut_{c}" for c in cats])


def clip_features(idx, t0: float, t1: float, info: dict) -> list[float]:
    s = info["scores"]
    inside = [b for b in info["inside"] if b.event.kind != "camera"]
    hb, cb = info["hook_beat"], info["cliff_beat"]
    dur = max(1e-6, idx.duration)
    lines = [b for b in inside if b.event.kind == "dialogue"]
    talk = sum(b.event.duration for b in lines) / max(1.0, t1 - t0)
    n = max(1, len(inside))
    cats = list(idx.cfg["stakes"].keys())
    present = {c: 0 for c in cats}
    for b in inside:
        for c in b.stake_tags:
            if c in present:
                present[c] = 1
    ctags = cb.stake_tags if cb else {}
    otags = cb.open_tags if cb else []
    f = [
        s["hook"], s["cliffhanger"], s["energy"], s["clarity"], s["duration"], s.get("formula", s["total"]) / 100,
        (t1 - t0) / 60, t0 / dur, t1 / dur, 1.0 if t1 >= dur - idx.cfg["promo"]["episode_end_tolerance"] else 0.0,
        hb.intensity if hb else 0.0, hb.stakes if hb else 0.0,
        1.0 if hb and hb.event.kind == "dialogue" else 0.0, (hb.event.start - t0) if hb else 3.0,
        cb.openness if cb else 0.0, cb.stakes if cb else 0.0, cb.intensity if cb else 0.0,
        cb.resolution if cb else 0.0, 1.0 if cb and cb.event.kind == "dialogue" else 0.0,
        1.0 if "question" in otags else 0.0, 1.0 if "cut-off line" in otags else 0.0,
        1.0 if "threat" in otags else 0.0, 1.0 if "dramatic sting" in otags else 0.0,
        info["resolution"], info["max_gap"], talk,
        len({b.event.speaker for b in lines}), len(lines) / 10, len({b.event.scene for b in inside}),
        sum(1 for b in lines if b.event.voice_over) / max(1, len(lines)),
        sum(b.intensity for b in inside) / n, max((b.stakes for b in inside), default=0.0),
        sum(b.hook for b in inside) / n, sum(b.cliff for b in inside) / n,
    ]
    f += [float(present[c]) for c in cats]
    f += [1.0 if c in ctags else 0.0 for c in cats]
    return f


def episode_windows(idx):
    """All candidate clips of one episode with their formula info (same search as the engine)."""
    from .engine import score_window, windows
    p = idx.cfg["promo"]
    for t0, t1, sb, eb in windows(idx, p["min_duration"], p["max_duration"]):
        yield t0, t1, score_window(idx, t0, t1)


def build_rows(indexes: list, labels: list[dict]):
    """Rows (features, label, episode) for every candidate clip of every labelled episode."""
    by_ep: dict[int, list[dict]] = {}
    for lab in labels:
        by_ep.setdefault(lab["episode"], []).append(lab)
    X, y, eps = [], [], []
    for idx in indexes:
        labs = by_ep.get(idx.ep.number)
        if not labs:
            continue
        for t0, t1, info in episode_windows(idx):
            pos = any(matches(l, idx.ep.number, t0, t1) for l in labs if l["label"] == 1)
            X.append(clip_features(idx, t0, t1, info))
            y.append(1 if pos else 0)
            eps.append(idx.ep.number)
    return X, y, eps


# ------------------------------------------------------------------------------- trees
def _bin_edges(X, n_bins: int):
    edges = []
    for j in range(len(X[0])):
        col = sorted(r[j] for r in X)
        qs = sorted({col[int(len(col) * k / n_bins)] for k in range(1, n_bins)})
        edges.append(qs)
    return edges


def _binned(X, edges):
    from bisect import bisect_right
    return [[bisect_right(edges[j], r[j]) for j in range(len(r))] for r in X]


def _grow(Xb, edges, rows, g, h, depth, p, rng, feats_per_split):
    G, H = sum(g[i] for i in rows), sum(h[i] for i in rows)
    lam = p["lambda"]
    leaf = {"v": -G / (H + lam)}
    if depth >= p["max_depth"] or len(rows) < 2 * p["min_rows"]:
        return leaf
    nf = len(edges)
    feats = rng.sample(range(nf), feats_per_split) if feats_per_split < nf else range(nf)
    parent = G * G / (H + lam)
    best = (p["min_gain"], None, None)
    for j in feats:
        nb = len(edges[j]) + 1
        if nb < 2:
            continue
        gs, hs, cs = [0.0] * nb, [0.0] * nb, [0] * nb
        for i in rows:
            b = Xb[i][j]
            gs[b] += g[i]; hs[b] += h[i]; cs[b] += 1
        gl = hl = 0.0
        cl = 0
        for b in range(nb - 1):
            gl += gs[b]; hl += hs[b]; cl += cs[b]
            cr = len(rows) - cl
            if cl < p["min_rows"] or cr < p["min_rows"]:
                continue
            gr, hr = G - gl, H - hl
            gain = gl * gl / (hl + lam) + gr * gr / (hr + lam) - parent
            if gain > best[0]:
                best = (gain, j, b)
    if best[1] is None:
        return leaf
    _, j, b = best
    left = [i for i in rows if Xb[i][j] <= b]
    right = [i for i in rows if Xb[i][j] > b]
    return {"f": j, "t": edges[j][b],
            "l": _grow(Xb, edges, left, g, h, depth + 1, p, rng, feats_per_split),
            "r": _grow(Xb, edges, right, g, h, depth + 1, p, rng, feats_per_split)}


def _tree_value(node, x):
    while "f" in node:
        node = node["l"] if x[node["f"]] < node["t"] else node["r"]
    return node["v"]


# bisect_right puts value == edge into the right bin, so "x < t" goes left; the split above
# uses bins <= b (values strictly below edges[j][b]) for the left side, which matches.

DEFAULTS = {
    "boosted": {"trees": 60, "learning_rate": 0.15, "max_depth": 3, "min_rows": 10, "lambda": 1.0,
                "min_gain": 1e-6, "row_sample": 0.8, "feature_sample": 0.8, "bins": 16},
    "forest": {"trees": 60, "max_depth": 6, "min_rows": 5, "lambda": 1.0, "min_gain": 1e-6,
               "bins": 16, "feature_sample": 0.4},
    "logistic": {"steps": 200, "learning_rate": 0.5, "l2": 0.01},
}


def _sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def train(X, y, kind: str = "boosted", params: dict | None = None, seed: int = 7) -> dict:
    if kind not in DEFAULTS:
        raise ValueError(f"Unknown model '{kind}'. Choose boosted, forest or logistic.")
    if not X or sum(y) == 0 or sum(y) == len(y):
        raise ValueError("Training needs both good and other clips; check that the labels match the episodes.")
    p = dict(DEFAULTS[kind], **(params or {}))
    rng = random.Random(seed)
    n, nf = len(X), len(X[0])
    if kind == "logistic":
        mu = [sum(r[j] for r in X) / n for j in range(nf)]
        sd = [math.sqrt(sum((r[j] - mu[j]) ** 2 for r in X) / n) or 1.0 for j in range(nf)]
        Z = [[(r[j] - mu[j]) / sd[j] for j in range(nf)] for r in X]
        pos = sum(y)
        wpos, wneg = n / (2 * pos), n / (2 * (n - pos))      # balance the classes
        w, b = [0.0] * nf, 0.0
        for _ in range(p["steps"]):
            gw, gb = [0.0] * nf, 0.0
            for zi, yi in zip(Z, y):
                e = (_sigmoid(b + sum(a * c for a, c in zip(w, zi))) - yi) * (wpos if yi else wneg)
                gb += e
                for j in range(nf):
                    gw[j] += e * zi[j]
            b -= p["learning_rate"] * gb / n
            w = [w[j] - p["learning_rate"] * (gw[j] / n + p["l2"] * w[j]) for j in range(nf)]
        return {"type": "logistic", "w": w, "b": b, "mu": mu, "sd": sd, "params": p}

    edges = _bin_edges(X, p["bins"])
    Xb = _binned(X, edges)
    fps = max(1, int(round(nf * p["feature_sample"])))
    trees = []
    if kind == "forest":
        for _ in range(p["trees"]):
            rows = [rng.randrange(n) for _ in range(n)]
            g = [-float(v) for v in y]              # leaf = mean label (h = 1, lambda small)
            h = [1.0] * n
            trees.append(_grow(Xb, edges, rows, g, h, 0, p, rng, fps))
        return {"type": "forest", "trees": trees, "params": p}

    pos = sum(y)
    base = math.log(pos / (n - pos))
    F = [base] * n
    for _ in range(p["trees"]):
        prob = [_sigmoid(v) for v in F]
        g = [pi - yi for pi, yi in zip(prob, y)]
        h = [max(1e-6, pi * (1 - pi)) for pi in prob]
        rows = [i for i in range(n) if rng.random() < p["row_sample"]] or list(range(n))
        tree = _grow(Xb, edges, rows, g, h, 0, p, rng, fps)
        trees.append(tree)
        F = [f + p["learning_rate"] * _tree_value(tree, x) for f, x in zip(F, X)]
    return {"type": "boosted", "trees": trees, "base": base, "params": p}


def predict(model: dict, x: list[float]) -> float:
    t = model["type"]
    if t == "logistic":
        z = model["b"] + sum(w * (v - m) / s for w, v, m, s in zip(model["w"], x, model["mu"], model["sd"]))
        return _sigmoid(z)
    if t == "forest":
        return max(0.0, min(1.0, sum(_tree_value(tr, x) for tr in model["trees"]) / len(model["trees"])))
    lr = model["params"]["learning_rate"]
    return _sigmoid(model["base"] + lr * sum(_tree_value(tr, x) for tr in model["trees"]))


def auc(scores: list[float], y: list[int]) -> float:
    """Chance that a random good clip scores above a random other clip (ties count half)."""
    pairs = sorted(zip(scores, y))
    pos = sum(y); neg = len(y) - pos
    if not pos or not neg:
        return float("nan")
    rank_sum, i = 0.0, 0
    while i < len(pairs):
        j = i
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:
            j += 1
        avg = (i + j + 1) / 2
        rank_sum += avg * sum(1 for k in range(i, j) if pairs[k][1] == 1)
        i = j
    return (rank_sum - pos * (pos + 1) / 2) / (pos * neg)


def importance(model: dict, names: list[str], top: int = 8) -> list[tuple[str, float]]:
    """Split counts per feature (trees) or |weight| (logistic): which signals the model uses."""
    if model["type"] == "logistic":
        vals = {n: abs(w) for n, w in zip(names, model["w"])}
    else:
        vals = {}

        def walk(node):
            if "f" in node:
                vals[names[node["f"]]] = vals.get(names[node["f"]], 0) + 1
                walk(node["l"]); walk(node["r"])
        for tr in model["trees"]:
            walk(tr)
    total = sum(vals.values()) or 1.0
    return [(k, round(v / total, 3)) for k, v in sorted(vals.items(), key=lambda kv: -kv[1])[:top]]


# --------------------------------------------------------------------------- the scorer
class MLScorer:
    """Plugs a trained model into the engine: blends its probability with the formula."""

    def __init__(self, model: dict, blend: float | None = None):
        self.model = model
        self.blend = float(model.get("blend", 1.0) if blend is None else blend)

    def apply(self, idx, t0: float, t1: float, info: dict) -> None:
        s = info["scores"]
        s.setdefault("formula", s["total"])
        prob = predict(self.model, clip_features(idx, t0, t1, info))
        s["ml"] = 100 * prob
        s["total"] = (1 - self.blend) * s["formula"] + self.blend * s["ml"]


def to_json(model: dict) -> str:
    return json.dumps(model, separators=(",", ":"))


def from_json(text: str) -> dict:
    m = json.loads(text)
    if m.get("kind") != "promofinder-model" or m.get("features_version") != FEATURES_VERSION:
        raise ValueError("This is not a Promo Finder model file (or it is from another version).")
    return m


# ------------------------------------------------------------------------- end to end
def split_episodes(labels: list[dict], test_share: float = 0.5, test_from: int | None = None):
    eps = sorted({l["episode"] for l in labels})
    if test_from is not None:
        train = [e for e in eps if e < test_from]
        test = [e for e in eps if e >= test_from]
    else:
        k = max(1, int(round(len(eps) * (1 - test_share))))
        train, test = eps[:k], eps[k:]
    if not train or not test:
        raise ValueError("Need labelled promos in at least two episodes to train and test.")
    return train, test


def found_rate(result: dict, labels: list[dict], eps: set) -> dict:
    """Share of posted promos (in `eps`) that the kept promos match, and top-1 hits."""
    posted = [l for l in labels if l["label"] == 1 and l["episode"] in eps]
    promos = [p for p in result["promos"] if set(p.episodes) & eps]
    found = sum(1 for l in posted if any(covered(l, s.episode, s.start, s.end) >= 0.5
                                         for p in promos for s in p.segments))
    best = {}
    for p in promos:
        if p.kind == "single":
            e = p.episodes[0]
            if e not in best or p.total > best[e].total:
                best[e] = p
    with_label = {l["episode"] for l in posted}
    def overlaps(l, s):   # same rule as eval/evaluate.py: 30% of the clip is the posted promo
        inter = max(0.0, min(l["end"], s.end) - max(l["start"], s.start))
        return l["episode"] == s.episode and inter / max(1e-6, s.end - s.start) >= 0.3
    top1 = sum(1 for e in with_label if e in best and any(
        overlaps(l, best[e].segments[0]) for l in posted if l["episode"] == e))
    return {"found": found, "of": len(posted), "rate": round(found / max(1, len(posted)), 3),
            "top1": round(top1 / max(1, len(with_label)), 3)}


# Defaults chosen on the tuning episodes only (train 1-17, validate 18-25; eval/evaluate_ml.py).
DEFAULT_MODEL, DEFAULT_BLEND = "forest", 0.25


def train_and_test(episodes: list, labels: list[dict], cfg: dict, kind: str = DEFAULT_MODEL,
                   test_share: float = 0.5, test_from: int | None = None, blend: float = DEFAULT_BLEND,
                   neg_per_pos: int = 4, seed: int = 7, progress=None) -> dict:
    """Train on some labelled episodes, test on the rest, and compare with the formula."""
    from .engine import EpisodeIndex, find_promos
    from .genre import resolve_genre
    cfg_g, _ = resolve_genre(episodes, cfg)
    train_eps, test_eps = split_episodes(labels, test_share, test_from)
    by_num = {e.number: e for e in episodes}
    missing = sorted({l["episode"] for l in labels} - set(by_num))
    if missing:
        raise ValueError(f"The labels mention episodes that were not uploaded: {missing[:8]}")

    def rows_for(nums):
        idxs = [EpisodeIndex(by_num[n], cfg_g) for n in nums]
        return build_rows(idxs, [l for l in labels if l["episode"] in set(nums)])

    if progress:
        progress(1, 4, "Building training rows")
    Xtr, ytr, _ = rows_for(train_eps)
    rng = random.Random(seed)
    pos_i = [i for i, v in enumerate(ytr) if v]
    neg_i = [i for i, v in enumerate(ytr) if not v]
    keep = pos_i + rng.sample(neg_i, min(len(neg_i), neg_per_pos * max(1, len(pos_i))))
    Xs, ys = [Xtr[i] for i in keep], [ytr[i] for i in keep]
    if progress:
        progress(2, 4, f"Training ({kind}) on {len(Xs)} clips")
    model = train(Xs, ys, kind, seed=seed)
    names = feature_names(cfg_g)
    model.update(kind="promofinder-model", features_version=FEATURES_VERSION, features=names,
                 blend=blend, trained_on={"episodes": train_eps, "rows": len(Xs), "positives": sum(ys)})

    if progress:
        progress(3, 4, "Testing on held-out episodes")
    Xte, yte, _ = rows_for(test_eps)
    clip_auc = {"model": round(auc([predict(model, x) for x in Xte], yte), 3),
                "formula": round(auc([x[5] for x in Xte], yte), 3)}
    test_set = set(test_eps)
    test_episodes = [by_num[n] for n in test_eps]
    base = find_promos(test_episodes, cfg)
    with_ml = find_promos(test_episodes, cfg, ml=MLScorer(model, blend))
    report = {
        "model": kind, "blend": blend, "train_episodes": train_eps, "test_episodes": test_eps,
        "train_rows": len(Xs), "train_positives": sum(ys), "test_rows": len(Xte), "test_positives": sum(yte),
        "clip_auc": clip_auc,
        "formula": found_rate(base, labels, test_set),
        "ml": found_rate(with_ml, labels, test_set),
        "top_features": importance(model, names),
    }
    model["test_report"] = report
    if progress:
        progress(4, 4, "Done")
    return {"model": model, "report": report}
