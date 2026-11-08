"""Genre mode: detect a show's genre from its scripts and adjust scoring for it.

A comedy promo is not built like a thriller promo. A thriller should cut on a threat or a
reveal, a romance on an almost-kiss or a confession, a comedy on an awkward beat. Each genre
in config.yaml has marker words used for detection and `overrides` that are merged into the
config (a list key ending in '+' adds words, '-' removes them). The default genre (drama)
has no overrides, so a drama show is scored exactly as the tuned base config.

Detection: for every non-camera line we check the text and the delivery cue for each
genre's markers. rate[g] = share of lines with at least one marker. The best non-default
genre wins only if its rate is at least `min_rate` and at least `lead` times the default
genre's rate; otherwise the show is the default genre.
"""
from __future__ import annotations

from .config import deep_merge, terms
from .parser import Episode


def genre_rates(episodes: list[Episode], cfg: dict) -> dict:
    profiles = cfg["genre"]["profiles"]
    pats = {g: terms(p.get("markers", [])) for g, p in profiles.items()}
    counts = {g: 0 for g in profiles}
    lines = 0
    for ep in episodes:
        for e in ep.events:
            if e.kind == "camera":
                continue
            lines += 1
            text = f"{e.cue} {e.text}"
            for g, pat in pats.items():
                if pat.search(text):
                    counts[g] += 1
    return {g: round(c / lines, 4) if lines else 0.0 for g, c in counts.items()}


def detect_genre(episodes: list[Episode], cfg: dict) -> tuple[str, dict]:
    gcfg = cfg["genre"]
    default = gcfg["default"]
    rates = genre_rates(episodes, cfg)
    others = sorted((g for g in rates if g != default), key=lambda g: -rates[g])
    if others:
        best = others[0]
        d = gcfg["detect"]
        if rates[best] >= d["min_rate"] and rates[best] >= d["lead"] * rates.get(default, 0.0):
            return best, rates
    return default, rates


def resolve_genre(episodes: list[Episode], cfg: dict) -> tuple[dict, dict]:
    """Returns (config with the genre's overrides applied, genre info for the report)."""
    gcfg = cfg.get("genre")
    if not gcfg:
        return cfg, {"genre": "", "label": "", "mode": "off", "rates": {}}
    mode = str(gcfg.get("mode", "auto") or "auto").lower()
    name, rates = detect_genre(episodes, cfg)
    if mode != "auto":
        if mode not in gcfg["profiles"]:
            raise ValueError(f"Unknown genre '{mode}'. Choose auto or one of: {', '.join(gcfg['profiles'])}")
        name = mode
    prof = gcfg["profiles"][name]
    out = deep_merge(cfg, prof.get("overrides") or {})
    info = {"genre": name, "label": prof.get("label", name.title()), "mode": mode,
            "detected": mode == "auto", "rates": rates, "hint": prof.get("hint", "")}
    return out, info


def promo_moods(promo, idx_by_ep: dict, cfg: dict) -> list[str]:
    """Mood tags for ad targeting, from the stake categories found in the promo's lines.
    Each category counts its stakes weight; the hook and cut lines count double."""
    mcfg = cfg.get("mood") or {}
    labels = mcfg.get("labels", {})
    totals: dict[str, float] = {}
    for seg in promo.segments:
        for b in idx_by_ep[seg.episode].inside(seg.start, seg.end):
            if b.event.kind == "camera":
                continue
            mult = 2.0 if b.id in (promo.hook_event, promo.cliff_event) else 1.0
            for cat in b.stake_tags:
                if cat in labels:
                    totals[cat] = totals.get(cat, 0.0) + mult * cfg["stakes"][cat]["weight"]
    if not totals:
        return []
    whole = sum(totals.values())
    ranked = sorted(totals, key=lambda c: (-totals[c], c))
    keep = [c for c in ranked if totals[c] / whole >= mcfg.get("min_share", 0.25)][: mcfg.get("max_tags", 2)]
    return [labels[c] for c in keep]
