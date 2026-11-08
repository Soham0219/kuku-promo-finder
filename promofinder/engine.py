"""Promo finder engine.

1. Score every beat (signals.py).
2. Enumerate every clip [t0, t1] whose start and end are beats, snapped to camera-shot
   boundaries without cutting a line of dialogue, with min <= t1 - t0 <= max.
3. Score each clip: Score = sum_k w_k * component_k (hook, cliffhanger, energy, clarity,
   duration), all components in [0, 1], total in [0, 100].
4. Keep the best non-overlapping clips per episode (greedy selection with an IoU limit).
5. Build multi-segment promos: a hook segment + a later cliffhanger segment.
6. Rank everything globally.

All numbers come from config.yaml; nothing here knows about a particular show.
"""
from __future__ import annotations

import bisect
import math
import re
from dataclasses import dataclass, field, asdict

from .config import terms
from .parser import Episode, Event, fmt_time
from .signals import Beat, score_episode

HONORIFICS = re.compile(r"^(the|lady|lord|sir|madam|mr|mrs|miss|dr|young|old|elder)\s+", re.I)


def norm_speaker(name: str) -> str:
    n = name.strip()
    prev = None
    while prev != n:
        prev = n
        n = HONORIFICS.sub("", n)
    return n.lower()


# ----------------------------------------------------------------------------- data
@dataclass
class Segment:
    episode: int
    start: float
    end: float
    first_event: str = ""
    last_event: str = ""

    @property
    def duration(self) -> float:
        return round(self.end - self.start, 2)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["start_ts"] = fmt_time(self.start)
        d["end_ts"] = fmt_time(self.end)
        d["duration"] = self.duration
        return d


@dataclass
class Promo:
    id: str
    kind: str                       # single | multi
    segments: list[Segment]
    scores: dict
    hook_event: str
    cliff_event: str
    hook: str
    cliffhanger: str
    why: str
    flags: list = field(default_factory=list)
    hook_tags: list = field(default_factory=list)
    cliff_tags: list = field(default_factory=list)
    speakers: list = field(default_factory=list)
    source: str = "rules"           # rules | llm | rules+llm
    llm: dict = field(default_factory=dict)
    rank: int = 0
    rank_score: float = 0.0
    rank_reason: str = ""
    moods: list = field(default_factory=list)   # e.g. ["Betrayal", "Romance"], for ad targeting

    @property
    def duration(self) -> float:
        return round(sum(s.duration for s in self.segments), 2)

    @property
    def episodes(self) -> list[int]:
        return sorted({s.episode for s in self.segments})

    @property
    def total(self) -> float:
        return self.scores.get("total", 0.0)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "rank": self.rank,
            "rank_score": round(self.rank_score, 2),
            "rank_reason": self.rank_reason,
            "kind": self.kind,
            "episodes": self.episodes,
            "duration": self.duration,
            "segments": [s.to_dict() for s in self.segments],
            "hook": self.hook,
            "cliffhanger": self.cliffhanger,
            "why": self.why,
            "scores": {k: round(v, 3) for k, v in self.scores.items()},
            "hook_event": self.hook_event,
            "cliff_event": self.cliff_event,
            "hook_tags": self.hook_tags,
            "cliff_tags": self.cliff_tags,
            "speakers": self.speakers,
            "moods": self.moods,
            "flags": self.flags,
            "source": self.source,
            "llm": self.llm,
        }


# ----------------------------------------------------------------------- episode index
class EpisodeIndex:
    """Pre-computed lookups for one episode."""

    def __init__(self, ep: Episode, cfg: dict):
        self.ep = ep
        self.cfg = cfg
        self.beats: list[Beat] = score_episode(ep, cfg)
        self.by_id = {b.id: b for b in self.beats}
        self.content = [b for b in self.beats if b.event.kind != "camera"]
        self.content_starts = [b.event.start for b in self.content]
        self.shots = [b.event for b in self.beats if b.event.kind == "camera"]
        self.dialogue = [b for b in self.beats if b.event.kind == "dialogue"]
        self.duration = ep.duration
        sens = cfg.get("sensitive", {}).get("words", [])
        self.sensitive_re = terms(sens) if sens else None
        self.ml = None          # optional learned scorer (promofinder.learn.MLScorer)

    # -- cut snapping ------------------------------------------------------------
    def _shot_at(self, t: float) -> Event | None:
        """The camera shot that is on screen at time t (latest shot starting at or before t)."""
        best = None
        for s in self.shots:
            if s.start <= t + 0.05:
                best = s
            else:
                break
        return best

    def snap_start(self, anchor: Beat) -> float:
        snap = self.cfg["promo"]["shot_snap"]
        cuts = self.cfg["cuts"]
        t = anchor.event.start
        shot = self._shot_at(t)
        t0 = shot.start if shot and 0 <= t - shot.start <= snap else max(0.0, t - cuts["pre_roll"])
        # never start in the middle of somebody's line
        for d in self.dialogue:
            ev = d.event
            if ev.start < t0 - 0.05 < ev.end - cuts["line_edge_tolerance"]:
                if ev.end <= t + 0.05:
                    t0 = ev.end            # cut right after the earlier line
                elif t - ev.start <= cuts["include_overlapping_line"]:
                    t0 = ev.start          # include the whole overlapping line
        return round(max(0.0, min(t0, t)), 2)

    def snap_end(self, anchor: Beat) -> float:
        snap = self.cfg["promo"]["shot_snap"]
        cuts = self.cfg["cuts"]
        t = anchor.event.end
        shot = None
        for s in self.shots:
            if s.start <= t + 0.05 and s.end >= t - 0.05:
                shot = s
        t1 = shot.end if shot and 0 <= shot.end - t <= snap else t + cuts["post_roll"]
        # 1) don't run into a line that starts after the anchor ends
        for d in self.dialogue:
            ev = d.event
            if ev.id != anchor.id and t - 0.05 <= ev.start < t1:
                t1 = max(t, ev.start - cuts["gap_before_next_line"])
                break
        # 2) if a line is still being spoken at the cut (it overlapped the anchor), finish it
        for _ in range(3):
            live = [d.event for d in self.dialogue if d.event.start < t1 - 0.05 < d.event.end - 0.05]
            if not live:
                break
            t1 = max(e.end for e in live) + cuts["gap_before_next_line"]
        return round(min(t1, self.duration), 2)

    # -- window contents ------------------------------------------------------------
    def inside(self, t0: float, t1: float, include_camera: bool = False) -> list[Beat]:
        pool = self.beats if include_camera else self.content
        return [b for b in pool if b.event.start >= t0 - 0.05 and b.event.start < t1 - 0.05
                and b.event.end <= t1 + self.cfg["cuts"]["beat_overrun"]]

    def prev_dialogue(self, t0: float) -> Beat | None:
        prev = None
        for d in self.dialogue:
            if d.event.end <= t0 + 0.05:
                prev = d
        return prev

    def next_dialogue(self, t1: float) -> Beat | None:
        for d in self.dialogue:
            if d.event.start >= t1 - 0.05:
                return d
        return None


# -------------------------------------------------------------------------- scoring
def duration_score(d: float, cfg: dict) -> float:
    p = cfg["promo"]
    return math.exp(-0.5 * ((d - p["ideal_duration"]) / p["duration_tolerance"]) ** 2)


def hook_component(idx: EpisodeIndex, t0: float, t1: float) -> tuple[float, Beat | None]:
    win = idx.cfg["promo"]["hook_window"]
    best, best_b = 0.0, None
    for b in idx.beats:
        if t0 - 0.05 <= b.event.start <= t0 + win and b.event.start < t1:
            if b.hook > best:
                best, best_b = b.hook, b
    return best, best_b


def cliff_component(idx: EpisodeIndex, t0: float, t1: float) -> tuple[float, Beat | None, float]:
    win = idx.cfg["promo"]["cliff_window"]
    cc = idx.cfg["cliffhanger"]
    best, best_b = 0.0, None
    resolution = 0.0
    for b in idx.beats:
        e = b.event
        if e.start < t0 - 0.05 or e.start >= t1:
            continue
        end = min(e.end, t1)
        if end >= t1 - win:
            recency = cc["recency_floor"] + (1 - cc["recency_floor"]) * max(0.0, 1.0 - (t1 - end) / win)
            w = recency if e.kind != "camera" else recency * cc["camera_factor"]
            if b.cliff * w > best:
                best, best_b = b.cliff * w, b
            if end >= t1 - cc["resolution_window"]:
                resolution = max(resolution, b.resolution)
    best -= cc["resolution_penalty"] * resolution
    # Structural prior (turning points): microdrama episodes are written to end on a
    # cliffhanger, so a clip that ends where the episode ends inherits the writers' cut.
    # Stronger when the script itself marks the ending (sting, "cliffhanger", cut to black).
    # Treated as a floor on the cliffhanger score, reduced if the ending resolves calmly.
    p = idx.cfg["promo"]
    if t1 >= idx.duration - p["episode_end_tolerance"]:
        marked = any(e.event.end >= idx.duration - p["episode_end_marked_window"]
                     and "dramatic sting" in e.open_tags
                     for e in idx.beats)
        prior = p["episode_end_prior"] + (p["episode_end_marked_bonus"] if marked else 0.0)
        prior *= 1.0 - resolution
        if prior > best:
            best = prior
            last = [b for b in idx.beats if b.event.kind != "camera" and b.event.start < t1]
            best_b = best_b or (last[-1] if last else None)
    return max(0.0, min(1.0, best)), best_b, resolution


def energy_component(idx: EpisodeIndex, t0: float, t1: float, inside: list[Beat]) -> tuple[float, float]:
    if not inside:
        return 0.0, t1 - t0
    vals = [b.intensity for b in inside if b.event.kind in ("dialogue", "action")] or [0.0]
    mean_i = sum(vals) / len(vals)
    # longest stretch with no dialogue/action/sfx
    cursor, max_gap = t0, 0.0
    for b in sorted(inside, key=lambda b: b.event.start):
        max_gap = max(max_gap, b.event.start - cursor)
        cursor = max(cursor, b.event.end)
    max_gap = max(max_gap, t1 - cursor)
    en = idx.cfg["energy"]
    gap_pen = max(0.0, max_gap - idx.cfg["promo"]["dead_air_gap"]) * en["gap_penalty_per_second"]
    talk = sum(b.event.duration for b in inside if b.event.kind == "dialogue") / max(1.0, t1 - t0)
    e = en["intensity_share"] * mean_i + en["talk_share"] * min(1.0, talk * en["talk_scale"]) - gap_pen
    return max(0.0, min(1.0, e * en["gain"])), max_gap


def clarity_component(idx: EpisodeIndex, t0: float, t1: float, inside: list[Beat]) -> tuple[float, list[str]]:
    c = idx.cfg["clarity"]
    score, notes = c["base"], []
    lines = [b for b in inside if b.event.kind == "dialogue"]
    if lines:
        first = lines[0].event
        txt = first.text.strip()
        prev = idx.prev_dialogue(t0)
        if re.match(r"^(\.\.\.|…)", txt) or (
            prev and prev.event.text.rstrip().endswith(("...", "…"))
            and norm_speaker(prev.event.speaker) == norm_speaker(first.speaker)
            and t0 - prev.event.end < c["continuation_max_gap"]):
            score -= c["continuation_penalty"]; notes.append("starts mid-sentence")
        fw = re.sub(r"[^a-z']", "", (txt.split() or [""])[0].lower())
        if fw in c["generic_openers"]:
            score -= c["opener_penalty"]
        vo = sum(1 for b in lines if b.event.voice_over) / len(lines)
        if vo > c["voiceover_share"]:
            score -= c["voiceover_penalty"]; notes.append("mostly voice-over")
        speakers = {norm_speaker(b.event.speaker) for b in lines}
        if len(speakers) >= 2:
            score += c["conflict_bonus"]
    else:
        score -= c["no_dialogue_penalty"]; notes.append("no dialogue")
    scenes = {b.event.scene for b in inside}
    if len(scenes) > 1:
        score -= c["scene_jump_penalty"] * (len(scenes) - 1)
    if any(b.stakes >= c["stakes_threshold"] for b in inside):
        score += c["stakes_bonus"]
    return max(0.0, min(1.0, score)), notes


def score_window(idx: EpisodeIndex, t0: float, t1: float) -> dict:
    cfg = idx.cfg
    inside = idx.inside(t0, t1)
    hook, hb = hook_component(idx, t0, t1)
    cliff, cb, resolution = cliff_component(idx, t0, t1)
    energy, max_gap = energy_component(idx, t0, t1, inside)
    clarity, notes = clarity_component(idx, t0, t1, inside)
    dur = duration_score(t1 - t0, cfg)
    w = cfg["weights"]
    total = 100 * (w["hook"] * hook + w["cliffhanger"] * cliff + w["energy"] * energy
                   + w["clarity"] * clarity + w["duration"] * dur) / sum(w.values())
    return {
        "scores": {"hook": hook, "cliffhanger": cliff, "energy": energy, "clarity": clarity,
                   "duration": dur, "total": total},
        "hook_beat": hb, "cliff_beat": cb, "resolution": resolution,
        "max_gap": max_gap, "notes": notes, "inside": inside,
    }


# -------------------------------------------------------------------- text & flags
TAG_PHRASES = {
    "violence": "physical violence", "danger": "life-or-death danger", "betrayal": "a betrayal",
    "reveal": "a shocking reveal", "humiliation": "public humiliation",
    "romance": "a relationship at stake", "defiance": "a defiant stand",
    "question": "an unanswered question", "cut-off line": "a line cut off mid-thought",
    "threat": "an open threat", "dramatic sting": "a dramatic sting",
}


def describe_beat(b: Beat | None) -> str:
    if b is None:
        return ""
    e = b.event
    if e.kind == "dialogue":
        tag = " (V.O.)" if e.voice_over else ""
        return f'{e.speaker}{tag}: "{e.text}"'
    return f"[{e.kind}] {e.text}"


def beat_tags(b: Beat | None, include_open: bool) -> list[str]:
    if b is None:
        return []
    order = ["violence", "danger", "reveal", "betrayal", "humiliation", "romance", "defiance"]
    tags = [t for t in order if t in b.stake_tags]
    if include_open:
        tags = b.open_tags + tags
    return tags


def make_why(hook_tags, cliff_tags, scores, cfg) -> str:
    h = TAG_PHRASES.get(hook_tags[0], "a high-emotion line") if hook_tags else "a high-emotion line"
    open_tags = [t for t in cliff_tags if t in ("question", "cut-off line", "threat", "dramatic sting")]
    stake_tags = [t for t in cliff_tags if t not in open_tags]
    c = TAG_PHRASES.get(open_tags[0], "an unresolved beat") if open_tags else "an unresolved beat"
    fresh = [t for t in stake_tags if t not in hook_tags[:1]]
    if fresh:
        c += f" raising {TAG_PHRASES.get(fresh[0], fresh[0])}"
    extra = ""
    if scores.get("clarity", 0) >= cfg["flags"]["clear_for_new_viewer"]:
        extra = " The conflict is stated on screen, so a new viewer can follow it."
    elif scores.get("clarity", 0) < cfg["flags"]["needs_context"]:
        extra = " Needs context a new viewer may lack."
    return f"Opens on {h} and cuts on {c}, so the viewer has to see what happens next.{extra}"


def make_flags(idx_by_ep: dict, segs: list[Segment], info: dict, cfg: dict) -> list[str]:
    p = cfg["promo"]
    flags = []
    total = sum(s.duration for s in segs)
    if total < p["min_duration"] - 0.01 or total > p["max_duration"] + 0.01:
        flags.append("duration_out_of_range")
    s = info["scores"]
    hb = info.get("hook_beat")
    if hb is None or hb.event.start - segs[0].start > p["hook_window"] + 0.01:
        flags.append("hook_after_3s")
    if s["hook"] < cfg["flags"]["weak_hook"]:
        flags.append("weak_hook")
    if s["cliffhanger"] < cfg["flags"]["weak_cliffhanger"]:
        flags.append("weak_cliffhanger")
    if info.get("resolution", 0) >= cfg["flags"]["resolves"]:
        flags.append("resolves_before_cut")
    if info.get("max_gap", 0) > p["dead_air_gap"]:
        flags.append("dead_air")
    for n in info.get("notes", []):
        flags.append(n.replace(" ", "_").replace("-", "_"))
    for seg in segs:
        idx = idx_by_ep[seg.episode]
        for d in idx.dialogue:
            e = d.event
            # a cut point strictly inside a spoken line (with a small tolerance at each edge)
            tol = cfg["cuts"]["flag_line_tolerance"]
            if any(e.start + tol < b < e.end - tol for b in (seg.start, seg.end)):
                flags.append("cuts_through_dialogue"); break
        if idx.sensitive_re:
            for b in idx.inside(seg.start, seg.end, include_camera=True):
                if idx.sensitive_re.search(b.event.text):
                    flags.append("check_ad_policy"); break
    if len(segs) > 1:
        flags.append("multi_segment_editor_check")
    return sorted(set(flags))


# ------------------------------------------------------------------ candidate search
def iou(a: Segment, b: Segment) -> float:
    if a.episode != b.episode:
        return 0.0
    inter = max(0.0, min(a.end, b.end) - max(a.start, b.start))
    union = max(a.end, b.end) - min(a.start, b.start)
    return inter / union if union > 0 else 0.0


def promo_overlap(a: Promo, b: Promo) -> float:
    return max((iou(x, y) for x in a.segments for y in b.segments), default=0.0)


def windows(idx: EpisodeIndex, lo: float, hi: float):
    """Yield (t0, t1, start_beat, end_beat) for every beat pair giving lo <= t1-t0 <= hi."""
    seen = set()
    starts = idx.content
    ends = [b for b in idx.content if b.event.kind in ("dialogue", "action", "sfx")]
    snapped_end = {b.id: idx.snap_end(b) for b in ends}
    for sb in starts:
        t0 = idx.snap_start(sb)
        for eb in ends:
            if eb.event.end <= sb.event.start:
                continue
            t1 = snapped_end[eb.id]
            d = t1 - t0
            if d < lo or d > hi:
                continue
            key = (round(t0, 1), round(t1, 1))
            if key in seen:
                continue
            seen.add(key)
            yield t0, t1, sb, eb


def build_promo(pid: str, idx_by_ep: dict, segs: list[Segment], info: dict, cfg: dict, kind: str) -> Promo:
    hb, cb = info["hook_beat"], info["cliff_beat"]
    htags, ctags = beat_tags(hb, False), beat_tags(cb, True)
    speakers = []
    for seg in segs:
        for b in idx_by_ep[seg.episode].inside(seg.start, seg.end):
            if b.event.kind == "dialogue" and b.event.speaker not in speakers:
                speakers.append(b.event.speaker)
    return Promo(
        id=pid, kind=kind, segments=segs, scores=info["scores"],
        hook_event=hb.id if hb else "", cliff_event=cb.id if cb else "",
        hook=describe_beat(hb), cliffhanger=describe_beat(cb),
        why=make_why(htags, ctags, info["scores"], cfg),
        flags=make_flags(idx_by_ep, segs, info, cfg),
        hook_tags=htags, cliff_tags=ctags, speakers=speakers,
    )


def select_non_overlapping(cands: list[Promo], k: int, max_iou: float) -> list[Promo]:
    kept: list[Promo] = []
    for c in sorted(cands, key=lambda p: -p.scores.get("final", p.total)):
        if all(promo_overlap(c, k_) <= max_iou for k_ in kept):
            kept.append(c)
        if len(kept) >= k:
            break
    return kept


def episode_candidates(idx: EpisodeIndex, idx_by_ep: dict, keep: int) -> list[Promo]:
    cfg = idx.cfg
    p = cfg["promo"]
    cands = []
    n = 0
    for t0, t1, sb, eb in windows(idx, p["min_duration"], p["max_duration"]):
        info = score_window(idx, t0, t1)
        if idx.ml is not None:
            idx.ml.apply(idx, t0, t1, info)
        seg = Segment(idx.ep.number, t0, t1, sb.id, eb.id)
        cands.append((info["scores"]["total"], seg, info))
    cands.sort(key=lambda x: -x[0])
    # greedy non-overlapping selection on the raw windows (cheap), then build promos
    kept = []
    for total, seg, info in cands:
        if all(iou(seg, k[1]) <= p["overlap_iou"] for k in kept):
            kept.append((total, seg, info))
        if len(kept) >= keep:
            break
    out = []
    for i, (total, seg, info) in enumerate(kept):
        n += 1
        out.append(build_promo(f"EP{idx.ep.number:02d}-{i + 1}", idx_by_ep, [seg], info, cfg, "single"))
    return out


# ------------------------------------------------------------------------- montages
def segment_pool(idx: EpisodeIndex, lo: float, hi: float, role: str):
    """Best hook segments (role='hook') or cliffhanger segments (role='cliff') of one episode."""
    m = idx.cfg["montage"]
    k = m["pool_size"]
    pool = []
    for t0, t1, sb, eb in windows(idx, lo, hi):
        info = score_window(idx, t0, t1)
        s = info["scores"]
        if role == "hook":
            val = sum(wt * s[comp] for comp, wt in m["hook_value"].items())
            if info["hook_beat"] is None or s["hook"] < m["hook_min"]:
                continue
        else:
            val = sum(wt * s[comp] for comp, wt in m["cliff_value"].items())
            if s["cliffhanger"] < m["cliff_min"] or info["resolution"] >= idx.cfg["flags"]["resolves"]:
                continue
        pool.append((val, Segment(idx.ep.number, t0, t1, sb.id, eb.id), info))
    pool.sort(key=lambda x: -x[0])
    kept = []
    for item in pool:
        if all(iou(item[1], k_[1]) <= m["pool_overlap_iou"] for k_ in kept):
            kept.append(item)
        if len(kept) >= k:
            break
    return kept


def build_montages(indexes: list[EpisodeIndex], idx_by_ep: dict, cfg: dict) -> list[Promo]:
    m = cfg["montage"]
    p = cfg["promo"]
    w = cfg["weights"]
    hooks, cliffs = {}, {}
    for idx in indexes:
        hooks[idx.ep.number] = segment_pool(idx, *m["hook_segment"], role="hook")
        cliffs[idx.ep.number] = segment_pool(idx, *m["cliff_segment"], role="cliff")
    cands = []
    eps = sorted(idx_by_ep)
    for i in eps:
        for j in eps:
            if j < i or j - i > p["montage_max_episode_gap"]:
                continue
            for hv, hs, hi in hooks.get(i, []):
                for cv, cs, ci in cliffs.get(j, []):
                    if i == j and cs.start < hs.end + m["same_episode_min_gap"]:
                        continue   # too close: that is just a single clip
                    total_d = hs.duration + cs.duration
                    if total_d < p["min_duration"] or total_d > p["max_duration"]:
                        continue
                    sp_h = {norm_speaker(b.event.speaker) for b in hi["inside"] if b.event.kind == "dialogue"}
                    sp_c = {norm_speaker(b.event.speaker) for b in ci["inside"] if b.event.kind == "dialogue"}
                    shared = len(sp_h & sp_c)
                    coherence = (m["shared_character_bonus"] if shared else -m["shared_character_bonus"]) \
                        - m["gap_penalty_per_episode"] * (j - i)
                    hsc, csc = hi["scores"], ci["scores"]
                    clarity = max(0.0, min(1.0, 0.5 * hsc["clarity"] + 0.5 * csc["clarity"] + coherence))
                    energy = (hsc["energy"] * hs.duration + csc["energy"] * cs.duration) / total_d
                    dur = duration_score(total_d, cfg)
                    total = 100 * (w["hook"] * hsc["hook"] + w["cliffhanger"] * csc["cliffhanger"]
                                   + w["energy"] * energy + w["clarity"] * clarity
                                   + w["duration"] * dur) / sum(w.values())
                    info = {
                        "scores": {"hook": hsc["hook"], "cliffhanger": csc["cliffhanger"],
                                   "energy": energy, "clarity": clarity, "duration": dur,
                                   "total": total},
                        "hook_beat": hi["hook_beat"], "cliff_beat": ci["cliff_beat"],
                        "resolution": ci["resolution"], "max_gap": max(hi["max_gap"], ci["max_gap"]),
                        "notes": hi["notes"],
                    }
                    cands.append((total, [hs, cs], info))
    cands.sort(key=lambda x: -x[0])
    out: list[Promo] = []
    used: list[Segment] = []
    for total, segs, info in cands:
        if any(iou(s, u) > m["pool_overlap_iou"] for s in segs for u in used):
            continue
        used.extend(segs)
        pid = f"MX{len(out) + 1:02d}"
        out.append(build_promo(pid, idx_by_ep, segs, info, cfg, "multi"))
        if len(out) >= p["max_montages"]:
            break
    return out


# -------------------------------------------------------------------------- ranking
def rank_promos(promos: list[Promo], cfg: dict) -> list[Promo]:
    r = cfg["ranking"]
    for pr in promos:
        bonus = r["early_episode_bonus"] if min(pr.episodes) <= r["early_episodes"] else 0.0
        pen = r["montage_penalty"] if pr.kind == "multi" else 0.0
        hard = sum(r["rule_violation_penalty"] for f in pr.flags if f in ("hook_after_3s", "cuts_through_dialogue",
                                                   "duration_out_of_range"))
        base = pr.scores.get("final", pr.total)
        pr.rank_score = base + bonus - pen - hard
        parts = [f"score {base:.0f}/100", f"hook {pr.scores['hook']:.2f}",
                 f"cliffhanger {pr.scores['cliffhanger']:.2f}"]
        if bonus:
            parts.append(f"early episode (+{bonus:g}: sends new viewers to the start)")
        if pen:
            parts.append(f"multi-segment (-{pen:g}: needs an editor's cut)")
        if hard:
            parts.append(f"rule violations (-{hard:g})")
        pr.rank_reason = "; ".join(parts)
    promos.sort(key=lambda p: -p.rank_score)
    for i, pr in enumerate(promos, 1):
        pr.rank = i
    return promos


def post_first(promos: list[Promo], cfg: dict) -> list[Promo]:
    """Top list with variety: one promo per episode, at most two multi-segment ones."""
    n = cfg["promo"]["post_first"]
    chosen, eps, multis = [], set(), 0
    for pr in promos:
        if any(f in pr.flags for f in ("hook_after_3s", "cuts_through_dialogue", "duration_out_of_range")):
            continue
        if set(pr.episodes) & eps:
            continue
        if pr.kind == "multi" and multis >= 2:
            continue
        chosen.append(pr)
        eps |= set(pr.episodes)
        multis += pr.kind == "multi"
        if len(chosen) >= n:
            break
    return chosen


# ---------------------------------------------------------------------------- run
def find_promos(episodes: list[Episode], cfg: dict, llm=None, progress=None, ml=None) -> dict:
    from .genre import promo_moods, resolve_genre
    cfg, genre = resolve_genre(episodes, cfg)
    if llm is not None and hasattr(llm, "set_genre"):
        llm.set_genre(cfg, genre)
    indexes = [EpisodeIndex(ep, cfg) for ep in episodes]
    for i in indexes:
        i.ml = ml
    idx_by_ep = {i.ep.number: i for i in indexes}
    per_ep = cfg["promo"]["per_episode"]
    shortlist = max(per_ep, cfg["llm"].get("shortlist_per_episode", per_ep)) if llm else per_ep
    promos: list[Promo] = []
    for n, idx in enumerate(indexes, 1):
        cands = episode_candidates(idx, idx_by_ep, shortlist)
        if llm is not None:
            cands = llm.review_episode(idx, cands, idx_by_ep)
            cands = select_non_overlapping(cands, per_ep, cfg["promo"]["overlap_iou"])
        promos.extend(cands[:per_ep])
        if progress:
            progress(n, len(indexes), f"Episode {idx.ep.number}")
    montages = build_montages(indexes, idx_by_ep, cfg) if cfg["promo"]["max_montages"] > 0 else []
    if llm is not None and montages:
        montages = llm.review_montages(montages, idx_by_ep)
    promos.extend(montages)
    promos = rank_promos(promos, cfg)
    if llm is not None and cfg["llm"].get("pairwise_rank_top", 0):
        promos = llm.pairwise_rerank(promos, idx_by_ep, cfg["llm"]["pairwise_rank_top"])
    for p in promos:
        p.moods = promo_moods(p, idx_by_ep, cfg)
    top = post_first(promos, cfg)
    return {"promos": promos, "post_first": top, "indexes": idx_by_ep, "genre": genre, "cfg": cfg}
