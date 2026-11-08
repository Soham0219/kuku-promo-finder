"""Beat-level signals: intensity, stakes, openness, resolution -> hook & cliffhanger scores.

Theory (see README): a hook is *surprise* (a big jump in what the viewer believes),
a cliffhanger is *suspense* (high uncertainty about what happens next), following
Ely, Frankel & Kamenica (2015) and Wilmot & Keller (2020). We approximate both with
cheap, explainable text signals whose word lists and weights live in config.yaml.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .config import terms
from .parser import Episode, Event


@dataclass
class Beat:
    event: Event
    intensity: float = 0.0
    stakes: float = 0.0
    stake_tags: dict = field(default_factory=dict)   # category -> matched word
    openness: float = 0.0
    open_tags: list = field(default_factory=list)
    resolution: float = 0.0
    hook: float = 0.0
    cliff: float = 0.0
    words: int = 0

    @property
    def id(self) -> str:
        return self.event.id


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _clean(text: str) -> str:
    return text.strip().strip('"“”').strip()


def stakes_of(text: str, cfg: dict, scale: float = 1.0) -> tuple[float, dict]:
    """Noisy-OR over stake categories: S = 1 - prod_k (1 - w_k * present_k)."""
    prod = 1.0
    tags = {}
    for cat, spec in cfg["stakes"].items():
        m = terms(spec["words"]).search(text)
        if m:
            tags[cat] = m.group(0).lower()
            prod *= 1.0 - spec["weight"] * scale
    return 1.0 - prod, tags


def score_event(ev: Event, cfg: dict) -> Beat:
    b = cfg["beat"]
    cues = cfg["cues"]
    text = _clean(ev.text)
    low_text = text.lower()
    beat = Beat(ev, words=len(text.split()))
    scale = b["camera_text_weight"] if ev.kind == "camera" else 1.0

    # --- stakes (what is at risk / shocking content) -------------------------
    beat.stakes, beat.stake_tags = stakes_of(text, cfg, scale)

    # --- intensity -------------------------------------------------------------
    high, mid, low = terms(cues["high"]), terms(cues["mid"]), terms(cues["low"])
    ic = b["intensity"]
    if ev.kind == "dialogue":
        cue = ev.cue.lower()
        i = ic["dialogue_base"]
        if high.search(cue):
            i += ic["high_cue"]
        elif mid.search(cue):
            i += ic["mid_cue"]
        elif low.search(cue):
            i += ic["low_cue"]
        if "!" in text:
            i += b["exclaim_bonus"]
        beat.intensity = _clip01(i)
    else:
        i = (ic["action_base"] + ic["action_high_word"] * (1 if high.search(low_text) else 0)
             + ic["action_stakes"] * beat.stakes)
        if terms(ic["sudden_words"]).search(low_text):
            i += ic["action_sudden"]
        beat.intensity = _clip01(i * (scale if ev.kind == "camera" else 1.0))

    # --- openness (unresolved / suspense) ------------------------------------
    o = cfg["openness"]
    parts = []
    if ev.kind == "dialogue":
        stripped = text.rstrip()
        if stripped.endswith("?") or stripped.endswith("?!"):
            parts.append(o["question"]); beat.open_tags.append("question")
        if re.search(r"(\.\.\.|…|—|--|-)$", stripped):
            parts.append(o["trailing"]); beat.open_tags.append("cut-off line")
        if beat.stakes >= b["threat_min_stakes"] and terms(o["threat_words"]).search(low_text):
            parts.append(o["threat_weight"] * beat.stakes); beat.open_tags.append("threat")
    else:
        if terms(o["sfx_words"]).search(low_text):
            parts.append(o["sfx"] * (scale if ev.kind == "camera" else 1.0))
            beat.open_tags.append("dramatic sting")
    prod = 1.0
    for p in parts:
        prod *= 1.0 - _clip01(p)
    beat.openness = 1.0 - prod

    # --- resolution (calm, closure) -------------------------------------------
    if terms(cfg["resolution"]["words"]).search(low_text):
        beat.resolution = 1.0 * (scale if ev.kind == "camera" else 1.0)
    if ev.kind == "dialogue" and low.search(ev.cue.lower()) and not high.search(ev.cue.lower()):
        beat.resolution = max(beat.resolution, b["low_cue_resolution"])

    # --- hook ("surprise") ------------------------------------------------------
    h = b["hook_intensity"] * beat.intensity + b["hook_stakes"] * beat.stakes
    if ev.kind == "dialogue":
        if beat.words <= b["punchy_words"]:
            h += b["hook_punchy_bonus"]
        if "question" in beat.open_tags:
            h += b["hook_question_bonus"]
        if re.match(r"^(\.\.\.|…)", text):
            h -= b["hook_continuation_penalty"]
        if ev.voice_over:
            h -= b["hook_voiceover_penalty"]
    beat.hook = _clip01(h)

    # --- cliffhanger ("suspense") ----------------------------------------------
    c = (b["cliff_open"] * beat.openness + b["cliff_stakes"] * beat.stakes
         + b["cliff_intensity"] * beat.intensity - b["cliff_resolution_penalty"] * beat.resolution)
    beat.cliff = _clip01(c)
    return beat


def score_episode(ep: Episode, cfg: dict) -> list[Beat]:
    return [score_event(e, cfg) for e in ep.events]
