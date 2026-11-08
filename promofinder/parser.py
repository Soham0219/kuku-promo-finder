"""Parse a Kuku-style timestamped screenplay into a list of events.

Each block in the script looks like:

    **MAYA (Outraged, Accusing):**
    Start: (00:12.40) | End: (00:15.00)
    "You sold the company behind my back!"

Blocks are CAMERA, ACTION, SOUND FX, or a speaking character. Everything the
rest of the tool does works off these events, so every timestamp in the output
can be traced back to a real line in the script.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict
from pathlib import Path

KIND_BY_LABEL = {
    "CAMERA": "camera",
    "ACTION": "action",
    "SOUND FX": "sfx",
    "SFX": "sfx",
    "SOUND": "sfx",
    "MUSIC": "sfx",
}

# Parenthetical tokens that describe delivery, not emotion.
VO_TOKENS = {"v.o.", "vo", "voice over", "voice-over"}
OS_TOKENS = {"o.s.", "os", "off-screen", "off screen", "offscreen"}
IGNORE_TOKENS = {"continues", "cont'd", "cont.", "contd"}

TIME_RE = re.compile(r"(?:(\d+):)?(\d+):(\d+(?:\.\d+)?)")
BLOCK_RE = re.compile(r"^\*\*(?P<label>.+?):?\*\*\s*:?\s*$")
STAMP_RE = re.compile(
    r"start\s*:\s*\(?\s*(?P<s>[\d:.]+)\s*\)?\s*\|\s*end\s*:\s*\(?\s*(?P<e>[\d:.]+)\s*\)?",
    re.I,
)


def parse_time(value: str) -> float:
    """'01:23.45' -> 83.45 ; '1:02:03.5' -> 3723.5"""
    m = TIME_RE.search(value.strip())
    if not m:
        raise ValueError(f"Bad timestamp: {value!r}")
    h = int(m.group(1) or 0)
    return h * 3600 + int(m.group(2)) * 60 + float(m.group(3))


def fmt_time(seconds: float) -> str:
    """83.45 -> '01:23.45' (same style as the scripts)."""
    seconds = max(0.0, seconds)
    m = int(seconds // 60)
    s = seconds - m * 60
    return f"{m:02d}:{s:05.2f}"


@dataclass
class Event:
    id: str
    episode: int
    scene: int
    index: int
    kind: str              # camera | action | dialogue | sfx
    start: float
    end: float
    text: str
    speaker: str = ""      # normalised speaker name for dialogue
    cue: str = ""          # emotion/delivery cue, e.g. "Outraged, Accusing"
    voice_over: bool = False
    off_screen: bool = False

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Scene:
    number: int
    heading: str
    start: float
    end: float


@dataclass
class Episode:
    number: int
    title: str
    logline: str
    continuity: str
    source: str
    scenes: list[Scene] = field(default_factory=list)
    events: list[Event] = field(default_factory=list)

    @property
    def duration(self) -> float:
        ends = [e.end for e in self.events] + [s.end for s in self.scenes]
        return max(ends) if ends else 0.0

    @property
    def dialogue(self) -> list[Event]:
        return [e for e in self.events if e.kind == "dialogue"]

    @property
    def shots(self) -> list[Event]:
        return [e for e in self.events if e.kind == "camera"]

    def get(self, event_id: str) -> Event | None:
        for e in self.events:
            if e.id == event_id:
                return e
        return None


def _split_label(label: str) -> tuple[str, str, bool, bool]:
    """'MAYA (V.O.) (Reflective, Vulnerable)' -> name, cue, vo, os"""
    name = re.split(r"\(", label, maxsplit=1)[0].strip().rstrip(":").strip()
    groups = re.findall(r"\(([^)]*)\)", label)
    cue_parts: list[str] = []
    vo = os_ = False
    for g in groups:
        for tok in re.split(r"[,/]", g):
            t = tok.strip()
            low = t.lower()
            if not t:
                continue
            if low in VO_TOKENS:
                vo = True
            elif low in OS_TOKENS:
                os_ = True
            elif low in IGNORE_TOKENS:
                continue
            else:
                cue_parts.append(t)
    return name, ", ".join(cue_parts), vo, os_


def parse_text(text: str, source: str = "<text>") -> Episode:
    lines = text.splitlines()

    def header(key: str) -> str:
        for ln in lines[:15]:
            if ln.upper().startswith(key):
                return ln.split(":", 1)[1].strip() if ":" in ln else ""
        return ""

    title = header("EPISODE NUMBER") or Path(source).stem
    m = re.search(r"(\d+)", title)
    if not m:
        m = re.search(r"episode[-_ ]?0*(\d+)", Path(source).name, re.I)
    number = int(m.group(1)) if m else 0
    ep = Episode(
        number=number,
        title=title,
        logline=header("EPISODE LOGLINE"),
        continuity=header("CONTINUITY NOTES"),
        source=Path(source).name,
    )

    scene_no = 0
    pending_scene: dict = {}
    i = 0
    idx = 0
    while i < len(lines):
        ln = lines[i].strip()
        sm = re.match(r"^scene\s+(\d+)\s*$", ln, re.I)
        if sm:
            if pending_scene:
                ep.scenes.append(Scene(**pending_scene))
            scene_no = int(sm.group(1))
            pending_scene = {"number": scene_no, "heading": "", "start": 0.0, "end": 0.0}
            i += 1
            continue
        hm = re.match(r"^scene\s+(heading|start|end)\s*:\s*(.*)$", ln, re.I)
        if hm and pending_scene:
            key, val = hm.group(1).lower(), hm.group(2).strip()
            if key == "heading":
                pending_scene["heading"] = val
            else:
                try:
                    pending_scene[key] = parse_time(val)
                except ValueError:
                    pass
            i += 1
            continue

        bm = BLOCK_RE.match(ln)
        if bm:
            label = bm.group("label").strip()
            # find the Start/End line (normally the next line)
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                j += 1
            stamp = STAMP_RE.search(lines[j]) if j < len(lines) else None
            if not stamp:
                i += 1
                continue
            start, end = parse_time(stamp.group("s")), parse_time(stamp.group("e"))
            k = j + 1
            body: list[str] = []
            while k < len(lines) and lines[k].strip() and not BLOCK_RE.match(lines[k].strip()) \
                    and not re.match(r"^scene\s+\d+", lines[k].strip(), re.I) \
                    and not lines[k].strip().startswith("---"):
                body.append(lines[k].strip())
                k += 1
            text_body = " ".join(body).strip()
            name, cue, vo, os_ = _split_label(label)
            kind = KIND_BY_LABEL.get(name.upper(), "dialogue")
            if kind == "dialogue":
                text_body = text_body.strip().strip('"“”').strip()
            if end < start:
                start, end = end, start
            ep.events.append(Event(
                id=f"E{number:02d}-{idx:03d}",
                episode=number,
                scene=scene_no or 1,
                index=idx,
                kind=kind,
                start=round(start, 2),
                end=round(end, 2),
                text=text_body,
                speaker=name.title() if kind == "dialogue" else "",
                cue=cue,
                voice_over=vo,
                off_screen=os_,
            ))
            idx += 1
            i = k
            continue
        i += 1

    if pending_scene:
        ep.scenes.append(Scene(**pending_scene))
    if not ep.scenes and ep.events:
        ep.scenes.append(Scene(1, "", 0.0, ep.duration))
    # Stable ordering by start time (scripts interleave camera/action/dialogue).
    order = {"camera": 0, "sfx": 1, "action": 2, "dialogue": 3}
    ep.events.sort(key=lambda e: (e.start, order.get(e.kind, 9), e.index))
    renumber(ep, number)
    return ep


def event_id(episode: int, n: int) -> str:
    return f"E{episode:02d}-{n:03d}"


def episode_of(eid: str) -> int:
    """'E07-012' -> 7, 'E104-003' -> 104 (works for any number of digits)."""
    return int(eid.split("-", 1)[0][1:])


def renumber(ep: Episode, number: int) -> None:
    """Set an episode's number and rebuild every event ID from it."""
    ep.number = number
    for n, e in enumerate(ep.events):
        e.index = n
        e.episode = number
        e.id = event_id(number, n)


def ensure_unique_numbers(eps: list[Episode]) -> list[Episode]:
    """Episodes without a number (0) or with a duplicate number get the next free number,
    in input order, so no two episodes ever share IDs. Returns them sorted by number."""
    used: set[int] = set()
    needs = []
    for ep in eps:
        if ep.number <= 0 or ep.number in used:
            needs.append(ep)
        else:
            used.add(ep.number)
    nxt = max(used, default=0) + 1
    for ep in needs:
        while nxt in used:
            nxt += 1
        renumber(ep, nxt)
        used.add(nxt)
    eps.sort(key=lambda e: e.number)
    return eps


def parse_file(path: str | Path) -> Episode:
    p = Path(path)
    return parse_text(p.read_text(encoding="utf-8", errors="replace"), source=str(p))


def load_episodes(paths) -> list[Episode]:
    return ensure_unique_numbers([parse_file(p) for p in paths])


def compact_transcript(ep: Episode, include_camera: bool = False) -> str:
    """Short, ID-tagged transcript used for the LLM prompt and for reading."""
    out = [f"EPISODE {ep.number}: {ep.logline}"]
    for e in ep.events:
        if e.kind == "camera" and not include_camera:
            continue
        t = f"[{e.id} {fmt_time(e.start)}-{fmt_time(e.end)}]"
        if e.kind == "dialogue":
            tag = " (V.O.)" if e.voice_over else (" (O.S.)" if e.off_screen else "")
            out.append(f'{t} {e.speaker.upper()}{tag} <{e.cue}>: "{e.text}"')
        else:
            out.append(f"{t} {e.kind.upper()}: {e.text}")
    return "\n".join(out)
