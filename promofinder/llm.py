"""Optional LLM layer. Swappable provider, strict validation.

The LLM acts as a senior promo editor on the rules' shortlist:
  * review_episode: rates each candidate (hook / cliffhanger / stands-alone, 1-10), writes
    the hook, cliffhanger and one-line "why", may move a cut to a different line, and may
    propose new promos. It may only refer to line IDs from the transcript; code turns IDs
    into timestamps, so the model can never invent a timestamp.
  * review_montages: the same for multi-segment promos.
  * pairwise_rerank: orders the top promos with pairwise comparisons in both orders
    (Pairwise Ranking Prompting, Qin et al., Findings of NAACL 2024), which is more reliable than asking a
    model for absolute scores.

Any API error, bad JSON or invalid ID falls back to the rule-based result and is flagged.
Responses are cached on disk, so re-running costs nothing.

Providers use plain HTTPS through one function, `http_post`, with no SDKs, so swapping a model is one
setting (and the website swaps in a browser transport):
  gemini | groq | openai | anthropic | openai_compatible | ollama | mock
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from pathlib import Path


from .engine import (EpisodeIndex, Promo, Segment, build_promo, score_window, make_flags)
from .parser import fmt_time

CACHE_DIR = Path(os.environ.get("PROMO_CACHE", Path(__file__).resolve().parent.parent / ".cache" / "llm"))

# First entry is the default; later entries are tried automatically if a model name is not
# found (model names change often). Pass --model to use exactly one model.
MODEL_CHAIN = {
    "gemini": ["gemini-3.8-flash", "gemini-3.5-flash", "gemini-3.1-flash-lite"],
    "groq": ["llama-3.3-70b-versatile", "openai/gpt-oss-120b"],
    "openai": ["gpt-5-mini", "gpt-4.1-mini", "gpt-4o-mini"],
    "anthropic": ["claude-haiku-5-5", "claude-sonnet-5-5"],
    "openai_compatible": [""],
    "ollama": ["llama3.1"],
    "mock": ["mock"],
}
DEFAULT_MODELS = {k: v[0] for k, v in MODEL_CHAIN.items()}
KEY_ENV = {
    "gemini": ["GEMINI_API_KEY", "GOOGLE_API_KEY"],
    "groq": ["GROQ_API_KEY"],
    "openai": ["OPENAI_API_KEY"],
    "anthropic": ["ANTHROPIC_API_KEY"],
    "openai_compatible": ["LLM_API_KEY", "OPENAI_API_KEY"],
}

SYSTEM = """You are a senior promo editor for vertical short-drama ads (TikTok, Reels, Shorts).
A promo is a 20-60 second clip cut from the show. A great promo:
- HOOK: something in the first 3 seconds makes a scrolling viewer stop (a slap, a reveal, a shocking line).
- CLIFFHANGER: it cuts at an unresolved moment so the viewer needs to know what happens next.
- STANDS ALONE: someone who has never seen the show understands the conflict.
Rules:
- Refer to moments ONLY by the line IDs given in the transcript (like E03-045). Never invent IDs or timestamps.
- Judge only what is in the script. Be strict: most clips are average (5-6/10).
- Reply with a single JSON object and nothing else."""


class LLMError(RuntimeError):
    pass


class HTTPStatusError(Exception):
    """A non-2xx reply from a provider."""
    def __init__(self, status: int, text: str = "", headers: dict | None = None):
        super().__init__(f"HTTP {status}: {text[:200]}")
        self.status, self.text, self.headers = status, text, headers or {}


class HTTPResponse:
    def __init__(self, status: int, text: str, headers: dict | None = None):
        self.status, self.text, self.headers = status, text, {k.lower(): v for k, v in (headers or {}).items()}

    def json(self):
        return json.loads(self.text)

    def raise_for_status(self):
        if self.status >= 400:
            raise HTTPStatusError(self.status, self.text, self.headers)


def _requests_post(url: str, headers: dict, body: dict, params: dict | None, timeout: float) -> HTTPResponse:
    import requests   # imported here so the browser build needs no requests library
    r = requests.post(url, headers=headers, json=body, params=params, timeout=timeout)
    return HTTPResponse(r.status_code, r.text, dict(r.headers))


# The one place network calls go through. Tests and the website replace it.
http_post = _requests_post


def _post(url: str, body: dict, headers: dict | None = None, timeout: float = 90) -> HTTPResponse:
    r = http_post(url, {"content-type": "application/json", **(headers or {})}, body, None, timeout)
    r.raise_for_status()
    return r


# ---------------------------------------------------------------------- providers
class Provider:
    def __init__(self, name: str, model: str = "", api_key: str = "", base_url: str = "",
                 temperature: float = 0.2, max_retries: int = 2, timeout: int = 90):
        if name not in MODEL_CHAIN:
            raise LLMError(f"Unknown provider {name!r}")
        self.name = name
        self.models = [model] if model else list(MODEL_CHAIN[name])
        self.model = self.models[0]
        self.api_key = api_key or next((os.environ[k] for k in KEY_ENV.get(name, []) if os.environ.get(k)), "")
        self.base_url = base_url
        self.temperature = temperature
        self.send_temperature = True
        self.max_retries = max_retries
        self.timeout = timeout
        self.calls = 0
        self.cache_hits = 0
        self.notes: list[str] = []
        if name in KEY_ENV and name != "openai_compatible" and not self.api_key:
            raise LLMError(f"No API key for {name}. Set {KEY_ENV[name][0]} or pass it in.")
        if name == "openai_compatible" and not base_url:
            raise LLMError("openai_compatible needs base_url (e.g. https://host/v1)")
        if name == "openai_compatible" and not self.model:
            raise LLMError("openai_compatible needs --model")

    @property
    def label(self) -> str:
        return f"{self.name}:{self.model}"

    # one JSON completion, cached
    def complete_json(self, system: str, user: str) -> dict:
        key = hashlib.sha256(f"{self.label}\n{system}\n{user}".encode()).hexdigest()[:32]
        path = CACHE_DIR / f"{key}.json"
        try:
            if path.exists():
                self.cache_hits += 1
                return json.loads(path.read_text())
        except Exception:
            pass                                   # unreadable cache entry: just call again
        last = None
        attempt = 0
        while attempt <= self.max_retries:
            try:
                data = parse_json(self._call(system, user))
                self.calls += 1
                try:
                    CACHE_DIR.mkdir(parents=True, exist_ok=True)
                    path.write_text(json.dumps(data))
                except OSError:
                    pass                           # read-only disk: caching is optional
                return data
            except HTTPStatusError as e:
                last = e
                body = e.text.lower()
                code = e.status
                if code in (400, 404) and "temperature" in body and self.send_temperature:
                    self.send_temperature = False  # some reasoning models reject temperature
                    self.notes.append(f"{self.model}: retrying without temperature")
                    continue
                if code in (400, 404) and "model" in body and self._next_model():
                    continue                       # model name not available: try the next one
                if code in (401, 403):
                    break                          # bad key: retrying will not help
                try:
                    wait = float(e.headers.get("retry-after", 0) or 0)
                except ValueError:
                    wait = 0.0
                time.sleep(min(60.0, max(wait, 5.0 * 2 ** attempt if code == 429 else 2.0 * 2 ** attempt)))
            except Exception as e:                 # network error, bad JSON
                last = e
                if attempt < self.max_retries:
                    time.sleep(min(20.0, 2.0 * 2 ** attempt))
            attempt += 1
        raise LLMError(f"{self.label} failed: {last}")

    def _next_model(self) -> bool:
        i = self.models.index(self.model)
        if i + 1 < len(self.models):
            self.notes.append(f"model {self.model} unavailable, switched to {self.models[i + 1]}")
            self.model = self.models[i + 1]
            return True
        return False

    def _temp(self) -> dict:
        return {"temperature": self.temperature} if self.send_temperature else {}

    def _call(self, system: str, user: str) -> str:
        n = self.name
        if n == "mock":
            return mock_answer(user)
        if n == "gemini":
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
            r = _post(url, headers={"x-goog-api-key": self.api_key}, timeout=self.timeout, body={
                "systemInstruction": {"parts": [{"text": system}]},
                "contents": [{"role": "user", "parts": [{"text": user}]}],
                "generationConfig": {**self._temp(), "responseMimeType": "application/json"}})
            parts = r.json()["candidates"][0]["content"]["parts"]
            return "".join(pt.get("text", "") for pt in parts if not pt.get("thought"))
        if n == "anthropic":
            r = _post("https://api.anthropic.com/v1/messages", timeout=self.timeout,
                      headers={"x-api-key": self.api_key, "anthropic-version": "2023-06-01",
                               # required when the call comes from a browser (the website)
                               "anthropic-dangerous-direct-browser-access": "true"},
                      body={"model": self.model, "max_tokens": 4000, "system": system, **self._temp(),
                            "messages": [{"role": "user", "content": user}]})
            return "".join(b.get("text", "") for b in r.json()["content"] if b.get("type") == "text")
        if n == "ollama":
            base = self.base_url or "http://localhost:11434"
            r = _post(f"{base}/api/chat", timeout=self.timeout * 3, body={
                "model": self.model, "stream": False, "format": "json", "options": self._temp(),
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]})
            return r.json()["message"]["content"]
        # OpenAI-style chat completions: openai, groq, openai_compatible
        base = {"openai": "https://api.openai.com/v1", "groq": "https://api.groq.com/openai/v1"}.get(n, self.base_url)
        body = {"model": self.model, **self._temp(),
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                "response_format": {"type": "json_object"}}
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        r = _post(f"{base.rstrip('/')}/chat/completions", body, headers=headers, timeout=self.timeout)
        return r.json()["choices"][0]["message"]["content"]


def parse_json(text: str) -> dict:
    text = text.strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.M).strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.S)
        if not m:
            raise
        data = json.loads(m.group(0))
    if not isinstance(data, dict):
        raise ValueError("model reply is not a JSON object")
    return data


def mock_answer(user: str) -> str:
    """Deterministic stand-in used for tests and offline demos. Not a real judgement."""
    if '"winner"' in user:
        return json.dumps({"winner": "A", "reason": "mock"})
    ids = re.findall(r"^(C\d+|M\d+):", user, re.M)
    return json.dumps({"candidates": [
        {"id": i, "hook": 6, "cliffhanger": 6, "standalone": 6,
         "hook_text": "(mock) hook", "cliffhanger_text": "(mock) cliffhanger", "why": "(mock) why"}
        for i in ids], "new": []})


# ------------------------------------------------------------------------- prompts
def transcript(idx: EpisodeIndex, t0: float = 0.0, t1: float = 1e9, camera: bool = True) -> str:
    out = []
    for b in idx.beats:
        e = b.event
        if e.start < t0 - 0.05 or e.start >= t1:
            continue
        if e.kind == "camera" and not camera:
            continue
        ts = f"{fmt_time(e.start)}-{fmt_time(e.end)}"
        if e.kind == "dialogue":
            vo = " (V.O.)" if e.voice_over else ""
            out.append(f'{e.id} [{ts}] {e.speaker.upper()}{vo} ({e.cue}): "{e.text}"')
        else:
            out.append(f"{e.id} [{ts}] {e.kind.upper()}: {e.text}")
    return "\n".join(out)


def cand_line(tag: str, p: Promo) -> str:
    segs = " + ".join(f"E{s.episode:02d} {s.first_event}..{s.last_event} "
                      f"({fmt_time(s.start)}-{fmt_time(s.end)}, {s.duration:.0f}s)" for s in p.segments)
    return f"{tag}: {segs}"


EPISODE_TASK = """Episode {n} logline: {logline}

TRANSCRIPT (ID [start-end] SPEAKER (delivery): "line"):
{transcript}

CANDIDATE PROMOS found by the rules (start line .. end line):
{cands}

For EVERY candidate, rate 1-10: hook (first 3 seconds), cliffhanger (last moment), standalone
(clear to a new viewer). Write hook_text and cliffhanger_text (quote or describe the moment) and a
one-line "why". If a different start or end line would clearly be better, give start_event and/or
end_event (IDs from the transcript; the clip must stay 20-60 seconds and the hook must open it).
Optionally propose up to 2 NEW promos the rules missed, same fields, with start_event and end_event.

JSON format:
{{"candidates": [{{"id": "C1", "hook": 7, "cliffhanger": 8, "standalone": 6, "hook_text": "...",
  "cliffhanger_text": "...", "why": "...", "start_event": "E01-010", "end_event": "E01-040"}}],
 "new": [{{"start_event": "...", "end_event": "...", "hook": 8, "cliffhanger": 7, "standalone": 7,
  "hook_text": "...", "cliffhanger_text": "...", "why": "..."}}]}}"""

MONTAGE_TASK = """Each promo below joins 2 segments (possibly from different episodes) into one clip.
Rate each 1-10 for hook, cliffhanger and standalone (does the join make sense to a new viewer?),
and write hook_text, cliffhanger_text and a one-line why. Do not change the cuts.

{blocks}

JSON format: {{"candidates": [{{"id": "M1", "hook": 7, "cliffhanger": 7, "standalone": 5,
  "hook_text": "...", "cliffhanger_text": "...", "why": "..."}}]}}"""

PAIR_TASK = """Two candidate promos for the same show. Which one would make a viewer who has never
seen the show more likely to stop scrolling AND tap to watch? Consider hook in the first 3 seconds,
the cliffhanger at the cut, and whether it makes sense on its own.

PROMO A:
{a}

PROMO B:
{b}

JSON format: {{"winner": "A" or "B", "reason": "one short sentence"}}"""


# ------------------------------------------------------------------------ reviewer
def _rating(d: dict, k: str) -> float | None:
    try:
        v = float(d.get(k))
        return max(1.0, min(10.0, v))
    except (TypeError, ValueError):
        return None


class LLMReviewer:
    def __init__(self, provider: Provider, cfg: dict, log=None):
        self.p = provider
        self.cfg = cfg
        self.blend = float(cfg["llm"].get("blend", 0.5))
        self.log = log or (lambda msg: None)
        self.errors: list[str] = []
        self.system = SYSTEM

    def set_genre(self, cfg: dict, genre: dict) -> None:
        """Use the genre-adjusted config and tell the model what kind of show this is."""
        self.cfg = cfg
        hint = (genre or {}).get("hint", "")
        self.system = SYSTEM + ("\n" + hint if hint else "")

    def _apply(self, promo: Promo, d: dict, idx_by_ep: dict, allow_recut: bool) -> Promo:
        """Validate one LLM judgement and merge it into the promo."""
        notes = []
        if allow_recut and len(promo.segments) == 1 and (d.get("start_event") or d.get("end_event")):
            seg = promo.segments[0]
            idx = idx_by_ep[seg.episode]
            sb = idx.by_id.get(d.get("start_event") or seg.first_event)
            eb = idx.by_id.get(d.get("end_event") or seg.last_event)
            if sb is None or eb is None:
                notes.append("llm_invalid_id")
            elif sb.event.kind == "camera" or eb.event.kind == "camera":
                sb = sb if sb.event.kind != "camera" else idx.by_id[seg.first_event]
                eb = eb if eb.event.kind != "camera" else idx.by_id[seg.last_event]
            if sb is not None and eb is not None and (sb.id, eb.id) != (seg.first_event, seg.last_event):
                t0, t1 = idx.snap_start(sb), idx.snap_end(eb)
                p = self.cfg["promo"]
                if p["min_duration"] <= t1 - t0 <= p["max_duration"]:
                    info = score_window(idx, t0, t1)
                    new = build_promo(promo.id, idx_by_ep, [Segment(seg.episode, t0, t1, sb.id, eb.id)],
                                      info, self.cfg, "single")
                    new.llm = {"recut_from": [seg.first_event, seg.last_event]}
                    promo = new
                else:
                    notes.append("llm_recut_rejected_duration")
        h, c, s = _rating(d, "hook"), _rating(d, "cliffhanger"), _rating(d, "standalone")
        if None in (h, c, s):
            promo.flags = sorted(set(promo.flags) | {"llm_missing_scores"})
            return promo
        llm_score = 100 * (0.4 * h + 0.4 * c + 0.2 * s) / 10
        rules = promo.scores["total"]
        promo.scores = dict(promo.scores, rules=rules, llm=llm_score,
                            final=(1 - self.blend) * rules + self.blend * llm_score)
        promo.llm = dict(promo.llm, model=self.p.label, hook=h, cliffhanger=c, standalone=s,
                         hook_text=d.get("hook_text", ""), cliffhanger_text=d.get("cliffhanger_text", ""),
                         why=d.get("why", ""))
        if d.get("why"):
            promo.why = str(d["why"]).strip()
        if d.get("hook_text"):
            promo.hook = f'{d["hook_text"]}  [rules: {promo.hook}]'
        if d.get("cliffhanger_text"):
            promo.cliffhanger = f'{d["cliffhanger_text"]}  [rules: {promo.cliffhanger}]'
        promo.source = "rules+llm"
        promo.flags = sorted(set(promo.flags) | set(notes))
        return promo

    def review_episode(self, idx: EpisodeIndex, cands: list[Promo], idx_by_ep: dict) -> list[Promo]:
        if not cands:
            return cands
        tags = {f"C{i + 1}": p for i, p in enumerate(cands)}
        user = EPISODE_TASK.format(n=idx.ep.number, logline=idx.ep.logline,
                                   transcript=transcript(idx),
                                   cands="\n".join(cand_line(t, p) for t, p in tags.items()))
        try:
            data = self.p.complete_json(self.system, user)
        except Exception as e:
            self.errors.append(f"episode {idx.ep.number}: {e}")
            for p in cands:
                p.flags = sorted(set(p.flags) | {"llm_failed"})
            return cands
        out = []
        seen = set()
        for d in data.get("candidates", []) if isinstance(data, dict) else []:
            if not isinstance(d, dict):
                continue
            tag = str(d.get("id", ""))
            if tag in tags and tag not in seen:
                seen.add(tag)
                out.append(self._apply(tags[tag], d, idx_by_ep, allow_recut=True))
        for tag, p in tags.items():           # anything the model skipped keeps the rules score
            if tag not in seen:
                p.flags = sorted(set(p.flags) | {"llm_skipped"})
                out.append(p)
        new = data.get("new", []) if isinstance(data, dict) else []
        for k, d in enumerate([x for x in new if isinstance(x, dict)][:2] if isinstance(new, list) else []):
            sb, eb = idx.by_id.get(d.get("start_event")), idx.by_id.get(d.get("end_event"))
            if sb is None or eb is None or eb.event.end <= sb.event.start:
                self.errors.append(f"episode {idx.ep.number}: proposal with invalid IDs {d}")
                continue
            t0, t1 = idx.snap_start(sb), idx.snap_end(eb)
            p = self.cfg["promo"]
            if not (p["min_duration"] <= t1 - t0 <= p["max_duration"]):
                self.errors.append(f"episode {idx.ep.number}: proposal outside 20-60 s rejected")
                continue
            info = score_window(idx, t0, t1)
            pr = build_promo(f"EP{idx.ep.number:02d}-L{k + 1}", idx_by_ep,
                             [Segment(idx.ep.number, t0, t1, sb.id, eb.id)], info, self.cfg, "single")
            out.append(self._apply(pr, d, idx_by_ep, allow_recut=False))
        for p in out:
            p.scores.setdefault("final", p.scores["total"])
        out.sort(key=lambda p: -p.scores["final"])
        return out

    def review_montages(self, montages: list[Promo], idx_by_ep: dict) -> list[Promo]:
        tags = {f"M{i + 1}": p for i, p in enumerate(montages)}
        blocks = []
        for t, p in tags.items():
            parts = [f"{t}:"]
            for k, s in enumerate(p.segments, 1):
                parts.append(f"  segment {k} (episode {s.episode}, {fmt_time(s.start)}-{fmt_time(s.end)}):")
                parts.append("    " + transcript(idx_by_ep[s.episode], s.start, s.end, camera=False)
                             .replace("\n", "\n    "))
            blocks.append("\n".join(parts))
        try:
            data = self.p.complete_json(self.system, MONTAGE_TASK.format(blocks="\n\n".join(blocks)))
        except Exception as e:
            self.errors.append(f"montages: {e}")
            return montages
        for d in data.get("candidates", []):
            if not isinstance(d, dict):
                continue
            t = str(d.get("id", ""))
            if t in tags:
                tags[t] = self._apply(tags[t], d, idx_by_ep, allow_recut=False)
        for p in tags.values():
            p.scores.setdefault("final", p.scores["total"])
        return list(tags.values())

    def _describe(self, p: Promo, idx_by_ep: dict) -> str:
        parts = []
        for s in p.segments:
            parts.append(f"(episode {s.episode}, {fmt_time(s.start)}-{fmt_time(s.end)})\n"
                         + transcript(idx_by_ep[s.episode], s.start, s.end, camera=False))
        return "\n".join(parts)

    def pairwise_rerank(self, promos: list[Promo], idx_by_ep: dict, top_n: int) -> list[Promo]:
        """One bottom-up sliding pass over the top N (PRP-sliding), each pair asked in both orders."""
        head, tail = promos[:top_n], promos[top_n:]
        for i in range(len(head) - 2, -1, -1):
            a, b = head[i], head[i + 1]
            try:
                r1 = self.p.complete_json(self.system, PAIR_TASK.format(a=self._describe(a, idx_by_ep),
                                                                   b=self._describe(b, idx_by_ep)))
                r2 = self.p.complete_json(self.system, PAIR_TASK.format(a=self._describe(b, idx_by_ep),
                                                                   b=self._describe(a, idx_by_ep)))
            except Exception as e:
                self.errors.append(f"pairwise: {e}")
                break
            b_wins = str(r1.get("winner", "")).upper() == "B" and str(r2.get("winner", "")).upper() == "A"
            if b_wins:                       # consistent in both orders -> swap
                head[i], head[i + 1] = b, a
                b.llm = dict(b.llm, pairwise=f"beat {a.id}: {r1.get('reason', '')}")
        out = head + tail
        for k, p in enumerate(out, 1):
            p.rank = k
        return out


def make_reviewer(cfg: dict, provider: str | None = None, model: str | None = None,
                  api_key: str | None = None, base_url: str | None = None, log=None):
    name = provider or cfg["llm"].get("provider", "none")
    if not name or name == "none":
        return None
    prov = Provider(name, model or cfg["llm"].get("model", ""), api_key or "",
                    base_url or cfg["llm"].get("base_url", ""),
                    temperature=cfg["llm"].get("temperature", 0.2),
                    max_retries=cfg["llm"].get("max_retries", 2))
    return LLMReviewer(prov, cfg, log)
