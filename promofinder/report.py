"""Write results as JSON, CSV, Markdown and a self-contained HTML report."""
from __future__ import annotations

import csv
import html
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from .parser import fmt_time

FLAG_HELP = {
    "weak_hook": "nothing strong in the first 3 s",
    "weak_cliffhanger": "the cut does not leave much unresolved",
    "resolves_before_cut": "the clip ends on a calm or resolved beat",
    "dead_air": "a long stretch with no dialogue or action",
    "starts_mid_sentence": "opens on the second half of a sentence",
    "mostly_voice_over": "mostly inner voice-over",
    "no_dialogue": "no dialogue in the clip",
    "hook_after_3s": "hook lands after the first 3 s",
    "cuts_through_dialogue": "a cut lands inside a spoken line",
    "duration_out_of_range": "outside 20-60 s",
    "multi_segment_editor_check": "joined clips: an editor should check the join",
    "check_ad_policy": "possibly restricted content for ads",
    "llm_failed": "AI review failed; rules result kept",
    "llm_skipped": "AI did not rate this one",
    "llm_invalid_id": "AI referred to a line that does not exist; ignored",
    "llm_recut_rejected_duration": "AI's re-cut broke the 20-60 s rule; ignored",
    "llm_missing_scores": "AI reply missing scores",
}


def seg_text(p) -> str:
    return " + ".join(f"Ep{s.episode} {fmt_time(s.start)}-{fmt_time(s.end)}" for s in p.segments)


def qc_summary(result: dict, cfg: dict) -> dict:
    promos = result["promos"]
    flags = Counter(f for p in promos for f in p.flags)
    by_ep = {}
    for p in promos:
        if p.kind == "single":
            e = p.episodes[0]
            by_ep[e] = max(by_ep.get(e, 0), p.scores.get("final", p.total))
    weak_eps = sorted(e for e, s in by_ep.items() if s < cfg["flags"]["weak_episode_score"])
    top = result["post_first"]
    return {
        "promos": len(promos),
        "single": sum(p.kind == "single" for p in promos),
        "multi": sum(p.kind == "multi" for p in promos),
        "flag_counts": dict(flags.most_common()),
        "flag_help": {k: FLAG_HELP.get(k, "") for k in flags},
        "rule_violations_in_post_first": sum(
            1 for p in top for f in p.flags if f in ("hook_after_3s", "cuts_through_dialogue", "duration_out_of_range")),
        "episodes_with_no_strong_promo": weak_eps,
        "duration_range": [min(p.duration for p in promos), max(p.duration for p in promos)] if promos else [],
    }


def genre_line(result: dict) -> str:
    g = result.get("genre") or {}
    if not g.get("genre"):
        return ""
    return f"{g['label']} ({'detected' if g.get('detected') else 'chosen'})"


def to_payload(result: dict, cfg: dict, meta: dict) -> dict:
    g = result.get("genre") or {}
    return {
        "meta": dict(meta, generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds")),
        "genre": {k: g[k] for k in ("genre", "label", "mode", "detected", "rates") if k in g},
        "post_first": [{"rank": i + 1, "id": p.id, "segments": seg_text(p), "reason": p.rank_reason,
                        "hook": p.hook, "cliffhanger": p.cliffhanger, "why": p.why, "moods": p.moods}
                       for i, p in enumerate(result["post_first"])],
        "qc": qc_summary(result, cfg),
        "promos": [p.to_dict() for p in result["promos"]],
    }


def write_csv(promos, path: Path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["rank", "id", "kind", "episodes", "segments", "duration_s", "hook", "cliffhanger",
                    "why", "score", "hook_score", "cliffhanger_score", "energy", "clarity", "llm_score",
                    "flags", "moods"])
        for p in promos:
            s = p.scores
            w.writerow([p.rank, p.id, p.kind, " ".join(map(str, p.episodes)), seg_text(p), p.duration,
                        p.hook, p.cliffhanger, p.why, round(s.get("final", s["total"]), 1),
                        round(s["hook"], 2), round(s["cliffhanger"], 2), round(s["energy"], 2),
                        round(s["clarity"], 2), round(s["llm"], 1) if "llm" in s else "",
                        " ".join(p.flags), " ".join(p.moods)])


def excerpt(p, indexes, max_lines=40) -> list[str]:
    lines = []
    for s in p.segments:
        idx = indexes[s.episode]
        if len(p.segments) > 1:
            lines.append(f"--- Episode {s.episode}, {fmt_time(s.start)}-{fmt_time(s.end)} ---")
        for b in idx.beats:
            e = b.event
            if e.kind == "camera" or e.start < s.start - 0.05 or e.start >= s.end:
                continue
            mark = " <== HOOK" if e.id == p.hook_event else (" <== CUT" if e.id == p.cliff_event else "")
            if e.kind == "dialogue":
                lines.append(f'{fmt_time(e.start)} {e.speaker}: "{e.text}"{mark}')
            else:
                lines.append(f"{fmt_time(e.start)} [{e.kind}] {e.text}{mark}")
    return lines[:max_lines]


def write_markdown(result: dict, cfg: dict, meta: dict, path: Path):
    qc = qc_summary(result, cfg)
    idx = result["indexes"]
    out = [f"# Promo Finder results: {meta.get('show', 'show')}", "",
           f"Episodes: {meta.get('episodes')} | Promos: {qc['promos']} ({qc['single']} single, "
           f"{qc['multi']} multi-segment) | Mode: {meta.get('mode')}"
           + (f" | Genre: {genre_line(result)}" if genre_line(result) else "") + " | Generated by the tool, unedited.", "",
           "## Post these first", ""]
    for i, p in enumerate(result["post_first"], 1):
        out += [f"### {i}. {p.id}: {seg_text(p)} ({p.duration:.0f} s)",
                f"- **Hook:** {p.hook}", f"- **Cliffhanger:** {p.cliffhanger}",
                f"- **Why it works:** {p.why}", f"- **Mood:** {', '.join(p.moods) or 'none'}",
                f"- **Why this rank:** {p.rank_reason}",
                f"- **Flags:** {', '.join(p.flags) or 'none'}", ""]
    out += ["## Quality check", "",
            f"- Rule violations in the post-first list: {qc['rule_violations_in_post_first']}",
            f"- Episodes with no strong promo (best score < {cfg['flags']['weak_episode_score']}): "
            f"{qc['episodes_with_no_strong_promo'] or 'none'}",
            "- Flag counts across all promos:"]
    for k, v in qc["flag_counts"].items():
        out.append(f"  - `{k}` x{v}: {FLAG_HELP.get(k, '')}")
    out += ["", "## All promos, ranked", ""]
    for p in result["promos"]:
        s = p.scores
        out += [f"### #{p.rank} {p.id} ({p.kind}): {seg_text(p)} ({p.duration:.0f} s), "
                f"score {s.get('final', s['total']):.1f}",
                f"- Hook: {p.hook}", f"- Cliffhanger: {p.cliffhanger}", f"- Why: {p.why}",
                f"- Mood: {', '.join(p.moods) or 'none'}",
                f"- Scores: hook {s['hook']:.2f}, cliffhanger {s['cliffhanger']:.2f}, energy {s['energy']:.2f}, "
                f"clarity {s['clarity']:.2f}" + (f", AI {s['llm']:.0f}" if "llm" in s else ""),
                f"- Flags: {', '.join(p.flags) or 'none'}", "", "```"] + excerpt(p, idx) + ["```", ""]
    path.write_text("\n".join(out), encoding="utf-8")


def write_html(result: dict, cfg: dict, meta: dict, path: Path):
    qc = qc_summary(result, cfg)
    idx = result["indexes"]
    top_ids = {p.id for p in result["post_first"]}
    esc = html.escape

    def card(p, n=None):
        s = p.scores
        score = s.get("final", s["total"])
        bars = "".join(
            f'<div class="bar"><span>{k}</span><i style="width:{v * 100:.0f}%"></i><b>{v:.2f}</b></div>'
            for k, v in (("hook", s["hook"]), ("cliffhanger", s["cliffhanger"]),
                         ("energy", s["energy"]), ("clarity", s["clarity"])))
        flags = "".join(f'<span class="flag" title="{esc(FLAG_HELP.get(f, ""))}">{esc(f)}</span>' for f in p.flags)
        ex = esc("\n".join(excerpt(p, idx)))
        star = '<span class="star">post first</span>' if p.id in top_ids else ""
        return f"""<article class="card" data-ep="{' '.join(map(str, p.episodes))}" data-kind="{p.kind}">
<header><span class="rank">#{n or p.rank}</span><h3>{esc(seg_text(p))}</h3>{star}
<span class="score">{score:.0f}</span></header>
<p class="meta">{esc(p.id)} · {p.kind} · {p.duration:.0f} s{(' · mood: ' + esc(', '.join(p.moods))) if p.moods else ''}</p>
<p><b>Hook</b> {esc(p.hook)}</p><p><b>Cliffhanger</b> {esc(p.cliffhanger)}</p>
<p><b>Why</b> {esc(p.why)}</p><p class="reason">{esc(p.rank_reason)}</p>
<div class="bars">{bars}</div><div class="flags">{flags}</div>
<details><summary>Script excerpt</summary><pre>{ex}</pre></details></article>"""

    first = "".join(card(p, i) for i, p in enumerate(result["post_first"], 1))
    allp = "".join(card(p) for p in result["promos"])
    flag_rows = "".join(f"<tr><td><code>{esc(k)}</code></td><td>{v}</td><td>{esc(FLAG_HELP.get(k, ''))}</td></tr>"
                        for k, v in qc["flag_counts"].items())
    doc = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Promo Finder Results</title>
<style>
:root{{--bg:#fbfaf7;--fg:#1d1b18;--muted:#6b665d;--card:#fff;--line:#e6e1d8;--acc:#b4441f;--bar:#e9dccf}}
@media (prefers-color-scheme:dark){{:root:not([data-theme=light]){{--bg:#16140f;--fg:#efe9df;--muted:#a59d90;--card:#1f1c16;--line:#3a352c;--acc:#f08a5d;--bar:#3d3227}}}}
:root[data-theme=dark]{{--bg:#16140f;--fg:#efe9df;--muted:#a59d90;--card:#1f1c16;--line:#3a352c;--acc:#f08a5d;--bar:#3d3227}}
body{{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,-apple-system,Segoe UI,sans-serif}}
main{{max-width:980px;margin:0 auto;padding:24px 16px 64px}} h1{{font-size:26px;margin:0 0 4px}}
h2{{font-size:19px;margin:32px 0 12px;border-bottom:1px solid var(--line);padding-bottom:6px}}
.sub{{color:var(--muted)}} .grid{{display:grid;gap:12px}}
.card{{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px 16px}}
.card header{{display:flex;align-items:center;gap:10px;flex-wrap:wrap}} .card h3{{font-size:15px;margin:0;flex:1}}
.rank{{font-weight:700;color:var(--acc)}} .score{{font-weight:700;font-size:18px}}
.star{{font-size:12px;background:var(--acc);color:#fff;border-radius:99px;padding:1px 8px}}
.meta,.reason{{color:var(--muted);font-size:13px;margin:4px 0}} .card p{{margin:6px 0}}
.bars{{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:4px 16px;margin:8px 0}}
.bar{{display:flex;align-items:center;gap:6px;font-size:12px}} .bar span{{width:72px;color:var(--muted)}}
.bar i{{display:block;height:6px;background:var(--acc);border-radius:3px;flex:none;max-width:90px}}
.flag{{display:inline-block;font-size:12px;border:1px solid var(--line);border-radius:6px;padding:0 6px;margin:2px 4px 0 0;color:var(--muted)}}
pre{{white-space:pre-wrap;font-size:12.5px;background:var(--bg);padding:10px;border-radius:8px;overflow-x:auto}}
table{{border-collapse:collapse;width:100%;font-size:14px}} td{{border-top:1px solid var(--line);padding:6px 8px;vertical-align:top}}
.filters{{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:10px}} input,select{{font:inherit;padding:6px 8px;border:1px solid var(--line);border-radius:6px;background:var(--card);color:var(--fg)}}
</style></head><body><main>
<h1>Promo Finder results</h1>
<p class="sub">{esc(str(meta.get('show', '')))} · {meta.get('episodes')} episodes · {qc['promos']} promos
({qc['single']} single, {qc['multi']} multi-segment) · mode: {esc(str(meta.get('mode')))}{(' · genre: ' + esc(genre_line(result))) if genre_line(result) else ''} · unedited tool output</p>
<h2>Post these first</h2><div class="grid">{first}</div>
<h2>Quality check</h2>
<p>Rule violations in the post-first list: <b>{qc['rule_violations_in_post_first']}</b>.
Episodes with no strong promo: <b>{esc(', '.join(map(str, qc['episodes_with_no_strong_promo'])) or 'none')}</b>.</p>
<table>{flag_rows}</table>
<h2>All promos</h2>
<div class="filters"><input id="ep" placeholder="Episode number" inputmode="numeric">
<select id="kind"><option value="">All kinds</option><option value="single">Single clip</option>
<option value="multi">Multi-segment</option></select></div>
<div class="grid" id="all">{allp}</div>
</main><script>
const ep=document.getElementById('ep'),kind=document.getElementById('kind');
function f(){{document.querySelectorAll('#all .card').forEach(c=>{{
const okE=!ep.value||c.dataset.ep.split(' ').includes(ep.value.trim());
const okK=!kind.value||c.dataset.kind===kind.value;c.style.display=okE&&okK?'':'none';}})}}
ep.oninput=f;kind.onchange=f;
</script></body></html>"""
    path.write_text(doc, encoding="utf-8")


def write_all(result: dict, cfg: dict, meta: dict, out_dir: str | Path) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    payload = to_payload(result, cfg, meta)
    (out / "promos.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    write_csv(result["promos"], out / "promos.csv")
    write_markdown(result, cfg, meta, out / "report.md")
    write_html(result, cfg, meta, out / "report.html")
    return payload
