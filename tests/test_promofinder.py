"""Tests: parser, hard promo rules, LLM validation, and generality on an unseen genre/format."""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from promofinder.config import load_config, terms              # noqa: E402
from promofinder.engine import find_promos                      # noqa: E402
from promofinder.llm import LLMReviewer, Provider               # noqa: E402
from promofinder.parser import fmt_time, parse_text, parse_time  # noqa: E402

CFG = load_config()

# A made-up script in a different genre (modern CEO revenge drama) and a looser format:
# lower-case headings, no "Actions and dialogue" line, a V.O. tag and an O.S. tag.
SYNTHETIC = """EPISODE NUMBER: Episode 7
EPISODE LOGLINE: A billionaire CEO humiliates his assistant at the gala, unaware she is the heiress who owns his company.

scene 1
scene heading: INT. GALA BALLROOM - NIGHT
scene start: 00:00.00
scene end: 01:05.00

**CAMERA:**
Start: (00:00.00) | End: (00:03.00)
Wide shot of a glittering ballroom.

**ACTION:**
Start: (00:00.50) | End: (00:02.50)
Ethan throws a glass of champagne in Mia's face as the guests gasp.

**ETHAN (Cruel, Mocking):**
Start: (00:02.60) | End: (00:05.00)
"You are just a worthless assistant."

**CAMERA:**
Start: (00:03.00) | End: (00:09.00)
Close-up on Mia, champagne dripping from her hair.

**MIA (Quiet, Defiant):**
Start: (00:05.50) | End: (00:08.50)
"You will regret that tonight."

**ETHAN (Scornful):**
Start: (00:09.00) | End: (00:12.00)
"Security! Get this beggar out of my party!"

**CAMERA:**
Start: (00:09.00) | End: (00:20.00)
Medium shot of guards approaching.

**MIA (V.O.) (Calm):**
Start: (00:13.00) | End: (00:17.00)
"He has no idea who signs his paychecks."

**ACTION:**
Start: (00:17.00) | End: (00:20.00)
The doors burst open and a lawyer strides in holding a folder.

**LAWYER (O.S.) (Booming):**
Start: (00:20.00) | End: (00:23.00)
"Nobody touches the heiress of Whitmore Group!"

**CAMERA:**
Start: (00:20.00) | End: (00:30.00)
Close-up of Ethan's face draining of colour.

**ETHAN (Shocked, Breathless):**
Start: (00:24.00) | End: (00:26.00)
"Heiress?"

**MIA (Cold, Commanding):**
Start: (00:27.00) | End: (00:31.00)
"I own this company, Ethan. And you are fired."

**CAMERA:**
Start: (00:30.00) | End: (00:40.00)
Push-in on Ethan as he falls to his knees.

**ETHAN (Desperate, Pleading):**
Start: (00:33.00) | End: (00:36.00)
"Please, I didn't know who you were!"

**MIA (Venomous):**
Start: (00:37.00) | End: (00:41.00)
"Then let me show you what a worthless assistant can do..."

**SOUND FX:**
Start: (00:41.00) | End: (00:43.00)
Ominous sting into a cut to black.
"""


def synthetic():
    return parse_text(SYNTHETIC, "synthetic.txt")


# ---------------------------------------------------------------- parser
def test_time_round_trip():
    assert parse_time("01:23.45") == pytest.approx(83.45)
    assert parse_time("1:02:03.5") == pytest.approx(3723.5)
    assert fmt_time(83.45) == "01:23.45"


def test_parser_reads_loose_format():
    ep = synthetic()
    assert ep.number == 7
    kinds = {e.kind for e in ep.events}
    assert kinds == {"camera", "action", "dialogue", "sfx"}
    vo = [e for e in ep.dialogue if e.voice_over]
    os_ = [e for e in ep.dialogue if e.off_screen]
    assert vo and vo[0].speaker == "Mia" and vo[0].cue == "Calm"
    assert os_ and os_[0].speaker == "Lawyer" and os_[0].cue == "Booming"
    assert all(e.end >= e.start for e in ep.events)


def test_whole_word_matching():
    r = terms(["hit", "heart", "humiliat*"])
    assert r.search("he hit her")
    assert not r.search("her breath hitching")      # 'hit' must not match 'hitching'
    assert not r.search("by the hearth")             # 'heart' must not match 'hearth'
    assert r.search("a humiliating defeat")          # '*' = prefix


# ------------------------------------------------------- generality + rules
def test_finds_promo_in_unseen_genre():
    res = find_promos([synthetic()], CFG)
    assert res["promos"], "should find at least one promo in a new genre"
    best = res["promos"][0]
    assert 20 <= best.duration <= 60
    assert best.scores["hook"] >= 0.5


def _check_hard_rules(res, idx_by_ep):
    for p in res["promos"]:
        assert CFG["promo"]["min_duration"] - 0.01 <= p.duration <= CFG["promo"]["max_duration"] + 0.01, p.id
        for s in p.segments:
            assert s.start < s.end
            ep = idx_by_ep[s.episode]
            assert s.end <= ep.duration + 0.01
        if p.kind == "single":
            s = p.segments[0]
            ev = idx_by_ep[s.episode].ep.get(p.hook_event)
            assert ev is not None and ev.start - s.start <= CFG["promo"]["hook_window"] + 0.01, p.id
            assert "cuts_through_dialogue" not in p.flags, p.id


@pytest.mark.skipif(not (ROOT / "data" / "episodes").exists(), reason="input pack not present")
def test_hard_rules_on_input_pack():
    from promofinder.parser import load_episodes
    files = sorted((ROOT / "data" / "episodes").glob("*.txt"))
    res = find_promos(load_episodes(files), CFG)
    _check_hard_rules(res, res["indexes"])
    assert len(res["post_first"]) == CFG["promo"]["post_first"]


def test_hard_rules_on_synthetic():
    res = find_promos([synthetic()], CFG)
    _check_hard_rules(res, res["indexes"])


# ---------------------------------------------------- LLM output validation
class FakeProvider(Provider):
    """Returns a scripted reply so we can test validation without a network."""
    def __init__(self, reply):
        super().__init__("mock")
        self.reply = reply

    def complete_json(self, system, user):
        return self.reply


def test_llm_invented_ids_are_ignored():
    ep = synthetic()
    reply = {"candidates": [{"id": "C1", "hook": 9, "cliffhanger": 9, "standalone": 9,
                             "why": "great", "start_event": "E07-999", "end_event": "E99-000"}],
             "new": [{"start_event": "E07-500", "end_event": "E07-501", "hook": 10,
                      "cliffhanger": 10, "standalone": 10}]}
    rev = LLMReviewer(FakeProvider(reply), CFG)
    res = find_promos([ep], CFG, llm=rev)
    p = res["promos"][0]
    assert "llm_invalid_id" in p.flags                 # bad re-cut ignored
    assert all(not x.id.endswith("L1") for x in res["promos"])   # bad proposal dropped
    assert rev.errors                                   # and reported
    _check_hard_rules(res, res["indexes"])


def test_llm_scores_are_blended_and_clamped():
    ep = synthetic()
    reply = {"candidates": [{"id": "C1", "hook": 50, "cliffhanger": -3, "standalone": 7}], "new": []}
    res = find_promos([ep], CFG, llm=LLMReviewer(FakeProvider(reply), CFG))
    p = [x for x in res["promos"] if x.source == "rules+llm"][0]
    assert p.llm["hook"] == 10 and p.llm["cliffhanger"] == 1     # clamped to 1..10
    blend = CFG["llm"]["blend"]
    assert p.scores["final"] == pytest.approx((1 - blend) * p.scores["rules"] + blend * p.scores["llm"])


def test_llm_failure_falls_back_to_rules():
    class Broken(FakeProvider):
        def complete_json(self, system, user):
            raise RuntimeError("quota exceeded")
    res = find_promos([synthetic()], CFG, llm=LLMReviewer(Broken({}), CFG))
    assert res["promos"] and "llm_failed" in res["promos"][0].flags


def test_outputs_are_written(tmp_path):
    from promofinder.report import write_all
    res = find_promos([synthetic()], CFG)
    write_all(res, CFG, {"show": "synthetic", "episodes": 1, "mode": "rules only"}, tmp_path)
    data = json.loads((tmp_path / "promos.json").read_text())
    assert data["promos"] and data["post_first"]
    for name in ("promos.csv", "report.md", "report.html"):
        assert (tmp_path / name).stat().st_size > 0


# ---------------------------------------------------- episode numbering
def test_missing_or_duplicate_episode_numbers_get_unique_ids():
    from promofinder.parser import ensure_unique_numbers, episode_of
    a = parse_text(SYNTHETIC.replace("EPISODE NUMBER: Episode 7", "EPISODE NUMBER: Pilot"), "a.txt")
    b = parse_text(SYNTHETIC.replace("EPISODE NUMBER: Episode 7", "EPISODE NUMBER: Finale"), "b.txt")
    c = parse_text(SYNTHETIC, "c.txt")                       # episode 7
    d = parse_text(SYNTHETIC, "d.txt")                       # duplicate 7
    eps = ensure_unique_numbers([a, b, c, d])
    nums = [e.number for e in eps]
    assert len(set(nums)) == 4 and 0 not in nums
    for e in eps:
        assert all(ev.episode == e.number and episode_of(ev.id) == e.number for ev in e.events)
    res = find_promos(eps, CFG)
    assert sorted(res["indexes"]) == sorted(nums)            # nothing silently overwritten


def test_three_digit_episode_ids():
    from promofinder.parser import episode_of
    ep = parse_text(SYNTHETIC.replace("Episode 7", "Episode 104"), "x.txt")
    assert ep.number == 104 and episode_of(ep.events[0].id) == 104


# ------------------------------------------- multi-segment promos obey the rules too
@pytest.mark.skipif(not (ROOT / "data" / "episodes").exists(), reason="input pack not present")
def test_multi_segment_rules_on_input_pack():
    from promofinder.parser import load_episodes
    res = find_promos(load_episodes(sorted((ROOT / "data" / "episodes").glob("*.txt"))), CFG)
    multis = [p for p in res["promos"] if p.kind == "multi"]
    assert multis
    for p in multis:
        first = p.segments[0]
        ev = res["indexes"][first.episode].ep.get(p.hook_event)
        assert ev is not None and ev.start - first.start <= CFG["promo"]["hook_window"] + 0.01
        assert "cuts_through_dialogue" not in p.flags
        assert "multi_segment_editor_check" in p.flags


# ------------------------------------------------- provider robustness (no network)
def FakeResp(code, payload=None, text=""):
    from promofinder.llm import HTTPResponse
    return HTTPResponse(code, text if payload is None else json.dumps(payload), {})


@pytest.fixture
def fake_http(monkeypatch, tmp_path):
    """Replace the one network function (llm.http_post) with a scripted responder."""
    import promofinder.llm as llm
    monkeypatch.setattr(llm, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(llm.time, "sleep", lambda s: None)
    calls = []

    def install(responder):
        def post(url, headers, body, params, timeout):
            kw = {"json": body, "headers": headers}
            calls.append((url, kw))
            return responder(url, kw, len(calls))
        monkeypatch.setattr(llm, "http_post", post)
        return calls
    return install


def test_unknown_model_falls_back_to_next(fake_http):
    calls = fake_http(lambda url, kw, n: FakeResp(404, text="model not found") if "3.8" in url
                      else FakeResp(200, {"candidates": [{"content": {"parts": [{"text": '{"ok": 1}'}]}}]}))
    p = Provider("gemini", api_key="k")
    assert p.complete_json("s", "u") == {"ok": 1}
    assert p.model == "gemini-3.5-flash" and p.notes
    assert "key=" not in calls[0][0]                         # key goes in a header, not the URL
    assert calls[0][1]["headers"].get("x-goog-api-key") == "k"


def test_temperature_rejected_is_retried_without_it(fake_http):
    def resp(url, kw, n):
        if "temperature" in kw["json"]:
            return FakeResp(400, text="Unsupported parameter: 'temperature'")
        return FakeResp(200, {"choices": [{"message": {"content": '{"ok": 2}'}}]})
    fake_http(resp)
    p = Provider("openai", api_key="k")
    assert p.complete_json("s", "u") == {"ok": 2} and p.send_temperature is False


def test_gemini_thought_parts_are_ignored(fake_http):
    fake_http(lambda url, kw, n: FakeResp(200, {"candidates": [{"content": {"parts": [
        {"text": "thinking about it...", "thought": True}, {"text": '{"ok": 3}'}]}}]}))
    assert Provider("gemini", api_key="k").complete_json("s", "u") == {"ok": 3}


def test_bad_key_is_not_retried(fake_http):
    calls = fake_http(lambda url, kw, n: FakeResp(401, text="invalid api key"))
    from promofinder.llm import LLMError
    with pytest.raises(LLMError):
        Provider("groq", api_key="bad").complete_json("s", "u")
    assert len(calls) == 1


def test_non_object_reply_is_rejected(fake_http):
    fake_http(lambda url, kw, n: FakeResp(200, {"choices": [{"message": {"content": "[1, 2, 3]"}}]}))
    from promofinder.llm import LLMError
    with pytest.raises(LLMError):
        Provider("openai", api_key="k", max_retries=1).complete_json("s", "u")


def test_malformed_items_in_reply_do_not_crash():
    reply = {"candidates": ["oops", {"id": "C1", "hook": 7, "cliffhanger": 7, "standalone": 7}],
             "new": ["bad", 3]}
    res = find_promos([synthetic()], CFG, llm=LLMReviewer(FakeProvider(reply), CFG))
    assert res["promos"]


def test_every_word_list_entry_is_a_string():
    """YAML 1.1 turns bare yes/no/on/off into booleans; a word list must only hold words."""
    bad = []

    def walk(node, path):
        if isinstance(node, dict):
            for k, v in node.items():
                walk(v, f"{path}.{k}")
        elif isinstance(node, list) and any(isinstance(x, str) for x in node):
            bad.extend(f"{path}: {x!r}" for x in node if not isinstance(x, str))
    walk(CFG, "")
    assert not bad, bad


def test_website_entry_matches_engine():
    """The website calls promofinder.web.run; it must give the same promos as find_promos."""
    from promofinder.web import run
    data = json.loads(run(json.dumps([{"name": "s.txt", "text": SYNTHETIC}]), json.dumps(CFG)))
    direct = find_promos([synthetic()], CFG)
    assert [p["id"] for p in data["promos"]] == [p.id for p in direct["promos"]]
    assert data["csv"].startswith("rank,id,") and data["episodes"][0]["curve"]


# ---------------------------------------------------- genre mode
from promofinder.config import deep_merge                       # noqa: E402
from promofinder.genre import detect_genre, resolve_genre       # noqa: E402
from promofinder.parser import load_episodes                    # noqa: E402

SAMPLES = ROOT / "samples"


def _show(name):
    return load_episodes(sorted((SAMPLES / name).glob("*.txt")))


def test_list_merge_adds_and_removes():
    base = {"a": {"words": ["x", "y"]}, "n": 1}
    out = deep_merge(base, {"a": {"words+": ["z", "x"], "w": 2}, "n": 3})
    assert out == {"a": {"words": ["x", "y", "z"], "w": 2}, "n": 3}
    assert deep_merge(base, {"a": {"words-": ["x"]}})["a"]["words"] == ["y"]
    assert base["a"]["words"] == ["x", "y"]   # the base config is never modified


def test_default_genre_changes_nothing():
    """A drama show must be scored exactly like the tuned base config."""
    assert CFG["genre"]["profiles"][CFG["genre"]["default"]]["overrides"] == {}
    cfg, info = resolve_genre(_show("demo_show"), CFG)
    assert info["genre"] == "drama" and cfg == CFG


@pytest.mark.parametrize("show,genre", [("demo_show", "drama"), ("demo_thriller", "thriller"),
                                        ("demo_comedy", "comedy")])
def test_genre_is_detected(show, genre):
    assert detect_genre(_show(show), CFG)[0] == genre


def test_genre_can_be_chosen_and_unknown_is_rejected():
    eps = _show("demo_comedy")
    forced = load_config(None, {"genre": {"mode": "thriller"}})
    assert resolve_genre(eps, forced)[1]["genre"] == "thriller"
    with pytest.raises(ValueError):
        resolve_genre(eps, load_config(None, {"genre": {"mode": "western"}}))


def test_genre_mode_changes_scoring_and_moods():
    eps = _show("demo_comedy")
    comedy = find_promos(eps, CFG)
    as_drama = find_promos(eps, load_config(None, {"genre": {"mode": "drama"}}))
    assert comedy["genre"]["genre"] == "comedy"
    assert "Comedy" in comedy["promos"][0].moods
    assert not any("Comedy" in p.moods for p in as_drama["promos"])
    for res in (comedy, as_drama, find_promos(_show("demo_thriller"), CFG)):
        _check_hard_rules(res, res["indexes"])


@pytest.mark.skipif(not (ROOT / "data" / "episodes").exists(), reason="input pack not present")
def test_input_pack_is_detected_as_drama():
    eps = load_episodes(sorted((ROOT / "data" / "episodes").glob("*.txt")))
    assert detect_genre(eps, CFG)[0] == "drama"


def test_website_passes_genre_choice():
    from promofinder.web import run
    files = [{"name": f.name, "text": f.read_text()} for f in sorted((SAMPLES / "demo_comedy").glob("*.txt"))]
    data = json.loads(run(json.dumps(files), json.dumps(CFG), json.dumps({"overrides": {"genre": {"mode": "horror"}}})))
    assert data["meta"]["genre"] == "Horror" and data["genre"]["detected"] is False
    data = json.loads(run(json.dumps(files), json.dumps(CFG)))
    assert data["meta"]["genre"] == "Comedy" and data["genre"]["detected"] is True


# ---------------------------------------------------- learning from posted promos
from promofinder import learn                                    # noqa: E402

DEMO_LABELS = """episode,start,end,label
1,00:23.4,00:53.0,1
2,15.2,40.0,1
2,0.0,14.0,0
3,00:14.0,00:45.0,1
"""


def test_labels_csv_is_read():
    rows = learn.parse_labels_csv(DEMO_LABELS)
    assert rows[0] == {"episode": 1, "start": 23.4, "end": 53.0, "label": 1}
    assert rows[2]["label"] == 0 and len(rows) == 4
    with pytest.raises(ValueError):
        learn.parse_labels_csv("ep,from,to\n1,2,3")
    with pytest.raises(ValueError):
        learn.parse_labels_csv("episode,start,end\n1,30,10")


@pytest.mark.parametrize("kind", ["forest", "boosted", "logistic"])
def test_models_learn_a_simple_rule(kind):
    """Label = first feature above 0.5; every model type should rank positives above negatives."""
    import random
    rng = random.Random(1)
    X = [[rng.random(), rng.random(), rng.random()] for _ in range(300)]
    y = [1 if x[0] > 0.5 else 0 for x in X]
    model = learn.train(X, y, kind)
    scores = [learn.predict(model, x) for x in X]
    assert all(0.0 <= s <= 1.0 for s in scores)
    assert learn.auc(scores, y) > 0.9


def test_train_and_test_end_to_end_and_model_file():
    eps = _show("demo_show")
    labels = learn.parse_labels_csv(DEMO_LABELS)
    out = learn.train_and_test(eps, labels, CFG, "forest", test_from=3)
    rep = out["report"]
    assert rep["train_episodes"] == [1, 2] and rep["test_episodes"] == [3]
    assert set(rep["formula"]) == set(rep["ml"]) == {"found", "of", "rate", "top1"}
    model = learn.from_json(learn.to_json(out["model"]))        # survives a save and load
    res = find_promos(eps, CFG, ml=learn.MLScorer(model))
    _check_hard_rules(res, res["indexes"])                      # the model never breaks the rules
    assert all("ml" in p.scores and "formula" in p.scores for p in res["promos"] if p.kind == "single")


def test_labels_for_missing_episodes_are_rejected():
    with pytest.raises(ValueError):
        learn.train_and_test(_show("demo_show"), learn.parse_labels_csv("episode,start,end\n9,1,30\n4,1,30"), CFG)


def test_website_trains_and_uses_a_model():
    from promofinder.web import run, train
    files = [{"name": f.name, "text": f.read_text()} for f in sorted((SAMPLES / "demo_show").glob("*.txt"))]
    out = json.loads(train(json.dumps(files), json.dumps(CFG), DEMO_LABELS, json.dumps({"model": "boosted"})))
    assert "error" not in out and out["report"]["model"] == "boosted"
    data = json.loads(run(json.dumps(files), json.dumps(CFG), json.dumps({"ml_model": out["model_json"]})))
    assert "learned model" in data["meta"]["mode"] and data["promos"]
    bad = json.loads(train(json.dumps(files), json.dumps(CFG), "nonsense"))
    assert "error" in bad
