# Promo Finder: Kuku AI Generalist Assignment, Problem 1

**Live website: https://soham0219.github.io/kuku-promo-finder/**

Reads microdrama screenplays and finds 20–60 s promo clips. Each clip opens on a **hook** (first 3 s), cuts on a **cliffhanger**, and **makes sense to a new viewer**. The output is ranked, and every promo comes with its quality flags.

- **General:** nothing in the code is specific to one show. All word lists, weights and rules live in [`config.yaml`](config.yaml).
- **Free by default:** rules-only mode needs no API key and processes 50 episodes in about 2 seconds.
- **One engine, two ways to run it:** the website runs this same Python package inside the visitor's browser (Pyodide), so its results are identical to the command line's. There is no server.
- **Genre-aware:** it detects whether a show is a drama, thriller, horror, romance or comedy and changes what counts as a hook and a good cut for that genre (`genre:` in `config.yaml`, or `--genre`). Each promo also gets mood tags (Betrayal, Danger, Comedy…) for ad targeting. A drama is scored exactly like the tuned base config, so the input-pack results don't change.
- **Learns from posted promos:** give it a CSV of promos your team already posted (`episode,start,end,label`) and it trains a small model (random forest, boosted trees like XGBoost, or logistic regression, all in plain Python) on every candidate clip, tests it on episodes it never saw, and compares it with the formula. Works on the website and with `train.py`.
- **Model-swappable:** add an AI reviewer with one flag (`--provider gemini|groq|openai|anthropic|openai_compatible|ollama`). The AI can only refer to line IDs. Code turns IDs into timestamps and re-checks every rule, so the model can't invent a timestamp. If a default model name has been retired, the next one in `MODEL_CHAIN` (`promofinder/llm.py`) is tried, and models that reject a `temperature` setting are retried without it. Any episode the AI can't review keeps its rules result and is flagged.

## Run it

```bash
pip install -r requirements.txt

# rules only (free)
python run.py --episodes path/to/scripts --out outputs/rules

# with an AI reviewer (free Gemini key from aistudio.google.com)
export GEMINI_API_KEY=...
python run.py --episodes path/to/scripts --provider gemini --out outputs/gemini

# force a genre instead of auto-detecting it
python run.py --episodes path/to/scripts --genre comedy --out outputs/comedy

# learn from posted promos, then use the model on a new show
python train.py --episodes path/to/scripts --labels posted.csv --out model.json
python run.py --episodes path/to/new_show --ml-model model.json --out outputs/new_show

# any OpenAI-compatible endpoint (e.g. a model Kuku provides)
python run.py --provider openai_compatible --base-url https://HOST/v1 --model MODEL --api-key KEY
```

Each run writes `promos.json`, `promos.csv`, `report.md` and `report.html` to the output folder.

## The website

`index.html` plus `site/` (page, styles, a worker that runs the engine) and `vendor/` (Pyodide, js-yaml and fonts, all served from this repo, so nothing loads from other servers). The page fetches `promofinder/*.py` and `config.yaml` from the repo itself.

- **Run it locally:** `python -m http.server 8000` in the repo folder, then open http://localhost:8000.
- **Deploy free on GitHub Pages:** repo **Settings → Pages → Build and deployment → Source: Deploy from a branch → Branch: `main`, folder `/ (root)` → Save**. The site appears at `https://<username>.github.io/<repo>/` within a minute or two. The `.nojekyll` file makes Pages serve `promofinder/__init__.py` as-is.
- The site bundles only three made-up 3-episode demo shows (`samples/`: a drama, a thriller and a comedy). Visitors upload their own screenplays; files never leave their browser.
- The optional AI review also works from the browser for providers that allow browser requests; the command line is the reference path.

## How it works

See [`docs/how_it_works.md`](docs/how_it_works.md). In short:

1. Parse the scripts into timestamped beats.
2. Score each beat for surprise (hook) and suspense (cliffhanger) with explainable text signals.
3. Try every 20–60 s clip whose cuts snap to camera shots and never land inside a line of dialogue.
4. Score each clip: `Score = Σ wₖ·componentₖ`, over hook, cliffhanger, energy, clarity and duration fit.
5. Keep the best non-overlapping clips, then build multi-segment promos.
6. Optionally let an LLM re-rate, re-cut by line ID, and rank the top promos pairwise.
7. Rank the results and flag problems.

## Quality

- `python -m pytest tests` (37 tests) checks the hard rules on every promo, single and multi-segment (20–60 s, hook within 3 s, no cut inside a spoken line). It also covers LLM-output validation (invented IDs, malformed replies, out-of-range scores, unknown model names, rejected temperature, bad keys, API failure), scripts with missing or duplicate episode numbers, the website entry point, config values YAML might misread, genre detection and genre scoring on the demo shows, the learned models (reading labels, training, saving, using them without breaking a rule), and a made-up script from a **different genre and looser format**.
- `python eval/evaluate.py --tune` measures the tool against draft editor picks in `data/editor_picks.yaml`. That file sits beside the input pack and is not in this repo, because its notes describe the confidential scripts. It tunes the weights on episodes 1–25 only and reports the held-out episodes 26–50. Results (numbers only) are saved to `eval/report.json`. Held-out result: 16 of 24 picks found (0.67).
- `python eval/evaluate_ml.py` checks whether a learned model beats the formula. The model type and blend are chosen on episodes 1–25 only (train 1–17, validate 18–25); the choice (random forest, blend 0.25) then finds 16 of 24 held-out picks, the same as the formula, with slightly higher precision. Boosted trees found 21 of 24, but they were not the validation choice, so that number isn't claimed. With 45 labels the evidence is thin; the feature is built for a real posting history with thousands of labels.

## Layout

```
promofinder/parser.py   screenplay -> timestamped events
promofinder/signals.py  beat scores: intensity, stakes, openness, resolution -> hook, cliffhanger
promofinder/genre.py    genre detection, genre overrides, mood tags
promofinder/learn.py    learn from posted promos: features, trees, forest, boosting, logistic regression
promofinder/engine.py   clip search, scoring, selection, multi-segment promos, ranking, flags
promofinder/llm.py      swappable AI reviewer (one HTTP function, cached, validated, pairwise ranking)
promofinder/report.py   JSON / CSV / Markdown / HTML outputs
promofinder/web.py      entry point the website calls (same engine)
config.yaml             every word list, weight and rule
run.py                  command line (find promos)
train.py                command line (learn from posted promos)
index.html, site/       the website; vendor/ holds its bundled runtime and fonts
samples/                three made-up 3-episode demo shows (drama, thriller, comedy)
eval/, tests/           editor-pick benchmark and tests
```
