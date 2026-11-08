// Promo Finder website. The engine runs in a worker (Python via Pyodide); this file is the UI.
// All text from uploaded scripts is inserted with textContent, never as HTML.
import { load as yamlLoad } from "../vendor/js-yaml.mjs";

const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];
const DEMO_FILES = ["episode-001-demo.txt", "episode-002-demo.txt", "episode-003-demo.txt"];
const DEMOS = { drama: "demo_show", thriller: "demo_thriller", comedy: "demo_comedy" };
const BAD_FLAGS = new Set(["weak_hook", "weak_cliffhanger", "resolves_before_cut", "hook_after_3s",
  "cuts_through_dialogue", "duration_out_of_range", "llm_failed"]);
const FLAG_WORDS = {
  weak_hook: "Weak hook", weak_cliffhanger: "Weak cliffhanger", resolves_before_cut: "Ends calmly",
  dead_air: "Dead air", starts_mid_sentence: "Starts mid-sentence", mostly_voice_over: "Mostly voice-over",
  no_dialogue: "No dialogue", hook_after_3s: "Hook after 3 s", cuts_through_dialogue: "Cuts a line",
  duration_out_of_range: "Wrong length", multi_segment_editor_check: "Joined: editor check",
  check_ad_policy: "Check ad policy", llm_failed: "AI review failed", llm_skipped: "AI skipped",
  llm_invalid_id: "AI cited a missing line", llm_recut_rejected_duration: "AI re-cut rejected",
  llm_missing_scores: "AI reply incomplete",
};

const state = { files: [], source: "none", cfg: null, payload: null, runId: 0, pending: new Map(),
  labels: "", labelsName: "", model: null, modelInfo: null, useModel: false };

// ------------------------------------------------------------------ helpers
function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v == null || v === false) continue;
    if (k === "class") node.className = v;
    else if (k === "text") node.textContent = v;
    else node.setAttribute(k, v === true ? "" : v);
  }
  for (const c of children.flat()) if (c != null) node.append(c instanceof Node ? c : document.createTextNode(c));
  return node;
}
const svgEl = (tag, attrs = {}) => {
  const n = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v);
  return n;
};
function fmt(t) {
  const m = Math.floor(t / 60), s = t - m * 60;
  return `${String(m).padStart(2, "0")}:${s.toFixed(1).padStart(4, "0")}`;
}
const plural = (n, one, many = one + "s") => `${n} ${n === 1 ? one : many}`;
function download(name, text, type) {
  const url = URL.createObjectURL(new Blob([text], { type }));
  const a = el("a", { href: url, download: name });
  document.body.append(a); a.click(); a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

// ------------------------------------------------------------------ engine
const worker = new Worker(new URL("./worker.js", import.meta.url), { type: "module" });
let engineReady = false, engineError = "";
worker.onmessage = (event) => {
  const msg = event.data;
  if (msg.type === "ready") {
    engineReady = true;
    document.documentElement.dataset.engine = "ready";
  } else if (msg.type === "fatal") {
    engineError = msg.message;
    document.documentElement.dataset.engine = "error";
    $("#form-msg").textContent = `The engine could not start: ${msg.message}`;
    for (const [, p] of state.pending) p.reject(new Error(msg.message));
  } else {
    const p = state.pending.get(msg.id);
    if (!p) return;
    if (msg.type === "progress") p.onProgress?.(msg);
    else if (msg.type === "result") { state.pending.delete(msg.id); p.resolve(JSON.parse(msg.json)); }
    else if (msg.type === "error") { state.pending.delete(msg.id); p.reject(new Error(msg.message)); }
  }
};
const cfgReady = fetch(new URL("../config.yaml", import.meta.url))
  .then((r) => { if (!r.ok) throw new Error(`config.yaml: HTTP ${r.status}`); return r.text(); })
  .then((t) => (state.cfg = yamlLoad(t)));

function runEngine(files, opts, onProgress, type = "run", labels = "") {
  const id = ++state.runId;
  return cfgReady.then((cfg) => new Promise((resolve, reject) => {
    state.pending.set(id, { resolve, reject, onProgress });
    worker.postMessage({ type, id, files, cfg, opts, labels });
  }));
}

async function loadDemo(kind = "drama") {
  const dir = DEMOS[kind] || DEMOS.drama;
  return Promise.all(DEMO_FILES.map(async (name) => {
    const r = await fetch(new URL(`../samples/${dir}/${name}`, import.meta.url));
    if (!r.ok) throw new Error(`Demo file ${name}: HTTP ${r.status}`);
    return { name, text: await r.text() };
  }));
}

// ------------------------------------------------------------------ rendering
function tape(seg, promo, episode) {
  const W = 600, H = 46, pad = 2, dur = Math.max(seg.episode_duration, 1);
  const x = (t) => pad + (t / dur) * (W - 2 * pad);
  const y = (v) => 40 - v * 32;
  const svg = svgEl("svg", { viewBox: `0 0 ${W} ${H}`, preserveAspectRatio: "none", role: "img",
    "aria-label": `Episode ${seg.episode}: clip from ${fmt(seg.start)} to ${fmt(seg.end)} of ${fmt(dur)}` });
  svg.append(svgEl("rect", { x: 0, y: 40, width: W, height: 4, rx: 2, fill: "rgba(140,152,190,.28)" }));
  svg.append(svgEl("rect", { x: x(seg.start), y: 2, width: Math.max(2, x(seg.end) - x(seg.start)), height: 42,
    rx: 4, fill: "#FFC53D", "fill-opacity": 0.13, stroke: "#FFC53D", "stroke-width": 1.2 }));
  if (episode) {
    const pts = (i) => episode.curve.map((c) => `${x(c[0]).toFixed(1)},${y(c[i]).toFixed(1)}`).join(" ");
    svg.append(svgEl("polyline", { points: pts(1), fill: "none", stroke: "#FF6B5E", "stroke-width": 1.4, opacity: 0.75, "vector-effect": "non-scaling-stroke" }));
    svg.append(svgEl("polyline", { points: pts(2), fill: "none", stroke: "#4D8DFF", "stroke-width": 1.4, opacity: 0.75, "vector-effect": "non-scaling-stroke" }));
  }
  const hook = seg.lines.find((l) => l.mark === "hook"), cut = seg.lines.find((l) => l.mark === "cut");
  if (hook) svg.append(svgEl("rect", { x: x(hook.t) - 1.5, y: 0, width: 3, height: 46, rx: 1.5, fill: "#FF6B5E" }));
  if (cut) svg.append(svgEl("rect", { x: x(cut.t) - 1.5, y: 0, width: 3, height: 46, rx: 1.5, fill: "#4D8DFF" }));
  return el("div", { class: "tape" }, svg,
    el("div", { class: "tape-label" }, el("span", { text: `Episode ${seg.episode}` }),
      el("span", { text: `${fmt(seg.start)} to ${fmt(seg.end)} of ${fmt(dur)}` })));
}

function scriptLines(seg, maxLines) {
  const lines = seg.lines;
  let shown = lines, hidden = 0;
  if (maxLines && lines.length > maxLines) {
    const head = Math.ceil(maxLines / 2), tail = maxLines - head;
    shown = [...lines.slice(0, head), null, ...lines.slice(lines.length - tail)];
    hidden = lines.length - maxLines;
  }
  return shown.map((l) => l === null ? el("div", { class: "script-more", text: plural(hidden, "more line") }) : lineNode(l));
}

function lineNode(l) {
  {
    const kind = l.kind === "dialogue" ? "dialogue" : "action";
    const txt = el("span", { class: "txt" });
    if (l.kind === "dialogue") {
      txt.append(el("b", { text: l.speaker.toUpperCase() + (l.vo ? " (V.O.)" : "") }), " ");
      if (l.cue) txt.append(el("span", { class: "cue", text: `(${l.cue}) ` }));
      txt.append(`“${l.text}”`);
    } else {
      txt.textContent = l.text;
    }
    const mark = l.mark ? el("span", { class: "mark", text: l.mark === "hook" ? "Hook" : "Cut" }) : el("span");
    return el("div", { class: `line ${kind} ${l.mark}`.trim() }, el("span", { class: "t", text: fmt(l.t) }), mark, txt);
  }
}

function promoCard(p, rankLabel, opts = {}) {
  const node = $("#t-promo").content.firstElementChild.cloneNode(true);
  const episodes = new Map((state.payload?.episodes || []).map((e) => [e.number, e]));
  $(".rank", node).textContent = rankLabel;
  $(".promo-title h4", node).textContent = p.segments
    .map((s) => `Episode ${s.episode}, ${fmt(s.start)} to ${fmt(s.end)}`).join("  +  ");
  $(".promo-meta", node).textContent = `${Math.round(p.duration)} s, ${p.kind === "multi" ? "two moments joined" : "one continuous clip"}`;
  for (const m of p.moods || []) $(".moods", node).append(el("li", { text: m }));
  const score = p.scores.final ?? p.scores.total;
  $(".score", node).textContent = `${Math.round(score)}`;
  $(".score", node).setAttribute("aria-label", `Clip score ${Math.round(score)} out of 100`);
  const tapes = $(".tapes", node);
  const script = $(".script", node);
  for (const seg of p.excerpt) {
    tapes.append(tape(seg, p, episodes.get(seg.episode)));
    if (p.excerpt.length > 1) script.append(el("div", { class: "seg-head", text: `Episode ${seg.episode}, ${fmt(seg.start)} to ${fmt(seg.end)}` }));
    script.append(...scriptLines(seg, opts.maxLines));
  }
  $(".why", node).textContent = p.why;
  $(".reason", node).textContent = p.rank_reason;
  const flags = $(".flags", node);
  const help = state.payload?.flag_help || {};
  for (const f of p.flags) {
    flags.append(el("li", { class: BAD_FLAGS.has(f) ? "bad" : null, title: help[f] || null, text: FLAG_WORDS[f] || f }));
  }
  if (!p.flags.length) flags.append(el("li", { text: "No flags" }));
  return node;
}

function renderResults(payload) {
  state.payload = payload;
  const box = $("#results");
  box.replaceChildren();
  const qc = payload.qc, meta = payload.meta;
  const byId = new Map(payload.promos.map((p) => [p.id, p]));

  const json = () => { const { csv, ...rest } = payload; return JSON.stringify(rest, null, 2); };
  box.append(el("div", { class: "summary-bar" },
    el("span", { class: "stat" }, el("b", { text: meta.episodes }), meta.episodes === 1 ? "episode" : "episodes"),
    el("span", { class: "stat" }, el("b", { text: qc.promos }), "promos"),
    el("span", { class: "stat" }, el("b", { text: qc.rule_violations_in_post_first }), "rule breaks in the top list"),
    el("span", { class: "stat" }, el("b", { text: `${meta.seconds} s` }), "to run"),
    el("span", { class: "mode-chip", text: meta.mode.replace(/^rules only/, "Rules only").replace(/^rules \+ AI/, "Rules + AI") }),
    payload.genre?.label ? el("span", { class: "genre-chip", title: "Share of lines with each genre's marker words: " +
      Object.entries(payload.genre.rates || {}).map(([g, r]) => `${g} ${(r * 100).toFixed(0)}%`).join(", "),
      text: `${payload.genre.label}${payload.genre.detected ? ", detected" : ", chosen"}` }) : null,
    el("span", { class: "downloads" },
      Object.assign(el("button", { class: "btn btn-ghost-dark", type: "button", text: "Download JSON" }), { onclick: () => download("promos.json", json(), "application/json") }),
      Object.assign(el("button", { class: "btn btn-ghost-dark", type: "button", text: "Download CSV" }), { onclick: () => download("promos.csv", payload.csv, "text/csv") }))));

  if (meta.llm_errors?.length) {
    box.append(el("p", { class: "warn", text: `AI review had ${plural(meta.llm_errors.length, "problem")}; rules results were kept there. First: ${meta.llm_errors[0]}` }));
  }
  if (qc.episodes_with_no_strong_promo?.length) {
    box.append(el("p", { class: "warn", text: `No strong promo found in ${qc.episodes_with_no_strong_promo.length === 1 ? "episode" : "episodes"} ${qc.episodes_with_no_strong_promo.join(", ")}. Quiet episodes are flagged rather than forced to the top.` }));
  }

  box.append(el("h3", { class: "group", text: "Post these first" }));
  payload.post_first_ids.forEach((id, i) => box.append(promoCard(byId.get(id), String(i + 1), { maxLines: 12 })));

  // all promos table
  box.append(el("h3", { class: "group", text: `All ${payload.promos.length} promos` }));
  const epSel = el("select", { "aria-label": "Filter by episode" }, el("option", { value: "", text: "All episodes" }),
    ...payload.episodes.map((e) => el("option", { value: e.number, text: `Episode ${e.number}` })));
  const kindSel = el("select", { "aria-label": "Filter by kind" }, el("option", { value: "", text: "Single and joined" }),
    el("option", { value: "single", text: "Single clips" }), el("option", { value: "multi", text: "Joined promos" }));
  box.append(el("div", { class: "table-filters" }, epSel, kindSel));
  const tbody = el("tbody");
  const table = el("table", { class: "all-table" },
    el("thead", {}, el("tr", {}, ...["Rank", "Clip", "Length", "Score", "Hook", "Cliffhanger", "Mood", "Flags"].map((h) => el("th", { scope: "col", text: h })))), tbody);
  box.append(el("div", { class: "table-scroll" }, table));
  const draw = () => {
    tbody.replaceChildren();
    for (const p of payload.promos) {
      if (epSel.value && !p.episodes.includes(Number(epSel.value))) continue;
      if (kindSel.value && p.kind !== kindSel.value) continue;
      const btn = el("button", { class: "linkish", type: "button", "aria-expanded": "false",
        text: p.segments.map((s) => `Ep ${s.episode} ${fmt(s.start)}–${fmt(s.end)}`).join(" + ") });
      const row = el("tr", {}, el("td", { class: "num", text: p.rank }), el("td", {}, btn),
        el("td", { class: "num", text: `${Math.round(p.duration)} s` }),
        el("td", { class: "num", text: Math.round(p.scores.final ?? p.scores.total) }),
        el("td", { class: "num", text: p.scores.hook.toFixed(2) }), el("td", { class: "num", text: p.scores.cliffhanger.toFixed(2) }),
        el("td", { text: (p.moods || []).join(", ") || "None" }),
        el("td", { text: p.flags.map((f) => FLAG_WORDS[f] || f).join(", ") || "None" }));
      btn.onclick = () => {
        const open = row.nextElementSibling?.classList.contains("detail-row");
        if (open) { row.nextElementSibling.remove(); btn.setAttribute("aria-expanded", "false"); return; }
        row.after(el("tr", { class: "detail-row" }, el("td", { colspan: 8 }, promoCard(p, `#${p.rank}`, { maxLines: 30 }))));
        btn.setAttribute("aria-expanded", "true");
      };
      tbody.append(row);
    }
  };
  epSel.onchange = draw; kindSel.onchange = draw; draw();
}

// ------------------------------------------------------------------ controls
const form = $("#controls"), runBtn = $("#run"), msg = $("#form-msg");
const fileInput = $("#files"), drop = $("#drop");

function setFiles(list, source) {
  state.files = list; state.source = source;
  $("#learn-scripts").textContent = list.length
    ? `Uses the ${plural(list.length, "script")} chosen above${source.startsWith("demo") ? ` (${source.slice(5)} demo show)` : ""}.`
    : "Uses the scripts chosen above. None chosen yet.";
  $("#file-summary").textContent = list.length
    ? `${plural(list.length, "file")} ready${source.startsWith("demo") ? ` (${source.slice(5)} demo show)` : ""}: ${list.map((f) => f.name).slice(0, 3).join(", ")}${list.length > 3 ? "…" : ""}`
    : "No files chosen.";
}
async function readFiles(fileList) {
  const txt = [...fileList].filter((f) => f.name.toLowerCase().endsWith(".txt") || f.type === "text/plain");
  const skipped = fileList.length - txt.length;
  const files = await Promise.all(txt.map(async (f) => ({ name: f.name, text: await f.text() })));
  files.sort((a, b) => a.name.localeCompare(b.name, undefined, { numeric: true }));
  setFiles(files, "upload");
  msg.textContent = skipped ? `${plural(skipped, "file")} skipped: only .txt screenplays are read.` : "";
}
fileInput.addEventListener("change", () => readFiles(fileInput.files));
for (const ev of ["dragenter", "dragover"]) drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("over"); });
for (const ev of ["dragleave", "drop"]) drop.addEventListener(ev, () => drop.classList.remove("over"));
drop.addEventListener("drop", (e) => { e.preventDefault(); readFiles(e.dataTransfer.files); });

const provider = $("#ai-provider");
provider.addEventListener("change", () => {
  const v = provider.value;
  $("#ai-fields").hidden = v === "none" || v === "mock";
  $("#ai-key-row").hidden = v === "ollama";
  $("#ai-base-row").hidden = !(v === "openai_compatible" || v === "ollama");
});

function readOptions() {
  const min = Number($("#min-d").value), max = Number($("#max-d").value), per = Number($("#per-ep").value);
  if (!(min > 0 && max > min)) throw new Error("The longest clip must be longer than the shortest clip.");
  if (!(per >= 1 && per <= 5)) throw new Error("Promos kept per episode must be between 1 and 5.");
  const promo = { min_duration: min, max_duration: max, per_episode: per, max_montages: $("#joins").checked ? 12 : 0 };
  const ideal = state.cfg?.promo?.ideal_duration ?? 35;
  if (ideal < min || ideal > max) promo.ideal_duration = (min + max) / 2;   // only when the default falls outside the range
  const opts = { overrides: { promo, genre: { mode: $("#genre").value } } };
  if (state.useModel && state.model) opts.ml_model = state.model;
  if (provider.value !== "none") {
    opts.ai = { provider: provider.value, model: $("#ai-model").value.trim(), api_key: $("#ai-key").value.trim(), base_url: $("#ai-base").value.trim() };
  }
  return opts;
}

async function runTool() {
  msg.textContent = "";
  let opts;
  try { opts = readOptions(); } catch (e) { msg.textContent = e.message; return; }
  if (engineError) { msg.textContent = `The engine could not start: ${engineError}`; return; }
  if (!state.files.length) { msg.textContent = "Choose screenplay files first, or try a demo show."; return; }
  runBtn.disabled = true;
  runBtn.textContent = engineReady ? "Finding promos…" : "Starting the engine…";
  try {
    const payload = await runEngine(state.files, opts, (p) => { runBtn.textContent = `Reading episode ${p.i} of ${p.n}…`; });
    if (payload.error) { msg.textContent = payload.error; return; }
    renderResults(payload);
    $("#results").scrollIntoView({ behavior: matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth", block: "start" });
  } catch (e) {
    msg.textContent = `Something went wrong: ${e.message}`;
  } finally {
    runBtn.disabled = false;
    runBtn.textContent = "Find promos";
  }
}
form.addEventListener("submit", (e) => { e.preventDefault(); runTool(); });
for (const b of $$('[data-action="demo"]')) {
  b.addEventListener("click", async () => {
    try { setFiles(await loadDemo(b.dataset.demo), `demo ${b.dataset.demo || "drama"}`); } catch (e) { msg.textContent = e.message; return; }
    $("#tool").scrollIntoView({ behavior: "smooth" });
    runTool();
  });
}

// ------------------------------------------------------------------ learn from posted promos
const SIGNAL_WORDS = {
  hook: "hook score", cliffhanger: "cliffhanger score", energy: "energy", clarity: "clarity",
  duration_fit: "length fit", formula: "formula score", seconds: "clip length", start_pos: "where the clip starts",
  end_pos: "where the clip ends", ends_episode: "ends at the episode's end", hook_intensity: "how intense the hook line is",
  hook_stakes: "stakes in the hook line", hook_is_dialogue: "hook is a spoken line", hook_offset: "how fast the hook arrives",
  cut_open: "how open the last line is", cut_stakes: "stakes at the cut", cut_intensity: "intensity at the cut",
  cut_resolution: "calm at the cut", cut_is_dialogue: "cut on a spoken line", cut_question: "cut on a question",
  cut_trailing: "cut on an unfinished line", cut_threat: "cut on a threat", cut_sting: "cut on a dramatic sound",
  end_resolution: "calm ending", max_gap: "longest silence", talk_share: "how much talking", speakers: "number of speakers",
  lines: "number of lines", scenes: "scene changes", voice_over_share: "voice-over share", mean_intensity: "average intensity",
  max_stakes: "highest stakes", mean_hook: "average hook score", mean_cliff: "average cliffhanger score",
};
const signalName = (f) => SIGNAL_WORDS[f] || f.replace(/^has_/, "has ").replace(/^cut_/, "cut on ").replaceAll("_", " ");
const learnMsg = $("#learn-msg"), trainBtn = $("#train");

function setModel(json, info) {
  state.model = json; state.modelInfo = info; state.useModel = true;
  $("#model-chip").hidden = false;
  $("#model-chip-text").textContent = `Using your learned model (${info.label})`;
}
$("#model-off").addEventListener("click", () => {
  state.useModel = false; $("#model-chip").hidden = true;
  $$(".use-model").forEach((b) => { b.textContent = "Use this model to find promos"; });
});

async function readLabels(file) {
  if (!file) return;
  state.labels = await file.text(); state.labelsName = file.name;
  const rows = state.labels.trim().split(/\r?\n/).length - 1;
  $("#labels-summary").textContent = `${file.name}: ${plural(Math.max(0, rows), "promo")} listed.`;
  learnMsg.textContent = "";
}
const labelsInput = $("#labels"), labelsDrop = $("#labels-drop");
labelsInput.addEventListener("change", () => readLabels(labelsInput.files[0]));
for (const ev of ["dragenter", "dragover"]) labelsDrop.addEventListener(ev, (e) => { e.preventDefault(); labelsDrop.classList.add("over"); });
for (const ev of ["dragleave", "drop"]) labelsDrop.addEventListener(ev, () => labelsDrop.classList.remove("over"));
labelsDrop.addEventListener("drop", (e) => { e.preventDefault(); readLabels(e.dataTransfer.files[0]); });
$("#sample-labels").addEventListener("click", () => download("labels-sample.csv",
  "episode,start,end,label\n1,00:46.8,01:10.5,1\n2,00:02.8,00:37.2,1\n2,01:05.0,01:40.0,0\n", "text/csv"));

const MODEL_LABEL = { forest: "random forest", boosted: "boosted trees", logistic: "logistic regression" };
const pct = (v) => `${Math.round(v * 100)}%`;

function renderReport(rep, modelJson) {
  const box = $("#learn-results");
  box.replaceChildren();
  const f = rep.formula, m = rep.ml;
  const cmp = (a, b, fmt) => {
    const cls = b > a ? "better" : b < a ? "worse" : null;
    return [el("td", { text: fmt(a) }), el("td", { class: cls, text: fmt(b) })];
  };
  const range = (eps) => eps.length > 1 ? `episodes ${eps[0]}–${eps[eps.length - 1]}` : `episode ${eps[0]}`;
  box.append(el("div", { class: "summary-bar" },
    el("span", { class: "stat" }, el("b", { text: rep.train_rows }), "clips learned from"),
    el("span", { class: "stat" }, el("b", { text: rep.train_positives }), "good clips among them"),
    el("span", { class: "stat" }, el("b", { text: `${rep.seconds} s` }), "to train and test"),
    el("span", { class: "mode-chip", text: MODEL_LABEL[rep.model] || rep.model })));
  box.append(el("p", { class: "note", text: `Trained on ${range(rep.train_episodes)}, tested on ${range(rep.test_episodes)} that the model never saw.` }));
  box.append(el("div", { class: "table-scroll" }, el("table", { class: "compare" },
    el("thead", {}, el("tr", {}, el("th", { scope: "col", text: "On the test episodes" }), el("th", { scope: "col", text: "Formula" }), el("th", { scope: "col", text: "With your model" }))),
    el("tbody", {},
      el("tr", {}, el("td", { text: "Posted promos found" }), ...cmp(f.rate, m.rate, (v) => `${pct(v)}`)),
      el("tr", {}, el("td", { text: "Best promo per episode matches a posted one" }), ...cmp(f.top1, m.top1, pct)),
      el("tr", {}, el("td", { text: "Good clips ranked above the rest (AUC)" }), ...cmp(rep.clip_auc.formula, rep.clip_auc.model, (v) => v.toFixed(2)))))));
  box.append(el("p", { class: "note", text: `Found ${f.found} of ${f.of} posted promos with the formula and ${m.found} of ${m.of} with your model. With few labels, a difference of one or two promos can be chance.` }));
  box.append(el("h3", { class: "group", text: "What the model pays attention to" }));
  box.append(el("ul", { class: "signals" }, ...rep.top_features.map(([name]) => el("li", { text: signalName(name) }))));
  const use = Object.assign(el("button", { class: "btn btn-primary use-model", type: "button", text: "Use this model to find promos" }), {
    onclick: () => {
      setModel(modelJson, { label: MODEL_LABEL[rep.model] || rep.model });
      use.textContent = "Model is on";
      $("#tool").scrollIntoView({ behavior: "smooth" });
      if (state.files.length) runTool();
    },
  });
  const save = Object.assign(el("button", { class: "btn btn-ghost-dark", type: "button", text: "Download model" }), {
    onclick: () => download(`promo-model-${rep.model}.json`, modelJson, "application/json"),
  });
  box.append(el("div", { class: "model-actions" }, use, save));
}

async function trainModel() {
  learnMsg.textContent = "";
  if (!state.files.length) { learnMsg.textContent = "Choose the scripts above first: the episodes your labels refer to."; return; }
  if (!state.labels) { learnMsg.textContent = "Choose a labels .csv file first."; return; }
  if (engineError) { learnMsg.textContent = `The engine could not start: ${engineError}`; return; }
  let opts;
  try { opts = readOptions(); } catch (e) { learnMsg.textContent = e.message; return; }
  delete opts.ml_model; delete opts.ai;
  opts.model = $("#ml-kind").value; opts.test_share = Number($("#ml-test").value);
  trainBtn.disabled = true; trainBtn.textContent = engineReady ? "Training…" : "Starting the engine…";
  try {
    const out = await runEngine(state.files, opts, (p) => { trainBtn.textContent = `${p.msg}…`; }, "train", state.labels);
    if (out.error) { learnMsg.textContent = out.error; return; }
    renderReport(out.report, out.model_json);
  } catch (e) {
    learnMsg.textContent = `Something went wrong: ${e.message}`;
  } finally {
    trainBtn.disabled = false; trainBtn.textContent = "Train and test";
  }
}
$("#learn-form").addEventListener("submit", (e) => { e.preventDefault(); trainModel(); });

$("#model-file").addEventListener("change", async (e) => {
  const file = e.target.files[0];
  if (!file) return;
  try {
    const text = await file.text();
    const m = JSON.parse(text);
    if (m.kind !== "promofinder-model") throw new Error("this is not a Promo Finder model file");
    setModel(text, { label: MODEL_LABEL[m.type] || m.type });
    learnMsg.textContent = "";
    $("#learn-results").replaceChildren(el("div", { class: "empty" }, el("h3", { text: "Model loaded" }),
      el("p", { text: m.test_report ? `Saved test result: found ${m.test_report.ml.found} of ${m.test_report.ml.of} posted promos (formula: ${m.test_report.formula.found}).` : "It will be used when you find promos above." })));
  } catch (err) {
    learnMsg.textContent = `Could not load the model: ${err.message}.`;
  }
});
