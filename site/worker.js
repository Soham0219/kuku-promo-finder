// Runs the Promo Finder Python package inside the browser (Pyodide), off the main thread.
// The engine files are fetched from this same site, so the website runs exactly the code in the repo.
import { loadPyodide } from "../vendor/pyodide/pyodide.mjs";

const ROOT = new URL("../", import.meta.url);
const MODULES = ["__init__", "config", "parser", "signals", "genre", "engine", "report", "llm", "learn", "web"];

// Synchronous HTTP for the optional AI review (allowed inside a worker). Returns a JSON string.
self.httpPostSync = (url, headersJson, bodyJson, timeoutSeconds) => {
  const xhr = new XMLHttpRequest();
  xhr.open("POST", url, false);
  const headers = JSON.parse(headersJson);
  for (const [k, v] of Object.entries(headers)) xhr.setRequestHeader(k, v);
  try {
    xhr.send(bodyJson);
  } catch (err) {
    return JSON.stringify({ status: 0, text: String(err), headers: {} });
  }
  return JSON.stringify({ status: xhr.status, text: xhr.responseText,
                          headers: { "retry-after": xhr.getResponseHeader("retry-after") } });
};

let py;
async function boot() {
  py = await loadPyodide({ indexURL: new URL("vendor/pyodide/", ROOT).href });
  py.FS.mkdirTree("/home/pyodide/promofinder");
  for (const m of MODULES) {
    const res = await fetch(new URL(`promofinder/${m}.py`, ROOT));
    if (!res.ok) throw new Error(`Could not load promofinder/${m}.py (HTTP ${res.status})`);
    py.FS.writeFile(`/home/pyodide/promofinder/${m}.py`, await res.text());
  }
  py.runPython(`
import sys, json, time, pathlib
sys.path.insert(0, "/home/pyodide")
import js
import promofinder.llm as llm

def _browser_post(url, headers, body, params, timeout):
    if params:
        from urllib.parse import urlencode
        url = url + ("&" if "?" in url else "?") + urlencode(params)
    r = json.loads(js.httpPostSync(url, json.dumps(headers), json.dumps(body), timeout))
    if r["status"] == 0:
        raise RuntimeError("The browser blocked the request (network or CORS). Try the command line for this provider.")
    return llm.HTTPResponse(r["status"], r["text"], {k: v for k, v in r["headers"].items() if v})

def _busy_sleep(seconds):
    end = time.time() + min(seconds, 30)
    while time.time() < end:
        pass

llm.http_post = _browser_post
llm.time.sleep = _busy_sleep
llm.CACHE_DIR = pathlib.Path("/tmp/llm-cache")
from promofinder.web import run as promo_run, train as promo_train
`);
}

const ready = boot().then(
  () => self.postMessage({ type: "ready" }),
  (err) => { self.postMessage({ type: "fatal", message: String(err && err.message || err) }); throw err; },
);

self.onmessage = async (event) => {
  const { type, id, files, cfg, opts, labels } = event.data;
  if (type !== "run" && type !== "train") return;
  try {
    await ready;
  } catch {
    return;
  }
  try {
    const progress = (i, n, msg) => self.postMessage({ type: "progress", id, i, n, msg: String(msg) });
    const out = type === "train"
      ? py.globals.get("promo_train")(JSON.stringify(files), JSON.stringify(cfg), labels, JSON.stringify(opts || {}), progress)
      : py.globals.get("promo_run")(JSON.stringify(files), JSON.stringify(cfg), JSON.stringify(opts || {}), progress);
    self.postMessage({ type: "result", id, json: out });
  } catch (err) {
    const msg = String(err && err.message || err).split("\n").filter(Boolean).slice(-1)[0];
    self.postMessage({ type: "error", id, message: msg });
  }
};
