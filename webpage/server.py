"""Local web UI: paste an article (or a link to one), get propaganda-technique highlights back.

Same model as scripts/predict.py, served to a browser. Run from the repo root
(run paths are resolved relative to where you start it):

    python webpage/server.py --si-run runs/si-base-s13 --tc-run runs/tc-base-s13
    # then open http://127.0.0.1:5000
"""
from __future__ import annotations

import argparse
import ipaddress
import socket
import threading
import time
import traceback
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
import trafilatura
from flask import Flask, jsonify, request, send_from_directory

# Absolute imports, like scripts/predict.py: this file lives outside the ptc
# package, so the relative form (`from .inference ...`) has no package to be relative to.
from ptc.inference import Pipeline, sentence_view
from ptc.render import HTML_HUES
from ptc.techniques import DISPLAY_NAMES

# index.html may sit in webpage/static/ or directly in webpage/; use whichever exists.
HERE = Path(__file__).resolve().parent
STATIC_DIR = HERE / "static" if (HERE / "static" / "index.html").exists() else HERE
REPO_ROOT = HERE.parent  # webpage/ sits at the top of the repo

# The changelog shown on the page, read fresh from the repo on each request.
CHANGELOG_NAMES = ("CHANGELOG.md", "CHANGELOGS.md")

MAX_CHARS = 50_000  # underscores in numeric literals are just digit separators
DOC_ID = "input"

# Limits on fetching a link.
MAX_PAGE_BYTES = 5_000_000   # stop reading a page after 5 MB
FETCH_TIMEOUT_S = 15         # give up if the site goes silent this long
USER_AGENT = "Mozilla/5.0 (compatible; ptc-propaganda-detector)"

app = Flask(__name__)

# Built once in main(). Loading DeBERTa happens inside Pipeline(...), so this
# is the only time the model goes onto the GPU.
_pipe: Pipeline | None = None

# Flask serves each request on its own thread; one model on one GPU should
# only see one article at a time. `with _model_lock:` is like `bracket`.
_model_lock = threading.Lock()


# ---------------------------------------------------------------- links

class FetchError(Exception):
    """A problem with the user's link. The message is shown to them as-is."""


def _check_public_url(url: str) -> None:
    """Only fetch ordinary public web pages.

    Stops a link like http://127.0.0.1:... or http://192.168.1.1/ from making this
    server request things on your own machine or network. Runs on the pasted link
    and again on every redirect (see _CheckedRedirects).
    """
    parts = urlparse(url)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise FetchError("That doesn't look like a web link. It should start with http:// or https://.")
    try:
        infos = socket.getaddrinfo(parts.hostname, parts.port or None)
    except socket.gaierror:
        # `from None` hides the low-level DNS error from the traceback; ours says it better.
        raise FetchError(f"Couldn't find the site {parts.hostname}. Check the link for typos.") from None
    for info in infos:
        ip = ipaddress.ip_address(info[4][0].split("%")[0])
        if not ip.is_global:
            raise FetchError("Links to local or private network addresses aren't allowed.")


class _CheckedRedirects(urllib.request.HTTPRedirectHandler):
    """urllib's normal redirect handling, plus the public-address check on each hop.

    Subclassing and overriding one method is how you customise library behaviour
    in Python; think of supplying your own instance method in place of a default.
    """

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        _check_public_url(newurl)  # raises FetchError, which aborts the download
        return super().redirect_request(req, fp, code, msg, headers, newurl)


_opener = urllib.request.build_opener(_CheckedRedirects)


def _download(url: str) -> str:
    """Fetch a page's HTML with a size cap and timeout. No JavaScript, cookies or files."""
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with _opener.open(req, timeout=FETCH_TIMEOUT_S) as resp:
            raw = resp.read(MAX_PAGE_BYTES + 1)  # one extra byte tells us it was too big
            charset = resp.headers.get_content_charset() or "utf-8"
    except urllib.error.HTTPError as exc:  # must come before URLError: it's a subclass
        raise FetchError(f"The site answered with an error (HTTP {exc.code}). It may block automated "
                         "requests or need a login. Paste the article text instead.") from None
    except (urllib.error.URLError, TimeoutError):
        raise FetchError("Couldn't reach that site. It may be down or not responding.") from None
    if len(raw) > MAX_PAGE_BYTES:
        raise FetchError("That page is over 5 MB, too large to analyse. Paste the article text instead.")
    try:
        return raw.decode(charset, errors="replace")
    except LookupError:  # the site declared a charset Python doesn't know
        return raw.decode("utf-8", errors="replace")


def fetch_article(url: str) -> tuple[str, str | None]:
    """Download a page and return (article text, title).

    trafilatura only parses here; all network access goes through _download.
    """
    _check_public_url(url)
    html = _download(url)
    text = trafilatura.extract(html, include_comments=False, include_tables=False)
    if not text or not text.strip():
        raise FetchError("Downloaded the page but found no article text on it. Pages built with "
                         "JavaScript often look empty to a fetcher. Paste the article text instead.")
    meta = trafilatura.extract_metadata(html)
    return text, (meta.title if meta else None)


# ---------------------------------------------------------------- model

def run_model(text: str) -> tuple[pd.DataFrame, list[dict]]:
    """Same calls predict.py makes, for a single document."""
    assert _pipe is not None, "run_model called before main() built the pipeline"
    with _model_lock:
        preds = _pipe({DOC_ID: text})
    preds = preds[preds["article_id"] == DOC_ID]
    sentences = sentence_view(text, preds)
    return preds, sentences


# ---------------------------------------------------------------- routes

@app.get("/")
def index():
    return send_from_directory(STATIC_DIR, "index.html")


@app.get("/api/changelog")
def changelog():
    for name in CHANGELOG_NAMES:
        path = REPO_ROOT / name
        if path.is_file():
            return jsonify(name=name, markdown=path.read_text(encoding="utf-8"))
    return jsonify(error="No CHANGELOG.md found at the top of the repo."), 404


@app.post("/api/analyze")
def analyze():
    payload = request.get_json(silent=True) or {}
    t0 = time.perf_counter()

    url = payload.get("url")
    source = None
    if url:
        if not isinstance(url, str):
            return jsonify(error="The link must be text."), 400
        try:
            text, title = fetch_article(url.strip())  # tuple unpacking, like a pattern match on (a, b)
        except FetchError as exc:
            return jsonify(error=str(exc)), 422
        except Exception as exc:
            traceback.print_exc()
            return jsonify(error=f"Couldn't fetch that page: {exc}"), 502
        source = {"url": url.strip(), "title": title}
    else:
        text = payload.get("text", "")
        if not isinstance(text, str) or not text.strip():
            return jsonify(error="Paste an article or a link into the box, then run again."), 400

    if len(text) > MAX_CHARS:
        return jsonify(error=f"Text is {len(text):,} characters; the limit is {MAX_CHARS:,}."), 413

    try:
        preds, sentences = run_model(text)
    except Exception as exc:  # local tool: show the real error in the UI
        traceback.print_exc()
        return jsonify(error=f"Model failed: {exc}"), 500
    elapsed = time.perf_counter() - t0

    preds = preds[preds["technique"] != "NONE"].sort_values(["start", "end"])

    # int()/float() matter: pandas hands back numpy scalars, which json can't always serialise.
    spans = [
        {
            "start": int(r.start),
            "end": int(r.end),
            "technique": str(r.technique),
            "confidence": float(r.confidence),
            "technique_confidence": float(r.technique_confidence),
        }
        for r in preds.itertuples(index=False)
    ]
    techniques = {
        t: {"name": DISPLAY_NAMES.get(t, t), "hue": HTML_HUES.get(t)}
        for t in {s["technique"] for s in spans}
    }
    # Same ordering and naming as render_sentences().
    sentence_rows = [
        {
            "sentence": r["sentence"],
            "confidence": float(r["confidence"]),
            "techniques": [DISPLAY_NAMES.get(t, t) for t in r["techniques"]],
        }
        for r in sorted(sentences, key=lambda r: -r["confidence"])
    ]
    # `text` goes back too: for a link, the browser has never seen it, and the
    # span offsets only make sense against exactly this string.
    return jsonify(text=text, source=source, spans=spans, techniques=techniques,
                   sentences=sentence_rows, elapsed_s=elapsed)


def main() -> None:
    ap = argparse.ArgumentParser(description="Serve the ptc web UI.")
    ap.add_argument("--si-run", required=True, help="span-identification run dir")
    ap.add_argument("--tc-run", required=True, help="technique-classification run dir")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5000)
    ap.add_argument("--no-warmup", action="store_true",
                    help="skip the test prediction at startup")
    args = ap.parse_args()

    # `global` says "assign the module-level _pipe", not a new local variable.
    # Python has no `let`: assigning a name inside a function makes it local by default.
    global _pipe
    print("Loading model...", flush=True)
    _pipe = Pipeline(args.si_run, args.tc_run, args.device)

    if not args.no_warmup:
        # First CUDA call is slow (kernel setup); pay it now, not on the first paste.
        run_model("This is a warm-up sentence.")
    print(f"Ready on http://{args.host}:{args.port}", flush=True)

    # debug=False on purpose: the debug reloader starts a second process,
    # which would load the model onto the GPU twice.
    app.run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()