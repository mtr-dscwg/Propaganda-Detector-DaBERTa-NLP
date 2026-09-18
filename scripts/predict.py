"""Run the detector on any text and show it in the terminal (the text-based UI).

    python scripts/predict.py --si-run runs/si-base-s13 --tc-run runs/tc-base-s13 article.txt
    python scripts/predict.py ... --dir some_folder/ --json out.jsonl --html out_html/
    echo "Some text" | python scripts/predict.py ... -
"""
import argparse
import json
import sys
from pathlib import Path

from ptc.inference import Pipeline, sentence_view
from ptc.render import render_html, render_sentences, render_terminal
from ptc.utils import setup_console


def read_inputs(args) -> dict[str, str]:
    docs = {}
    for f in args.files:
        if f == "-":
            docs["stdin"] = sys.stdin.read()
        else:
            docs[Path(f).stem] = Path(f).read_bytes().decode("utf-8")
    if args.dir:
        for f in sorted(Path(args.dir).glob("*.txt")):
            docs[f.stem] = f.read_bytes().decode("utf-8")
    return docs


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("files", nargs="*")
    p.add_argument("--dir")
    p.add_argument("--si-run", required=True)
    p.add_argument("--tc-run", required=True)
    p.add_argument("--min-confidence", type=float, default=0.0, help="hide highlights below span*label confidence")
    p.add_argument("--json", help="write predictions as JSON lines")
    p.add_argument("--html", help="directory for highlighted HTML pages")
    p.add_argument("--device", default="auto")
    args = p.parse_args()
    setup_console()

    docs = read_inputs(args)
    if not docs:
        p.error("no input (give files, --dir, or - for stdin)")
    pipe = Pipeline(args.si_run, args.tc_run, args.device)
    preds = pipe(docs)

    json_fh = open(args.json, "w", encoding="utf-8", newline="\n") if args.json else None
    for doc_id, text in docs.items():
        doc_preds = preds[preds["article_id"] == doc_id]
        sentences = sentence_view(text, doc_preds)
        print(f"\n\033[1m=== {doc_id} ===\033[0m  ({len(doc_preds)} spans, {len(sentences)} sentences)\n")
        print(render_terminal(text, doc_preds, args.min_confidence))
        print()
        print(render_sentences(sentences))
        if json_fh:
            json_fh.write(json.dumps({"id": doc_id, "spans": doc_preds.drop(columns="article_id").to_dict("records"),
                                      "sentences": sentences}, ensure_ascii=False) + "\n")
        if args.html:
            Path(args.html).mkdir(parents=True, exist_ok=True)
            (Path(args.html) / f"{doc_id}.html").write_text(render_html(text, doc_preds, doc_id), encoding="utf-8")
    if json_fh:
        json_fh.close()


if __name__ == "__main__":
    main()
