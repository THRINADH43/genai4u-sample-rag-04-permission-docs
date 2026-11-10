"""Runs eval/questions.json through search + answer. A question passes when the answer cites
the expected document, or refuses when expected_doc is null. Exits 1 below --min (CI gate).

python evaluate.py --min 0.85
"""

import argparse
import json
import os
import sys

from docservice.answer import REFUSAL, FakeLLM, GroqLLM, answer_stream
from docservice.ingest import Ingestor
from docservice.search import hybrid_search
from docservice.server import load_users
from docservice.store import DocumentStore


def build_store():
    store = DocumentStore()
    ingestor = Ingestor(store)
    for name in sorted(os.listdir("seed")):
        d = json.load(open(os.path.join("seed", name)))
        ingestor.submit("upsert", d["id"], d["text"], d["groups"])
    ingestor.wait()
    return store


def evaluate(store, users, questions, llm):
    rows = []
    for q in questions:
        hits = hybrid_search(store, users[q["user"]]["groups"], q["question"])
        reply = "".join(answer_stream(q["question"], hits, llm)).strip()
        if q["expected_doc"] is None:
            ok = reply == REFUSAL
        else:
            ok = f"[{q['expected_doc']}#" in reply
        rows.append({"user": q["user"], "question": q["question"], "ok": ok, "reply": reply[:160]})
    return sum(r["ok"] for r in rows) / len(rows), rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--min", type=float, default=0.85)
    args = p.parse_args()
    llm = GroqLLM() if os.environ.get("GROQ_API_KEY") else FakeLLM()
    score, rows = evaluate(build_store(), load_users(), json.load(open("eval/questions.json")), llm)
    for r in rows:
        print(("PASS " if r["ok"] else "FAIL ") + f"{r['user']:<6} {r['question']}  ->  {r['reply']}")
    print(f"\nscore {score:.3f} with {type(llm).__name__} (minimum {args.min})")
    if score < args.min:
        sys.exit(1)


if __name__ == "__main__":
    main()
