"""Runs eval/questions.json through search and answer, and reports two scores:

- retrieval hit rate: for questions with an expected document, is it in the user's top 3 chunks?
- answer score: does the answer cite the expected document, or refuse when expected_doc is null?

Exits 1 when the retrieval hit rate is below --min (the CI gate).  python evaluate.py --min 0.9
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
        retrieved = {c.doc_id for c, _ in hits}
        reply = "".join(answer_stream(q["question"], hits, llm)).strip()
        if q["expected_doc"] is None:
            ok = reply == REFUSAL
        else:
            ok = f"[{q['expected_doc']}#" in reply
        hit = None if q["expected_doc"] is None else q["expected_doc"] in retrieved
        rows.append({"user": q["user"], "question": q["question"], "ok": ok, "hit": hit, "reply": reply[:160]})
    answerable = [r for r in rows if r["hit"] is not None]
    hit_rate = sum(r["hit"] for r in answerable) / len(answerable)
    return hit_rate, sum(r["ok"] for r in rows) / len(rows), rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--min", type=float, default=0.9, help="minimum retrieval hit rate")
    args = p.parse_args()
    llm = GroqLLM() if os.environ.get("GROQ_API_KEY") else FakeLLM()
    hit_rate, answer_score, rows = evaluate(build_store(), load_users(), json.load(open("eval/questions.json")), llm)
    for r in rows:
        print(("PASS " if r["ok"] else "FAIL ") + f"{r['user']:<6} {r['question']}  ->  {r['reply']}")
    print(f"\nretrieval hit rate {hit_rate:.3f} (minimum {args.min}), answer score {answer_score:.3f} "
          f"with {type(llm).__name__}")
    if hit_rate < args.min:
        sys.exit(1)


if __name__ == "__main__":
    main()
