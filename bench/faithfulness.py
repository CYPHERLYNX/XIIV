"""Faithfulness benchmark: structural enforcement vs prompt-only hoping.

Same model (XIIV-52.5M), same retrieved Wikipedia facts, same seeds.
Arm A (enforce=True): facts anchored + key terms as required token ids.
Arm B (enforce=False): facts pasted into the prompt as ordinary context.

Metric: recall of single-token key fact-terms in the generated answer,
plus the verified guarantee rate for arm A. Run:
    .venv/bin/python bench/faithfulness.py
"""
import json
import sys
import time

sys.path.insert(0, ".")

from xiiv import XIIV
from xiiv.knowledge import pipeline

QUESTIONS = [
    "What is the capital of France?",
    "Who painted the Mona Lisa?",
    "What is the largest planet in the solar system?",
    "In which year did World War II end?",
    "What is the chemical symbol for gold?",
    "Who wrote Romeo and Juliet?",
    "What is the capital of Japan?",
    "How many continents are there?",
    "What year did the Titanic sink?",
    "Who invented the telephone?",
    "What is the fastest land animal?",
    "Which planet is known as the Red Planet?",
]


def term_recall(answer, terms, tok):
    if not terms:
        return None
    low = answer.lower()
    hit = sum(1 for t in terms if tok.decode(t).lower() in low)
    return hit / len(terms)


def main():
    model = XIIV("weights_52m/xiiv-52m-fp16.safetensors")
    rows = []
    for i, q in enumerate(QUESTIONS):
        facts = pipeline.retrieve(q, k_wiki=2, k_news=0)
        if not facts:
            print(f"[{i+1:2d}] SKIP (no facts): {q}")
            continue
        terms = pipeline.key_terms([f["text"] for f in facts], model.tok,
                                   single_token=True)
        t0 = time.time()
        a = pipeline.answer(q, model, steps=8, seed=7, enforce=True)
        t1 = time.time()
        b = pipeline.answer(q, model, steps=8, seed=7, enforce=False)
        t2 = time.time()
        ra = term_recall(a["answer"], terms, model.tok)
        rb = term_recall(b["answer"], terms, model.tok)
        rows.append({"q": q, "recall_enforced": ra, "recall_prompt": rb,
                     "guaranteed": a["guaranteed"],
                     "secs_enforced": round(t1 - t0, 1),
                     "secs_prompt": round(t2 - t1, 1)})
        print(f"[{i+1:2d}] enforced={ra} prompt={rb} "
              f"guaranteed={a['guaranteed']} :: {q[:44]}")
    ea = [r["recall_enforced"] for r in rows
          if r["recall_enforced"] is not None]
    eb = [r["recall_prompt"] for r in rows
          if r["recall_prompt"] is not None]
    g = sum(1 for r in rows if r["guaranteed"]) / max(len(rows), 1)
    print(f"\nN={len(rows)}  mean recall enforced={sum(ea)/len(ea):.3f} "
          f"prompt-only={sum(eb)/len(eb):.3f}  guarantee_rate={g:.3f}")
    with open("bench/faithfulness_results.json", "w") as f:
        json.dump(rows, f, indent=1)
    print("wrote bench/faithfulness_results.json")


if __name__ == "__main__":
    main()
