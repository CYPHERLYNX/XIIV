"""Guaranteed-faithful knowledge answers for XIIV.

Pipeline: retrieve facts (Wikipedia + historical news) -> extract key
content terms -> generate with ONLY "Question: ... Answer:" as frozen
prompt and the fact terms as REQUIRED token ids. The slot-reservation
invariant force-places any missing term into the answer, so the answer
contains the facts BY CONSTRUCTION — not because they were pasted into
the prompt and hoped for.

If no verified facts are retrieved, the model says so plainly instead of
hallucinating.
"""
import re

import torch

from ..api import IDENTITY_RESPONSE, is_identity_question
from ..diffusion import check_constraints, sample_trajectory
from . import news as _news
from . import wikipedia as _wiki

MAX_EVIDENCE_TOKENS = 64
MAX_QUESTION_TOKENS = 24
MAX_REQUIRED_TOKENS = 16
TOTAL_LENGTH = 128

NO_EVIDENCE = ("I couldn't find verified information on that in Wikipedia "
               "or the news archive, so I won't guess.")

_STOPWORDS = frozenset((
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "of", "in", "on", "at", "to", "for", "with", "by", "from", "as",
    "and", "or", "but", "it", "its", "this", "that", "these", "those",
    "he", "she", "they", "we", "you", "i", "his", "her", "their", "our",
    "has", "have", "had", "will", "would", "can", "could", "which",
    "what", "when", "where", "who", "whom", "how", "why", "not", "no",
    "into", "over", "under", "between", "during", "after", "before",
))


def key_terms(facts, tokenizer, max_terms=8, single_token=False):
    """Extract constrainable key content terms from fact sentences.

    Keeps capitalized words (names/entities), numbers/years, and long
    content words; drops stopwords. Returns token-id lists, one per
    term. With ``single_token=True``, only terms that encode to exactly
    one token are kept — required for word-level guarantees, since the
    sampler enforces individual token ids (a multi-token word's tokens
    would be placed independently and scatter).
    Deterministic: first-appearance order.
    """
    cands = []
    for fact in facts:
        for w in re.findall(r"[A-Za-z]+|\d+", fact):
            lw = w.lower()
            if lw in _STOPWORDS:
                continue
            if w[0].isupper() or w.isdigit() or len(w) >= 5:
                cands.append(w)
    seen, terms = set(), []
    for c in cands:
        key = c.lower()
        if key in seen:
            continue
        seen.add(key)
        ids = [i for i in tokenizer.encode(c)
               if i not in (tokenizer.mask_id, tokenizer.eos_id)]
        if not ids:
            continue
        if single_token and len(ids) != 1:
            continue
        if not single_token and not (1 <= len(ids) <= 4):
            continue
        terms.append(ids)
        if len(terms) >= max_terms:
            break
    return terms


def retrieve(question, k_wiki=2, k_news=2):
    """Fetch fact records: [{text, source}]."""
    facts = []
    for hit in _wiki.search_wikipedia(question, k=k_wiki):
        sents = re.split(r"(?<=[.!?])\s+", hit["extract"])
        text = " ".join(sents[:2]).strip()
        if text:
            facts.append({"text": text,
                          "source": f"Wikipedia: {hit['title']}"})
    for item in _news.search_news(question, k=k_news):
        facts.append({"text": item["text"],
                      "source": f"News {item['date']}"})
    return facts


def answer(question, model, steps=16, seed=7, temperature=1.0, top_p=0.95,
           enforce=True):
    """Answer ``question``; retrieved facts enforced structurally.

    With ``enforce=True`` (default): the frozen prompt is just the
    question; the top fact sentence is ANCHORED verbatim at the start of
    the answer (position-pinned, byte-exact), and single-token key terms
    from the remaining facts are REQUIRED ids (slot-reservation
    invariant force-places each). Multi-token terms are deliberately
    excluded from required_ids: the sampler enforces token ids, and a
    multi-token word's tokens would scatter instead of forming the word.
    With ``enforce=False``: the facts are pasted into the prompt as
    ordinary context (the standard RAG baseline: hoping, not enforcing).

    Returns {"answer", "facts", "sources", "key_terms", "guaranteed"}.
    ``guaranteed`` is verified post-hoc with check_constraints, not
    claimed.
    """
    if is_identity_question(question):
        return {"answer": IDENTITY_RESPONSE, "facts": [], "sources": [],
                "key_terms": [], "guaranteed": True}
    facts = retrieve(question)
    if not facts:
        return {"answer": NO_EVIDENCE, "facts": [], "sources": [],
                "key_terms": [], "guaranteed": True}

    tok = model.tok
    q_ids = tok.encode("Question: " + question.strip() + "\nAnswer:")
    q_ids = q_ids[:MAX_QUESTION_TOKENS + 8]
    prompt_ids = torch.tensor(q_ids, dtype=torch.long)
    a0 = len(q_ids)  # answer starts here

    constraints = None
    terms = []
    if enforce:
        # Anchor the top fact sentence verbatim at the answer start.
        fact0 = re.split(r"(?<=[.!?])\s+", facts[0]["text"])[0]
        f0_ids = [i for i in tok.encode(fact0)
                  if i not in (tok.mask_id, tok.eos_id)][:30]
        anchors = {a0 + p: t for p, t in enumerate(f0_ids)}
        # Require single-token key terms absent from the anchored fact.
        f0_low = tok.decode(f0_ids).lower()
        terms = [t for t in key_terms([f["text"] for f in facts], tok,
                                      single_token=True)
                 if tok.decode(t).lower() not in f0_low]
        required = sorted({t[0] for t in terms})[:MAX_REQUIRED_TOKENS]
        constraints = {"anchors": anchors}
        if required:
            constraints["required_ids"] = required
    else:
        ev_text = " ".join(f["text"] for f in facts)
        prompt_text = ("Facts: " + ev_text + "\nQuestion: "
                       + question.strip() + "\nAnswer:")
        prompt_ids = torch.tensor(
            tok.encode(prompt_text)[:MAX_EVIDENCE_TOKENS
                                    + MAX_QUESTION_TOKENS + 8],
            dtype=torch.long)

    guaranteed = constraints is not None
    try:
        ids, _ = sample_trajectory(
            model.model, TOTAL_LENGTH, tok.mask_id, tok.eos_id,
            steps=steps, temperature=temperature, top_p=top_p, seed=seed,
            prompt_ids=prompt_ids, constraints=constraints,
            device=model.device)
    except ValueError:
        # unsatisfiable constraint set: honest fallback, no enforcement
        guaranteed = False
        constraints = None
        ids, _ = sample_trajectory(
            model.model, TOTAL_LENGTH, tok.mask_id, tok.eos_id,
            steps=steps, temperature=temperature, top_p=top_p, seed=seed,
            prompt_ids=prompt_ids, device=model.device)
    if constraints:
        check = check_constraints(ids, constraints, tok.mask_id)
        guaranteed = guaranteed and check["required_present"] \
            and check["anchors_intact"]
    gen_ids = ids.tolist()[len(prompt_ids):]
    return {"answer": tok.decode(gen_ids),
            "facts": [f["text"] for f in facts],
            "sources": [f["source"] for f in facts],
            "key_terms": terms,
            "guaranteed": guaranteed}
