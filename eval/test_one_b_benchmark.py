"""Tests for eval/one_b_benchmark.py — pure helpers only, no model load."""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.one_b_benchmark import (  # noqa: E402
    CATEGORIES,
    GATE_THRESHOLD,
    check_prompt,
    load_prompts,
    score_run,
    unique_out_path,
    word_repeat_ratio,
)


def test_prompt_set_is_100_and_10_per_category():
    prompts = load_prompts()
    assert len(prompts) == 100
    for cat in CATEGORIES:
        assert sum(1 for p in prompts if p["category"] == cat) == 10


def test_prompt_ids_unique_and_prompts_short():
    prompts = load_prompts()
    ids = [p["id"] for p in prompts]
    assert len(set(ids)) == len(ids)
    # Short by construction: every prompt is under 25 whitespace words,
    # so it cannot fill the 64-token canvas before generation starts.
    assert all(len(p["prompt"].split()) <= 25 for p in prompts)


def test_gate_threshold_is_9():
    assert GATE_THRESHOLD == 9


def test_checks_run_on_generated_text_only():
    # A keyword present in the prompt but absent from the generation
    # must FAIL — scoring the full text would pass this trivially.
    passed, _ = check_prompt(
        {"checks": [{"type": "contains", "value": "Paris"}]},
        "some other words here")
    assert not passed
    passed, _ = check_prompt(
        {"checks": [{"type": "contains", "value": "Paris"}]},
        "the answer is Paris")
    assert passed


def test_repetition_screen():
    assert word_repeat_ratio("hello hello hello hello hello hello hello "
                             "hello hello hello hello") > 0.5
    assert word_repeat_ratio("the cat sat on the mat by the sea") < 0.5


def _passing_gen(item):
    """Build generated text satisfying this item's own checks.

    Different items legitimately contradict each other (one requires
    'cold', another forbids it), so a single canned text cannot pass
    the whole set — the fixture must be per-item, like a real run.
    """
    words = ["alpha", "bravo", "charlie", "delta", "echo",
             "foxtrot", "golf", "hotel", "india", "juliet"]
    for check in item["checks"]:
        if check["type"] == "contains":
            words.append(check["value"])
    return " ".join(words)


def test_gate_fails_if_any_category_below_9():
    prompts = load_prompts()
    items = []
    for p in prompts:
        items.append({"id": p["id"], "category": p["category"],
                      "checks": p["checks"],
                      "generated_text": _passing_gen(p),
                      "generation_seconds": 1.0, "n_tokens_non_mask": 10})
    # Break one category completely: all its items fail the contains check.
    for it in items:
        if it["category"] == "arithmetic_reasoning":
            it["generated_text"] = "no numbers here at all"
    report = score_run({"items": items})
    assert report["gate"] == "FAIL"
    assert "arithmetic_reasoning" in report["gate_failures"]
    assert report["categories"]["factual_explanation"]["status"] == "PASS"


def test_gate_pass_requires_all_categories():
    prompts = load_prompts()
    items = []
    for p in prompts:
        items.append({"id": p["id"], "category": p["category"],
                      "checks": p["checks"],
                      "generated_text": _passing_gen(p),
                      "generation_seconds": 1.0, "n_tokens_non_mask": 10})
    report = score_run({"items": items})
    assert report["gate"] == "PASS"
    assert report["gate_failures"] == []


def test_incomplete_run_is_not_a_pass():
    report = score_run({"items": [{
        "id": "x", "category": "factual_explanation", "checks": [],
        "generated_text": "hi", "generation_seconds": 1.0,
        "n_tokens_non_mask": 1}]})
    assert report["gate"] == "FAIL"
    assert report["categories"]["factual_explanation"]["status"] == "INCOMPLETE"


def test_competitors_all_unmeasured():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "one_b_competitors.json")
    with open(path) as f:
        data = json.load(f)
    assert data["xiiv"]["scores"] == "UNMEASURED"
    assert len(data["competitors"]) >= 5
    for comp in data["competitors"]:
        assert comp["scores"] == "UNMEASURED", comp["model"]
        assert comp["hf_repo"] and comp["model"]


def test_output_path_never_overwrites(tmp_path, monkeypatch):
    import eval.one_b_benchmark as mod
    monkeypatch.setattr(mod, "RESULTS_DIR", str(tmp_path))
    p1 = unique_out_path("xiiv_1b", "cuda")
    open(p1, "w").write("{}")
    p2 = unique_out_path("xiiv_1b", "cuda")
    assert p1 != p2
