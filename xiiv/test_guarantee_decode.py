"""Tests for xiiv/guarantee_decode.py (GCDD: LAVE-style lookahead-then-verify).

Run:  cd ~/workspace/ai-tools/corsac && python3 -m xiiv.test_guarantee_decode
Exit code 0 = all pass. No model weights needed; pure CPU.
"""
import itertools
import json
import time

import torch

from .guarantee_decode import (
    WILDCARD,
    GuaranteeLookahead,
    JsonViabilityChecker,
    _REP,
)
from .diffusion import DenoisingState

W = WILDCARD
PASS = []
FAIL = []


def check(name, cond):
    (PASS if cond else FAIL).append(name)
    print(("PASS " if cond else "FAIL ") + name)


def main():
    t0 = time.time()
    c = JsonViabilityChecker()
    warm = time.time() - t0  # noqa: F841 (construction is trivial)

    # ---- positive: valid documents -------------------------------------
    for doc in ['{"a": 1, "b": [true, null, "x"]}', '{"a":1}', "{}", "[]",
                "[1,2]", "5", "true", '"s"', '{"n": -12.5e+3}',
                '{"s": "a\\"b"}', '{"u": "\\u0041"}', "  { }  \n",
                '{"a": {"b": [{"c": []}]}}']:
        check(f"valid: {doc[:28]}", c.viable(doc) is True)

    # ---- positive: viable prefixes (no wildcard) ------------------------
    for pre in ['{"a": 1', "[1, ", '{"a":', '{"a"', "[",
                '"unclosed', '{"a": tru', '{"n": -1']:
        check(f"viable-prefix: {pre[:24]}", c.viable(pre) is True)

    # ---- positive: wildcards -------------------------------------------
    check("wild-only", c.viable(W) is True)
    check("wild-then-close", c.viable(W + "}") is True)   # W -> "{"
    check("open-then-wild", c.viable("{" + W) is True)
    check("obj-wild-close", c.viable('{"a": ' + W + "}") is True)
    check("key-wild", c.viable('{"a" ' + W + "}") is True)  # W -> ": 1,"
    check("wild-mid", c.viable('{"a": ' + W) is True)
    check("arr-wild-close", c.viable("[" + W + "]") is True)
    check("lit-wild", c.viable('{"a": tru' + W) is True)   # W -> 'e"}'

    # ---- negative: concrete contradictions (sound rejection) -----------
    for bad in ['{"a" "b"}', "}", "]", "[1 2]", '{"a": 01}', '{"k": tru}',
                "5 5", '{"a": }', '{"a": 1,}', "[1,]", "[,]", "}{",
                '{"a": truX}']:
        if bad == '{"a"':  # '{"a"' is a viable prefix, not a contradiction
            continue
        check(f"reject: {bad[:24]}", c.viable(bad) is False)
    check("reject wild-after-dead", c.viable("}" + W) is False)
    check("reject trailing junk", c.viable('{"a": 1} x') is False)

    # ---- brute-force soundness cross-check ------------------------------
    # For canvases where viable() says False, confirm by exhaustive search
    # (short replacements) that json.loads really cannot succeed. A brute
    # force HIT on a viable()==False canvas would be a soundness bug.
    def brute_no_completion(canvas, alphabet='{}[]",: 01', max_len=2):
        parts = canvas.split(W)
        n_wild = len(parts) - 1
        strs = [""]
        for leng in range(1, max_len + 1):
            strs.extend("".join(p)
                        for p in itertools.product(alphabet, repeat=leng))
        for combo in itertools.product(strs, repeat=n_wild):
            filled = parts[0]
            for p, r in zip(parts[1:], combo):
                filled += r + p
            try:
                json.loads(filled)
                return False
            except Exception:
                pass
        return True

    for bad in ['{"a" "b"}', '{"a": 01}', "[1 2]", "}" + W, '{"k": tru}',
                '{"a": }']:
        check(f"brute-sound: {bad[:20]}",
              c.viable(bad) is False and brute_no_completion(bad))

    # ---- truncation abstains (never False on deep nesting) -------------
    deep = "[" * 40 + W
    t1 = time.time()
    r = c.viable(deep)
    dt = time.time() - t1
    check("deep-nesting abstains True", r is True)
    print(f"    [info] deep-nesting check took {dt:.2f}s")

    # ---- representative alphabet sanity --------------------------------
    check("_REP non-empty", len(_REP) > 0)
    for must in '{}[],:"-01.e':
        check(f"_REP has {must!r}", must in _REP)

    # ---- GuaranteeLookahead: veto_reason --------------------------------
    # stub tokenizer: ids -> single chars
    TOK = {0: W, 1: "{", 2: "}", 3: '"', 4: "a", 5: ":", 6: "1",
           7: ",", 8: "[", 9: "]", 10: " ", 11: "x"}

    def token_text(tid):
        return TOK[int(tid)]

    gl = GuaranteeLookahead(json_checker=JsonViabilityChecker(),
                            token_text=token_text, mask_id=0)
    st = DenoisingState(length=6, mask_id=0, eos_id=99, steps=4, seed=7)
    st.x = torch.tensor([1, 3, 4, 3, 0, 0])  # '{"a"' + 2 masks
    check("veto quote-after-key",
          gl.veto_reason(st, 4, 3) == "json_unviable")   # '{"a"" dead
    check("allow colon-after-key",
          gl.veto_reason(st, 4, 5) is None)              # '{"a":.. ok
    check("veto close-after-key",
          gl.veto_reason(st, 4, 2) == "json_unviable")   # '{"a"} dead
    check("json_viable_placement True",
          gl.json_viable_placement(st, 4, 5) is True)
    check("json_viable_placement False",
          gl.json_viable_placement(st, 4, 3) is False)

    # ---- contract half of the veto --------------------------------------
    st2 = DenoisingState(length=2, mask_id=0, eos_id=99, steps=2, seed=1,
                         constraints={"required_ids": [6, 1]})
    # x=[5,0]: placing ':' (5) at 0 leaves 1 mask but 2 required missing
    st2.x = torch.tensor([5, 0])
    check("veto contract_unsatisfiable",
          gl.veto_reason(st2, 0, 5) == "contract_unsatisfiable")

    # ---- try_commit: resample + defer -----------------------------------
    st3 = DenoisingState(length=6, mask_id=0, eos_id=99, steps=4, seed=42)
    st3.x = torch.tensor([1, 3, 4, 3, 0, 0])
    gl2 = GuaranteeLookahead(json_checker=JsonViabilityChecker(),
                             token_text=token_text, mask_id=0,
                             max_veto_attempts=2)
    p = torch.zeros(12)
    p[3] = 1.0  # only '"' — always vetoed here
    n_int_before = len(st3.interventions)
    got = gl2.try_commit(st3, 4, p)
    check("try_commit defers when all vetoed", got is None)
    check("vetoes logged to interventions",
          len(st3.interventions) == n_int_before + 1
          and st3.interventions[-1]["kind"] == "guarantee_veto")

    # viable candidates commit deterministically
    def fresh_state():
        s = DenoisingState(length=6, mask_id=0, eos_id=99, steps=4,
                           seed=123)
        s.x = torch.tensor([1, 3, 4, 3, 0, 0])
        return s

    p2 = torch.zeros(12)
    p2[5] = 0.5
    p2[6] = 0.5  # ':' or '1' — both JSON-viable here
    a = gl2.try_commit(fresh_state(), 4, p2)
    b = gl2.try_commit(fresh_state(), 4, p2)
    check("try_commit deterministic", a == b and a in (5, 6))

    # ---- constructor validation -----------------------------------------
    try:
        GuaranteeLookahead(json_checker=JsonViabilityChecker(),
                           token_text=None, mask_id=0)
        check("token_text required", False)
    except ValueError:
        check("token_text required", True)
    try:
        JsonViabilityChecker(max_depth=0)
        check("max_depth validated", False)
    except ValueError:
        check("max_depth validated", True)

    # ---- end-to-end: decode loop with guarantee -------------------------
    # Pathological model: always proposes '}' (id 2). With GCDD on, every
    # such commit after '{"a"' is vetoed (sound rejection), so the final
    # canvas must still be JSON-viable. With GCDD off, it is not.
    from .diffusion import sample_trajectory_stateful

    class AlwaysClose(torch.nn.Module):
        def forward(self, x, **kw):
            B, L = x.shape
            logits = torch.full((B, L, 12), -1e9)
            logits[:, :, 2] = 0.0
            return logits

    def run_e2e(guarantee, seed):
        gl_e2e = guarantee
        ids, conf, stx = sample_trajectory_stateful(
            AlwaysClose(), 8, 0, 99, steps=4, temperature=0.1, seed=seed,
            prompt_ids=torch.tensor([1, 3, 4, 3]),  # '{"a"'
            frozen=torch.tensor([True] * 4 + [False] * 4),
            guarantee=gl_e2e)
        return ids, stx

    gl3 = GuaranteeLookahead(json_checker=JsonViabilityChecker(),
                             token_text=token_text, mask_id=0,
                             max_veto_attempts=2)
    ids_g, st_g = run_e2e(gl3, 5)
    gl3b = GuaranteeLookahead(json_checker=JsonViabilityChecker(),
                              token_text=token_text, mask_id=0,
                              max_veto_attempts=2)
    ids_g2, _ = run_e2e(gl3b, 5)

    def canvas_text(ids):
        return "".join(W if int(t) == 0 else TOK[int(t)]
                       for t in ids.tolist())

    check("e2e deterministic with guarantee",
          bool((ids_g == ids_g2).all()))
    check("e2e final canvas JSON-viable",
          JsonViabilityChecker().viable(canvas_text(ids_g)) is True)
    kinds = {iv["kind"] for iv in st_g.interventions}
    check("e2e vetoes logged", "guarantee_veto" in kinds)
    check("e2e deferrals logged", "guarantee_defer" in kinds)

    ids_plain, _ = run_e2e(None, 5)
    check("e2e control: no-guarantee canvas broken",
          JsonViabilityChecker().viable(canvas_text(ids_plain)) is False)
    print(f"    [info] with-GCDD canvas:    {canvas_text(ids_g)!r}")
    print(f"    [info] without-GCDD canvas: {canvas_text(ids_plain)!r}")

    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        print("FAILURES:", FAIL)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
