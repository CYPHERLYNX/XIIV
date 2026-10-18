# XIIV Self-Improvement Loop — "The Forge"

**Directive from Fahad (2026-10-11):** Add a self-learning approach alongside the guarantee-native work: the model generates (e.g., code), runs it, checks the result against expectations, and if it doesn't get 100% of what it expects, rewrites until it does. This pushes XIIV toward true ASI capability.

**Status:** Implemented in code — `xiiv/forge.py` + `xiiv/forge_agent.py` (ForgeTask, ForgeResult, SkillLibrary, ForgeAgent, ThinkTrace; imports verified clean 2026-10-13). End-to-end execution against the independent verifier pending.

---

## The core loop

```
1. TASK + CONTRACT    — a task with a machine-checkable success criterion
2. GENERATE           — XIIV produces a candidate solution under the contract
3. EXECUTE            — the solution runs in a sandbox (code exec, checker, etc.)
4. VERIFY             — INDEPENDENT verifier checks output against the contract
5. FEEDBACK           — on failure: structured failure report (which clause failed,
                        test output, error trace) fed back to the model
6. REWRITE            — model regenerates with failure context
7. REPEAT             — until 100% verification pass or max iterations
8. LEARN             — successful (task, solution, failure-history) stored in
                        a skill library for future retrieval
```

## Why this is uniquely powerful (vs existing work)

Existing self-improvement loops have a circularity problem:

| Approach | Judge | Problem |
|---|---|---|
| Reflexion / Self-Refine | The model critiques itself | Circular — a weak model can't reliably judge itself |
| Self-Debugging | Test output, model interprets | Better, but the model still decides what "fixed" means |
| Voyager | Game environment | Real grounding, but Minecraft-specific |
| **XIIV Forge** | **Independent contract verifier** | **Deterministic, non-circular, proof-carrying** |

The verifier is the guarantee-native contract checker — separate code, no model weights, deterministic. The model cannot grade its own homework. When the loop terminates with 100%, that 100% is independently established, not self-reported. The final output carries its certificate: what was tried, what failed, what passed.

This is the composition that makes it unique: **self-improvement loop + guarantee-native verification**. Nobody else has both halves.

## What it can and cannot do (honest)

**CAN:**
- Dramatically improve reliability on verifiable tasks (code, math, structured generation, any task with a checkable contract). Existing literature (Self-Debugging, Reflexion) shows large gains on such tasks from iteration alone.
- Accumulate a skill library: solved tasks become retrievable experience, so the system gets better at task families over time without retraining.
- Produce auditable improvement traces: every iteration logged, every success certified.

**CANNOT (yet):**
- Bootstrap general intelligence by itself. This loop improves *task success*, not the model's underlying capability. A weak model iterating 100 times on a task beyond its reach still fails — the loop amplifies capability, it doesn't create it.
- True recursive self-improvement (rewriting its own weights/architecture — the hypothesized "intelligence explosion" mechanism). That's a separate, much harder research problem nobody has demonstrated. The Forge operates at the output level, not the weight level.

**The honest ASI claim:** the Forge is task-level recursive self-improvement with independent verification. Combined with a strong base model, it produces a system that reliably gets better at verifiable work on its own — a genuine step toward the agentic+verified ASI path, not the full thing.

## Safety requirements (non-negotiable)

1. **Sandboxed execution.** The EXECUTE step runs in an isolated sandbox with no network, no filesystem writes outside a scratch dir, CPU/memory/time limits. A self-rewriting code loop without a sandbox is a hazard, not a feature.
2. **Iteration cap.** Max iterations per task (default 10). Prevents infinite rewrite loops and bounds compute.
3. **Verifier independence.** The verifier must never be modified by the loop. The model can rewrite the solution; it can never rewrite the test.
4. **No weight self-modification.** The loop improves outputs and accumulates skills. It does not touch model weights. Weight-level self-modification is a separate research track requiring its own safety design.
5. **Audit log.** Every iteration (input, output, verifier verdict) is logged immutably. If something goes wrong, the full trace exists.

## Implementation plan

**Phase 1 — Harness (no model changes):** DONE 2026-10-11
- `xiiv/forge.py`: task definition format, sandbox executor, verifier interface, feedback compiler, skill library (SQLite). Self-tested: fail → feedback → fix → verify → store.
- Verifier interface matches the guarantee-benchmark verifier (shared contract definitions).

**Phase 1b — Agentic layer (Jarvis/Ultron upgrade, Fahad 2026-10-11):** DONE 2026-10-11
- `xiiv/forge_agent.py`: the agent THINKS (explicit reasoning trace), DECOMPOSES goals into verifiable sub-tasks, ACTS via the Forge loop, REFLECTS after each sub-task, CORRECTS the plan itself when the approach is wrong (not just the code), and REMEMBERS full episodes in persistent memory.
- Self-tested: planned 2 sub-tasks, first failed once → self-corrected → verified, reflected, goal achieved. Every thought logged and auditable.
- Honest scope: thinking quality is bounded by the base model. The loop amplifies intelligence; it doesn't create it. Not sentient — a structured, verifiable reasoning loop.

**Phase 2 — Contract task suite:**
- Seed tasks: code generation with unit tests, structured data generation with schema checks, math with symbolic verification.
- Each task = (prompt, contract, verifier, max_iterations).

**Phase 3 — Skill library + retrieval:**
- Store successful solutions with embeddings; retrieve relevant past solutions as context for new tasks.
- Measure: does retrieval reduce iterations-to-success on related tasks?

**Phase 4 — Integration with training:**
- High-quality (task, final-solution) pairs from the Forge become training data — the model learns from its own verified successes. This is the closest honest version of "learning by itself": verified self-generated data improving the base model.

## Metrics

- **Success rate:** % of tasks reaching 100% verification within max iterations.
- **Iterations to success:** distribution; lower = stronger.
- **Skill transfer:** iterations-to-success on novel tasks with vs without skill library retrieval.
- **Certificate rate:** % of successes carrying a complete auditable trace (target: 100%).

## Relationship to existing XIIV work

- Guarantee Benchmarks 1–3 define the verifier the Forge uses.
- Benchmark 2 (Intervention Rate) is the single-generation version; the Forge extends intervention across generations.
- The 1B training plan gains a Phase 4 consumer: Forge-generated verified pairs as training data.
