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

## Reward design — the Honesty-Race rule (Forge RL constraint)

When the Forge's verified successes become a training signal (Phase 4, verifiable-reward RL), the reward rule is itself a failure mode: a model paid for an unverifiable verdict ("this program is correct") can learn to *bluff* — to assert correctness the checker cannot confirm — instead of learning to produce checkable certificates. Bottman & Oblomkov, "Honesty Is a Race" (arXiv:2611.15779, Nov 11 2026), derive the first quantitative rule for this regime:

- A GRPO step makes the model more honest exactly when **p̃·r/b > 1+κ**, where p̃ = rate the checker accepts the model's certificates, r = payoff for a checker-accepted certificate, b = payoff for an unverifiable verdict, κ = pull of items where the unverifiable verdict is right.
- Below the threshold, runs collapse to bluffing. Just above it, runs first dip toward bluffing (on code, nearly to zero) before climbing back. There is also a **ceiling**: at p·r/b = 3, both code models called about half of correct programs buggy — paying the honest verdict too much makes the model deny correct work.
- Training schedule matters: training generously under the checker and then cutting the payoff kept the model ≥67% honest across three seeds, where the same model trained from the start at that ratio fell to 4–15% honest.

**Forge adoption (design constraint, implementation later):**
1. Forge rewards pay **only for checker-accepted certificates**. An unverifiable verdict pays 0 — never a partial reward for asserting correctness the checker cannot confirm.
2. The certificate/verdict payoff ratio must be kept **above the honesty threshold** (p̃·r/b > 1+κ) and **below the ceiling** where honest verdicts get paid for wrong answers. The ratio is tracked in the audit log alongside each run's honesty rate.
3. Use the **generous-then-cut schedule**: train generously under the checker first, then cut the certificate payoff — do not start at the final ratio.

**Status label (honest):** authors' claims, small-scale, single-lab, unreplicated — propositional Datalog, two code models, abstention on grade-school math. Adopted here as a cheap spec constraint because the failure mode it guards against (Forge teaching itself to bluff) would silently kill the guarantee-native thesis; the rule does not need replication before being written into the spec, but it must be re-checked against replication before any RL implementation relies on it.

## Capability must precede reward — the sharpening rule (Forge RL constraint)

A second failure mode for Phase 4: expecting verifiable-reward RL to *create* capability the base model does not have. Bangera & Gupta, "RLVR Sharpens the Pencil, Not the Mind" (arXiv:2611.02574, Nov 2 2026), tracked OLMo-2-1B across all 13 RLVR checkpoints on GSM8K:

- First-try accuracy 58.0% → 61.3%, but pass@256 solution coverage 94.5% → 96.9% — the knowledge was already there; RLVR made the model reach for its best solution paths more often.
- Probes for planted reasoning mistakes held AUROC 0.798–0.840 across all checkpoints and error types (spread smaller than the confidence interval) — the model never built a sharper internal picture of what a reasoning mistake looks like.
- The only real internal change was at the final-answer token (early-vs-late probe AUROC 0.93), while the same probe on the actual reasoning steps landed near chance. RLVR makes the model lead with its best thinking; it does not change what that thinking is.

**Forge adoption (design constraint, implementation later):**
1. RLVR/Forge sharpens *selection among latent capabilities*; it does not create new capability. Forge Phase 4 (training on verified successes) must not be expected to conjure chess legality or proof-carrying generation from a base that lacks it — capability has to be present via training/data first.
2. Before running a Forge reward loop on a task family, measure the base's latent coverage (pass@k). If coverage is ~0, the loop will sharpen nothing — route that family to training/data, not to more reward iterations.
3. Direct consequence for the current generation gap: the 1B scores 0/5 generation at 20 evaluated checkpoints, so there is no latent coherent generation for a reward loop to sharpen. Forge/RLVR cannot fix that; the path is token scale + LR (see `research/generation_gap_diagnosis_2026-11-13.md`).

This corroborates the "CANNOT (yet)" above — the loop amplifies capability, it doesn't create it — and pairs with the Honesty-Race rule: together they say *what* the reward loop can select for (honest, certificate-carrying behaviour the model already approximates) and *what it cannot* (capability the model does not have).

**Status label (honest):** single 1B model, single task (GSM8K), workshop paper (COLM 2026, MOSS). An independent re-run (Novel Cognition, Nov 10) found the headline overstates: the 13 snapshots come from a *second* RLVR round — the gain over the pre-RLVR baseline (51.8%) was made in the first round — and the pass@256 rise is ~3 problems out of 128. The probe-flatness finding stands; the headline does not fully. Adopted as a cheap spec constraint for the same reason as the Honesty-Race rule; re-check against replication before any RL implementation relies on it.

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
