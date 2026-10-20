# Pre-verdict fact-check: beyond-proof-machine deep hunt
Date: 2026-10-16 ~05:50Z (11:20 IST) · For the commissioned 23:00 IST verdict. NOT for standalone delivery.

The interim report (research/beyond_proof_machine.md) made three load-bearing factual claims.
I checked each against live sources. Two need correction, one needs reframing.

## 1. Interaction Nets — claim: "zero ML prior art, verified clean"
**Mostly true, but imprecise in a way that matters.**
- True: nobody has published a *learnable* interaction-net neural architecture (learned rewrite rules / differentiable interaction nets). That specific claim appears clean.
- Missing context the verdict MUST include: the execution substrate already exists and is mature. HVM2 (HigherOrderCO, Victor Taelin) is a massively parallel evaluator for interaction combinators: 74 billion interactions/sec on an RTX 4090, near-ideal scaling to 16k+ cores; the Bend language compiles to it (github.com/HigherOrderCO/HVM3, HVM2 paper PDF). Lafont 1990/1997 originals; Lamping optimal reduction is the theory behind it.
- Implications: (a) novelty statement must be "no learned interaction-net architecture," not "zero prior art"; (b) GOOD NEWS for the kill experiment — prototype the *learning* layer on top of HVM/Bend instead of building a runtime from scratch; cheapest-to-falsify just got cheaper; (c) watch HigherOrderCO — they own the adjacent execution layer and could move up the stack.
- NAMING COLLISION the verdict must disambiguate explicitly: "Interaction Networks" = Battaglia et al., NIPS 2016 (a GNN variant, hugely cited) is NOT Lafont's interaction nets (1990 graph rewriting / interaction combinators). Anyone searching the term lands on Battaglia first; the novelty claim must always read "no learned architecture on Lafont interaction nets," never "no prior art on interaction networks," or it will be read as a claim about a crowded GNN field and instantly discredited.

## 2. Ephaptic Field Computing — claim: "essentially zero AI prior art"
**False as stated. Downgrade to "nascent fringe, no mainstream adoption."**
- Prior art found: (i) ephapsys.com paper draft (June 2025, rev 0.9.6) proposing ephaptic-inspired activation for ANNs — claims 74.65% perplexity reduction on GPT-2/WikiText-103, +3.08% on ResNet-18/ImageNet-100, +6.67% PPO reward, at ~0.5–2.2% parameter overhead. SELF-PUBLISHED, unreviewed — treat numbers as unverified. (ii) github.com/anttiluode/ephapticfieldintelligence (2025) — "instanton" field simulations, hobbyist. (iii) Zenodo preprints on ephaptic fields as binding substrates.
- Biology side CONFIRMED: Pinotsis et al., Cerebral Cortex (2026) — ephaptic coupling explains trial-by-trial neural variability, Granger causality shows fields dominate neural activity; Miller/Pinotsis "cytoelectric coupling" hypothesis. The 2026 neuroscience tailwind is real.
- Honest verdict framing: the O(n) field-replaces-attention idea has fringe predecessors with unverified claims; the differentiator would be a rigorous, falsifiable architecture + real benchmarks, not the bare idea.

## 3. Immune Generalization — claim: "Aug 2026 Science Advances reframed why artificial immune systems failed — missing cross-reactivity"
**Paper confirmed; the "reframed AIS failure" narrative is invented. Drop it.**
- Confirmed: Meyer, Navlakha (CSHL), Science Advances, Aug 2026 ("Does your immune system learn like AI?", via medicalxpress/earth.com). Findings: 90% of self-reactive T cells correctly deleted in the thymus despite each encountering only ~10% of the body's self-peptides; two enabling conditions — (1) thymic self-peptide abundance mirrors tissue abundance (train≈test), (2) TCR cross-reactivity (one receptor recognizes several similar peptides).
- What the paper does NOT say: anything about artificial immune systems failing. It is "ImmunoAI" — studying how the immune system solves ML problems. The AIS-failure story was fabricated in the interim relay.
- Honest verdict framing: biological negative-selection-as-generalization is real and grounded — a safety paradigm of "model self, delete what matches" has a stronger biological existence proof than the interim report suggested; but it must be presented as bio-inspired, not as a correction to AIS history.

## 4. Meta-find ("learning, inference, memory, safety as the same operation — nobody has it")
Not fact-checkable; treat as direction, not finding. The verdict should not present it as a discovery.

## Recommendation for the verdict
Lead with corrected novelty statements. Interaction nets remains the strongest bet: cheapest to falsify AND now has a free runtime (HVM/Bend) for the prototype. Ephaptic downgraded on novelty, biology tailwind real. Immune needs the bio-grounded reframe. Do not repeat the interim report's unverified claims — Fahad already caught the rushed relay once.
