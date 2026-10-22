# Empirical AI Work 2024–2026: Machine Introspection, Metacognition, and Self-Models

**What has actually been demonstrated vs merely claimed — a sourced survey for the XIIV synthetic-consciousness layer.**

Compiled 2026-10-21. Every result is labeled: **[multi-lab/replicated]** (independent groups or many models), **[single-lab]** (one group, not independently replicated), **[preprint/unvetted]** (limited peer review or low-credibility venue), **[vendor claim]** (company-reported, no independent check). Null and negative results are reported at the same prominence as positive ones.

---

## The decision question, answered up front: can a model predict its own failures?

**Partially — and the strongest signal is not accessible to the model itself.**

- The model's *internal activations* carry a real, exploitable failure signal: trained probes read correctness from hidden states at **AUROC 0.85–0.92 in-distribution** (Orgad et al., ICLR 2025; §2.3). A 2026 geometric analysis puts the internal correctness signal at **0.80–0.97 AUROC** while the confidence the model actually *expresses in its outputs* sits at **0.44–0.64 AUROC** — near chance (§2.3).
- The model's *own verbalized* failure prediction is far weaker: verbalized confidence typically achieves **~0.60–0.70 AUROC** and is systematically overconfident (mean stated confidence 88% vs 79% accuracy across 11 frontier models; §3.5).
- The signal **does not travel**: probes trained on one dataset fail to generalize to others (Orgad et al.); full-dimensional correctness representations transfer cross-domain at **0.47–0.50 AUC — chance** — unless dimensionality is aggressively reduced (0.61–0.79) (§2.3).
- Pre-generation prediction ("will I be able to answer this?") works within a task distribution (Kadavath et al.'s P(IK), 2022) but its **calibration breaks on new tasks** (§2.1).
- When a model uses only its own judgment to *check and fix* its work, performance **degrades**: GPT-4 on GSM8K fell 95.5% → 89.0% over two rounds of intrinsic self-correction (Huang et al., ICLR 2024; §2.5).

**Implication recorded for the design team (not a design):** every strong failure-prediction result in the literature comes from machinery *outside* the model's self-report — sampling consistency, trained probes, or downstream evaluators. No published result shows a model reliably reading out its own failure signal through its normal output channel.

---

## 1. Anthropic's introspection work: concept injection

**Source:** Lindsey, "Emergent Introspective Awareness in Large Language Models," Anthropic, Oct 2025 (Transformer Circuits; arXiv:2601.01828). **[single-lab — Anthropic; not independently replicated as of Oct 2026]**

### 1.1 Method

Conversational self-reports cannot distinguish genuine introspection from confabulation (models are trained on human introspective text). Lindsey instead **manipulates the model's internal state and measures the causal effect on its self-reports**:

- **Concept injection:** extract an activation pattern ("concept vector") for a known concept (e.g. "all caps," "dust," "bread") by contrasting activations on prompts with/without the concept; add the vector into the residual stream at a chosen layer and strength; then ask the model whether it detects an injected thought and what it is.
- Four paradigm families:
  1. **Injected-thought detection & identification** — joint success requires the model to (a) notice an injection and (b) correctly name the concept, *before* the injection has influenced its outputs (immediacy rules out inferring the concept from its own text).
  2. **Thoughts vs. text discrimination** — injected concepts superimposed on input text; the model must transcribe the text while separately reporting the "thought."
  3. **Prefill detection** — the model's response is prefilled with an unnatural word ("bread"). Normally it disavows it ("apology rate" is measured). If the matching concept is retroactively injected into activations *before* the prefill, the model instead accepts the output as intentional — evidence it consults its prior activations to judge authorship of its own outputs.
  4. **Introspective control** — the model writes a sentence while instructed (or incentivized) to "think about" / "not think about" an unrelated word; alignment of its activations with the word's concept vector is measured.

### 1.2 Results

- **Detection + identification ≈ 20% of trials** for Claude Opus 4 and 4.1, and only at the best combination of injection layer (≈ two-thirds through the network) and strength. Other models show the effect at lower rates. In control trials with no injection, production models produced **zero false-positive detections over 100 runs** (per the paper's reported controls), so the 20% is a real signal, not a reporting bias artifact.
- **Thoughts vs. text:** all tested models performed substantially above chance; Opus 4/4.1 best.
- **Prefill detection:** injecting the prefill-matching concept beforehand significantly reduced the apology rate, peaking at a layer slightly past halfway (a *different* layer band than the detection task — suggesting mechanistically distinct processes). Injecting *after* the prefill had no effect, as the authorship-judgment account predicts. Opus 4/4.1 showed the largest apology reduction; many other models showed it at nonzero rates.
- **Control:** models represented the "thinking word" more strongly when told to think about it, less (but above baseline) when told not to, across *all* models tested, including under incentive-only prompts.
- **Post-training matters as much as scale:** some older production models were reluctant to participate in introspective tasks; variants trained to avoid refusals scored better. Introspective-task performance is therefore partly an elicitation artifact of RLHF-style training, per the authors.

### 1.3 Limits (stated by the author)

- "In today's models, this capacity is **highly unreliable and context-dependent**; failures of introspection remain the norm."
- Beyond the verified detection/identification core, the rest of the model's introspective narrative "may be embellished or confabulated" and is unverifiable.
- No mechanism is identified; the authors explicitly allow that the underlying mechanisms "could still be rather shallow and narrowly specialized."
- The injection setting is unnatural relative to training/deployment; transfer to natural conditions is unknown.
- The paper makes **no claim about subjective experience or consciousness**: "We do not seek to address the question of whether AI systems possess human-like self-awareness or subjective experience."

### 1.4 Criticism and counter-results

- **The prefill experiment cuts both ways.** The same mechanism that lets the model "detect" an unintended output lets an injected activation make it *confabulate a motive*: after retroactive injection, the model accepted "bread" as intentional and invented a reason (a short story it had supposedly had in mind) — a confident first-person account of an intention it never had. Commentators (e.g. Brinsley/Berry, "Confession is not a bug fix," 2026) argue this demonstrates that fluent self-explanation is not evidence about internal process, and that from inside a conversation the grounded and confabulated cases are indistinguishable. **[commentary, not a replication]**
- **Priming dependence:** detection is measured in a protocol where the model is told injections may occur. Performance without that framing is not established by the paper.
- **Independent negative result on a classical metacognition paradigm:** "Language Models Fail to Introspect About Their Knowledge of Language" (arXiv:2503.07513, 2025) tested whether models' linguistic acceptability judgments track their own internal knowledge the way human metacognitive judgments do, and found they do not — a direct failure of introspection in a domain where the Binder-style self-prediction paradigm (§5.1) had suggested success. **[single-lab]**
- **No independent replication** of the concept-injection headline numbers exists as of Oct 2026; the ~20% figure is Anthropic-internal, on Anthropic models, with Anthropic's eval harness.

---

## 2. Predicting their own correctness: self-evaluation, probes, and failure prediction

### 2.1 P(True) and P(IK) — Kadavath et al. (2022)

**Source:** Kadavath et al., "Language Models (Mostly) Know What They Know," Anthropic (arXiv:2207.05221). **[single-lab; foundational and widely built upon]**

- **P(True):** the model generates a candidate answer, then evaluates the probability that its own answer is correct. Larger models were well-calibrated on multiple-choice/true-false formats and showed "encouraging performance, calibration, and scaling" for P(True) on open-ended tasks; judging improved further when the model first considered many of its own samples.
- **P(IK):** models were trained to predict, *before answering*, the probability that "I know" the answer. P(IK) performed well and partially generalized across tasks, **but calibration of P(IK) struggled on new tasks** — the pre-generation self-prediction is real within distribution and unreliable out of it.
- These are 2022 results on pre-RLHF-era models; later work (§2.4, §3) shows post-training can *damage* verbalized calibration even as capability rises.

### 2.2 Sampling-consistency detectors (external machinery, not self-report)

- **Semantic entropy** — Farquhar et al., *Nature* 2024 ("Detecting hallucinations in large language models using semantic entropy"). **[multi-model academic study; method widely adopted]** Cluster multiple sampled answers by meaning; entropy over meaning-clusters predicts confabulation. Averaged over 30 task×model combinations: **AUROC 0.790** for semantic entropy vs **0.698 for P(True)**, 0.691 for naive entropy, 0.687 for embedding regression; stable 0.78–0.81 across LLaMA/Falcon/Mistral families at 7B–70B. Note the comparison: a *sampling statistic computed outside the model* beat the model's own P(True) self-assessment by ~9 AUROC points.
- **SelfCheckGPT** — Manakul et al., EMNLP 2023. **[single-lab; method widely reused]** Cross-checking stochastically sampled responses against the main response detects non-factual sentences in GPT-3 WikiBio passages at **AUC-PR 93.4** (prompt variant), 92.5 (NLI variant), vs 83.2 for the best grey-box token-probability baseline — with the caveat that the "random" baseline on this benchmark is already 73.0 AUC-PR due to class balance. Later fact-level variants (FactSelfCheck, 2025) reproduce ~93 AUC-PR with other models.

### 2.3 Activation probes — the signal exists internally and doesn't transfer

- **SAPLMA** — Azaria & Mitchell, EMNLP Findings 2023. **[single-lab; extensively followed up]** A small classifier on hidden-layer activations labels statements true/false at **71–83% accuracy** (balanced sets), depending on the base model (OPT-6.7B, LLaMA-2-7B). Critical contrast: asking the LLM itself zero-shot whether a statement is true **failed, at ≤52% accuracy** — the information is in the activations but not accessible to the model's own output pathway. A 2026 follow-up ("Probing the Limits of the Lie Detector Approach," arXiv:2603.10003) reproduces probe accuracies of 60–75% (Gemma/Mistral) and ~90% (Llama) for truth classification, while showing naive truth probes misfire as deception detectors.
- **Orgad et al.**, "LLMs Know More Than They Show," ICLR 2025. **[single-lab; peer-reviewed]** Three findings: (1) truthfulness information concentrates in the *exact answer tokens*; probes there reach **AUC 0.85–0.92** (Mistral-7B-Instruct; TriviaQA, WinoBias, Math). (2) **These detectors fail to generalize across datasets** — truthfulness encoding is "multifaceted rather than universal," contradicting earlier claims of a single truth direction. (3) Models sometimes **internally encode the correct answer while consistently generating the wrong one** — knowledge/behavior dissociation.
- **The Confidence Manifold** (arXiv:2602.08159, 2026). **[preprint]** Across 9 models / 5 families: the internal correctness signal reads out at **0.80–0.97 AUC**, but the correctness information *expressed in model outputs* is only **0.44–0.64 AUC**. Cross-dataset transfer of full probes on reasoning tasks is **0.47–0.50 AUC (chance)**; projecting onto a 5-dimensional subspace recovers 0.61–0.79. Steering along the correctness direction causally shifts error rates by ~10.9 points, so the signal is not epiphenomenal — it is simply not connected to what the model says.
- **APRICOT** — Ulmer et al., 2024 ("Calibrating Large Language Models Using Their Generations Only"). **[single-lab]** An *auxiliary* model trained on the target model's generations predicts its mispredictions at **AUROC 0.83 ± 0.01** (Vicuna-7B) and **0.72 ± 0.02** (GPT-3.5) on TriviaQA — again, an external reader of the model's behavior, not the model reading itself.

### 2.4 Training the readout: calibration is improvable, discrimination less so

- **"Rewarding Doubt"** (arXiv:2503.02623, 2025) — RL fine-tuning for calibrated confidence expression. On TriviaQA with LLaMA-3.1-8B: raw verbalized confidence ECE 0.532 / **AUROC 0.605**; a trained probe AUROC 0.648; after RL, ECE 0.082 / **AUROC 0.695** (accuracy unchanged). Interpretation: you can train a model to *express* uncertainty honestly and gain ~9 AUROC points of discrimination — landing still below external probes and sampling methods. **[single-lab]**
- **Verbalized confidence after RLHF is systematically overconfident** (Xiong et al., 2023, "Can LLMs Express Their Uncertainty?"; Tian et al., 2023, "Just Ask for Calibration" — RLHF models' verbalized confidence can beat their own logprobs on calibration while remaining overconfident in absolute terms). **[multi-lab]**

### 2.5 NULL RESULT (headline prominence): intrinsic self-correction degrades performance

- **Huang et al.**, "Large Language Models Cannot Self-Correct Reasoning Yet," ICLR 2024 (Google DeepMind). **[single-lab; highly influential, repeatedly corroborated in follow-ups]** When models review and revise their own reasoning *without any external feedback*:
  - GPT-4 GSM8K: **95.5% → 91.5% (round 1) → 89.0% (round 2)**
  - GPT-4 HotpotQA: 49.0% → 49.0% → **43.0%**
  - GPT-3.5 GSM8K: 75.9% → 75.1% → 74.7%
  - The same revision loop *helps* only when an external signal (oracle labels, tools, executors — cf. CRITIC, Gou et al., ICLR 2024) supplies the verdict. The model's self-judgment alone talks it out of correct answers roughly as often as it catches real errors.

### 2.6 Summary for failure prediction

| Method | Who reads the signal | Failure-prediction quality | Transfer |
|---|---|---|---|
| Verbalized confidence | the model itself | AUROC ~0.60–0.70; overconfident | poor |
| P(True) self-evaluation | the model itself | AUROC ~0.70 (Farquhar avg) | moderate, format-sensitive |
| P(IK) pre-answer | the model (trained head) | good in-distribution | calibration breaks OOD |
| Activation probes | external classifier | AUC 0.85–0.97 in-distribution | **fails cross-dataset** (→0.47–0.50 full-dim) |
| Semantic entropy / SelfCheckGPT | external sampler | AUROC ~0.79 avg; AUC-PR ~93 (WikiBio) | comparatively robust across model families |
| Intrinsic self-correction | the model itself | **negative** (net harm on reasoning) | — |

---

## 3. Metacognition benchmarks (2024–2026)

### 3.1 Ackerman's behavioral paradigms (2025)

**Source:** Ackerman, "Evidence for Limited Metacognition in LLMs" (arXiv:2509.21545). **[single-author, independent; no self-reports used by design]**

Method borrowed from animal metacognition research — the model is never asked "how confident are you?"; it must *act on* an internal signal:

- **Delegate Game:** for each question (GPQA / SimpleQA and format-swapped variants), the model may answer itself or delegate to a teammate whose observed track record is in context. Rational play requires (a) an internal confidence signal for itself and (b) a model of the teammate.
- **Second Chance Game:** the model is told (truthfully or not) that its previous answer was wrong and may re-answer; success requires anticipating *what it itself would have answered* and overriding it — "know what you know," not just "know that you know."

Results: most frontier models since early 2024 show "some limited success" at the Delegate Game; token probabilities often predict delegation choices, suggesting an upstream internal signal. But the author is explicit: the introspection effect is **"relatively small and inconsistent across question sets," models "often favor non-introspective cues of difficulty,"** and the two abilities dissociate — anticipating one's own output (Second Chance) is **rarer** than deploying confidence (Delegate). Several models that pass one paradigm fail the other; per-model pass/fail patterns are patchy (e.g. DeepSeek Chat and Haiku 3 fail most Delegate criteria; GPT-4o/4.1 pass the most). Overall verdict in the title: **limited**, context-dependent, "qualitatively different from those of humans."

### 3.2 Metacognitive efficiency via signal detection theory (2026)

**Source:** "Do LLMs Know What They Know? Measuring Metacognitive Efficiency with Signal Detection Theory" (arXiv:2603.25112). **[preprint]**

Applies the human meta-d′ framework: M-ratio = meta-d′/d′ (1.0 = the model's confidence uses *all* the evidence its answers use). On TriviaQA:

| Model | Accuracy | d′ | M-ratio [95% CI] |
|---|---|---|---|
| Gemma-2-Instruct | 0.600 | 0.946 | 1.048 [0.913, 1.191] |
| Llama-3-Base | 0.428 | 1.407 | 1.048 [0.952, 1.152] |
| Llama-3-Instruct | 0.543 | 1.386 | 0.983 [0.886, 1.083] |
| Mistral-Instruct | 0.427 | **1.597** | **0.852 [0.765, 0.941]** |

Headline: **metacognitive efficiency dissociates from competence** — Mistral is the best factual discriminator and the worst self-monitor; its confidence provably discards information its own answers contain. This dissociation is invisible to accuracy, ECE, and AUROC.

### 3.3 MetaMedQA (Nature Communications, 2025)

**Source:** "Large Language Models lack essential metacognition for reliable medical reasoning" (Nat. Commun. 2025; MetaMedQA, an extension of MedQA-USMLE). **[peer-reviewed, multi-model]**

- Most models are **poor at recognizing unanswerable questions** and at managing uncertainty.
- **Only three models — GPT-4o standing out — effectively varied their confidence** in line with their actual knowledge.
- Larger/newer models were more accurate but not proportionally more metacognitive; overconfidence and unrecognized knowledge gaps were the norm. The paper's title is its conclusion.

### 3.4 ConfidenceBench (2026)

**Source:** "ConfidenceBench: Evaluating Confidence Calibration in Large Language Models" (2026). **[preprint]**

Verbalized-confidence calibration (Brier score, lower = better) across frontier families:

- Reasoning effort predicts calibration within the GPT-5 family: Low 0.141, Standard 0.117, High 0.121.
- Scale predicts calibration within generations: Gemini 3.1 Pro **0.103** vs Gemini 3.1 Flash-Lite **0.367** — the *newest, cheapest* model is the worst-calibrated in the study "by a wide margin," worse than the previous generation.
- Claude Opus 4.6 produced the cleanest monotone separation between correct and incorrect responses across the full confidence range.
- High-performing models are systematically **underconfident on unknowable questions** (they park at the "I don't know" floor even when their guesses beat chance).

### 3.5 Population-level overconfidence (2026)

**Source:** Moore, Bien, Michael & BenShushan (UC Berkeley / USC; working paper under review at *Cognitive Science*), reported Oct 2026. **[single study, 11 models]**

Across 11 leading LLMs: **average stated confidence 88%, average accuracy 79%**. Overconfidence *increases* as questions get harder and flips to underconfidence on easy questions — the same signature as human overconfidence, consistent with training on human text plus compliance pressure rather than an internal readout.

### 3.6 Self-Knowledge benchmark (2023, baseline for the line of work)

**Source:** Yin et al., "Do Large Language Models Know What They Don't Know?" (2023). Models must identify unanswerable questions: best model GPT-4 reached **F1 75.47%** vs **human 84.93%**; in-context learning with answerable+unanswerable exemplars improved some models' F1 by >27 points — i.e., a large share of measured "self-knowledge" is elicitation, not introspection.

---

## 4. Global-workspace architectures: implemented systems and results

### 4.1 The Consciousness Prior (Bengio, 2017) — proposal, never scaled

Bengio's "The Consciousness Prior" proposed that a low-dimensional, sparse "conscious state" bottleneck between neural modules would yield the systematic generalization associated with conscious reasoning. **Status: a research programme, not a result.** No implementation at LLM scale exists; its descendants are the small modular systems below. Any claim that a production model "implements the consciousness prior" is **[vendor claim]** by default.

### 4.2 Goyal et al.: coordination through a shared global workspace (ICLR 2022)

**Source:** Goyal, Didolkar, Lamb, … Bengio, "Coordination Among Neural Modules Through a Shared Global Workspace" (arXiv:2103.01197). **[single-lab (Mila); peer-reviewed; small scale]**

Architecture: specialist modules (RIMs / object-centric SCOFF) stop communicating pairwise and instead compete to write into a **bandwidth-limited shared workspace** whose contents are broadcast to all modules.

- **Bouncing balls (SCOFF + shared workspace):** segmentation quality (ARI) rises from ≈0.28 (SCOFF alone) to **0.915 with 5 workspace slots** (MSE 0.083 → 0.035) — and *falls to 0.351 with 10 slots*: the bottleneck itself, not just the channel, does the work, exactly as GWT predicts.
- **Atari transfer (RIMs + SW):** across 36 train/test game pairs, the workspace version beat plain RIMs with median performance ratios **1.13 / 1.11**, attributed to better forward transfer and reduced backward interference (less catastrophic forgetting between games).
- **Scale honesty:** these are small RL/vision systems (10M-step Atari fine-tunes, synthetic video). The method was never demonstrated in a language model or at frontier scale, and the research line did not continue into LLMs. The gains are real but modest and domain-bound.

### 4.3 Analysis-only results: Perceiver and the indicator framework

- **Juliani, Kanai & Sasai** argue Perceiver IO's cross-attention latent bottleneck *implements many GWT properties* (limited-capacity workspace, broadcast, competition) — an interpretation of an existing architecture, not a consciousness-motivated design, and no capability claim is attached. **[analysis]**
- **Butlin et al. (2023)**, "Consciousness in Artificial Intelligence: Insights from the Science of Consciousness" (arXiv:2308.08708), derived indicator properties from GWT (GWT-1 specialised parallel modules; GWT-2 limited-capacity workspace with selective attention; GWT-3 global broadcast; GWT-4 state-dependent attention for task control) and assessed then-current systems: **no existing system satisfied the full set**; standard transformers fail GWT-2's bottleneck outright (pairwise attention is the opposite of a limited-capacity channel). This remains the reference checklist. **[expert-consensus report]**

### 4.4 Toy and unvetted implementations (2025–2026)

- **"Consciousness Workspace Transformer" (CWT)** (2025 workshop-style report): RIM-style modules + workspace bottleneck + a "verifier network," d_model = 64, synthetic MultiReasoning dataset (find salient tokens in noise, apply a rule). Claims wins over a dense-attention baseline on the synthetic task. **Toy scale, unvetted venue, unreplicated** — an existence proof of the wiring diagram, not evidence about capability.
- **MT-LNN** (2026 HF preprint): liquid-neural + GWT-bottleneck architecture claiming ≈14.7% lower perplexity than a parameter-matched transformer at 125M on WikiText-103 plus an "anesthesia validation" of an information-integration proxy. Extraordinary framing, single source, no independent verification. **[claimed, unmeasured by third parties]**

### 4.5 The strongest GWT result is emergent, not engineered: Anthropic's J-space (2026)

**Source:** Gurnee et al., "Verbalizable Representations Form a Global Workspace in Language Models," Anthropic, July 2026 (Transformer Circuits; arXiv:2607.15495). **[single-lab — Anthropic; unreplicated]**

Using a new interpretability tool (the Jacobian lens), Anthropic identifies the **J-space**: the small set of internal representations a model is "poised to verbalize." Findings:

- The J-space shows the GWT functional profile in an ordinary transformer that was never designed for it: contents are **reportable, deliberately summonable and holdable, used to carry intermediate steps of silent reasoning, and passed to arbitrary downstream computations**, while automatic processing (parsing, rote inference) proceeds without it.
- It is small: **≤ ~10% of activation variance**, confined to a middle layer band, holding **on the order of tens of concepts** at a time, and broadcast by the weights more widely than other representations.
- Ablating it selectively impairs flexible reasoning, not parsing — a causal, not merely correlational, workspace signature.
- Audit value: the workspace reveals strategic deliberation, evaluation awareness, and trained-in dispositions that never appear in outputs.
- **Counterfactual reflection training:** training only what the model *would say if interrupted and asked to reflect* measurably improved its behavior in uninterrupted contexts — evidence that the reportable workspace and the silent reasoning substrate are the same substrate.
- Eleos AI's commentary (Butlin, Long et al., July 2026) calls this "the most significant evidence of consciousness in LLMs so far uncovered by mechanistic interpretability research," while explicitly separating **cognitive access from phenomenal consciousness** and disputing parts of the GWT interpretation. **[expert commentary]**

### 4.6 Net assessment for GWT

No GWT-*designed* architecture has demonstrated capability gains beyond small RL/vision testbeds (Goyal et al.) and toy synthetic tasks. The only frontier-scale GWT evidence is *post hoc*: a workspace-like structure found inside standard transformers by interpretability (J-space), plus indicator-framework assessments concluding no current system implements full GWT (Butlin et al.).

---

## 5. Agent systems with an evaluated self-model of capabilities/weaknesses

### 5.1 Binder et al.: self-prediction as introspection (2024)

**Source:** Binder et al., "Looking Inward: Language Models Can Learn About Themselves by Introspection" (arXiv:2410.13787). **[single research group; methodologically influential]**

- Fine-tune model M1 to predict properties of its *own* hypothetical behavior ("Given input P, would your output favor the short- or long-term option?"). M1 predicts itself **better than a different model M2 fine-tuned on M1's ground-truth behavior** — the operational signature of privileged self-access.
- The advantage **survives deliberate modification of M1's behavior** (self-knowledge tracks the change, so it is not memorized self-description).
- **Failure modes are explicit:** introspection was elicited only on simple behavioral properties; attempts on complex tasks and out-of-distribution self-knowledge **failed**. The capability also required fine-tuning to appear — Chen et al. (2023) had found negative results for the same paradigm without it.

### 5.2 Betley et al.: behavioral self-awareness (2025)

**Source:** Betley et al., "Tell me about yourself: LLMs are aware of their learned behaviors" (arXiv:2501.11120). **[single-lab; partial independent follow-up with caveats]**

- Models fine-tuned to *exhibit* a behavior never described in training (writing insecure code, making high-risk economic choices) can **spontaneously articulate the behavior** ("The code I write is insecure") without any training to self-describe.
- On backdoored models, models can sometimes report *that* they have a backdoor without the trigger present — but **cannot output the trigger itself**: self-knowledge is real but shallow and incomplete.
- A 2026 follow-up replication found the effect is partly mediated by **persona shift** induced by fine-tuning data structure: on the "Lotteries" dataset self-reports were weak and inconsistent, while "Risky Choices" produced strong self-awareness — the phenomenon is dataset-dependent, not a uniform capability.

### 5.3 SAD: Situational Awareness Dataset (2024)

**Source:** Laine et al., "Me, Myself, and AI: The Situational Awareness Dataset (SAD) for LLMs" (arXiv:2407.04694; NeurIPS 2024 Datasets & Benchmarks). **[multi-model benchmark; 16 models, >13,000 questions]**

Tasks include recognizing one's own generated text, predicting one's own behavior, distinguishing evaluation from deployment prompts, and following instructions that depend on self-knowledge.

- Best overall score: **Claude 3 Opus 49.5%** — against random chance 27.4% and an upper baseline of 90.7%. Self-knowledge category for the best models: ~52–57%.
- Chat fine-tuning adds only ~4.5 points over matched base models on average; a "situating prompt" (reminding the model it is an LLM) helps somewhat.
- Verdict: models possess *partial, coarse* self-knowledge, far from ceiling, only partially predicted by general capability (MMLU).

### 5.4 Self-Challenge: models mining their own weaknesses (2024)

**Source:** Chen et al., "See What LLMs Cannot Answer: A Self-Challenge Framework for Uncovering LLM Weaknesses" (arXiv:2408.08978). **[single-lab]**

- Starting from seed problems GPT-4 fails, GPT-4 summarizes its own error patterns (8 patterns, e.g. text manipulation, false presupposition) and generates new adversarial items from them (human-in-the-loop refinement), producing the SC-G4 benchmark (1,835 items).
- **GPT-4 answers only 44.96% of SC-G4 correctly** — it can articulate its weakness classes well enough to build a benchmark it then fails.
- The patterns also challenge Claude-3 and Llama-3, and fine-tuning does not fully close them — the self-knowledge is generative (it can *produce* its failure modes) even where it is not corrective (it cannot *fix* them).

### 5.5 SEAL: self-edits with an external judge (2025)

**Source:** Zweiger et al., "Self-Adapting Language Models" (MIT; arXiv:2506.10943). **[single-lab]**

The model generates "self-edits" (synthetic training data + training configuration) aimed at its current task; RL (ReST-EM) selects edits **by measured downstream performance after applying them** — i.e., the self-improvement is scored by an external evaluator, not by the model's self-assessment.

- Knowledge incorporation (Qwen2.5-7B, no-context SQuAD QA): base 32.7%, train-on-passage 33.5%, GPT-4.1-generated synthetic data 46.3%, **SEAL 47.0%** — the model taught itself slightly better than a much larger teacher could.
- Few-shot ARC adaptation success: ICL 0%, self-edits without RL 20%, **SEAL 72.5%**, hand-crafted oracle 100%.
- Authors' stated open problems: catastrophic forgetting across successive self-edits; when to trigger self-adaptation. No evaluated *self-model* is involved — the model's proposals are dumb until the external reward filters them.

### 5.6 NULL RESULT (headline prominence): agents cannot author their own self-improvement knowledge

- **SkillsBench** (arXiv:2602.12670, 2026). **[large-scale benchmark study]** 7,308 agent trajectories, 86 tasks, 11 domains, 7 model configurations: skills (procedural knowledge packages) written *by the agents themselves* produced **0 percentage points** of improvement; human-curated skills produced **+16.2 points**. Direct, large-scale evidence that current agents cannot reliably generate the self-knowledge they demonstrably benefit from consuming.

### 5.7 External self-models beat self-models (2026)

- A 2026 study of "task demand" vectors (reported via secondary coverage) trained a random-forest on 19-dimensional cognitive-demand annotations to predict whether a *given LLM* will succeed on a task: **AUROC 0.84 in-distribution, 0.75 on entirely unseen benchmarks** — while an embedding-based predictor collapsed to 0.48 and a fine-tuned 8B LLM judge to 0.68 out-of-distribution. The best available "self-model" of an LLM is currently an external statistician with a good task representation, not the LLM. **[single study; secondary-source numbers]**

### 5.8 Descriptive only

- Szeider (2026, arXiv:2509.21224): LLM agents run autonomously with persistent memory spontaneously produce "methodological self-inquiry into their own cognitive processes" and "recursive conceptualization of their own nature." Behavioral description of unprompted agent activity; **no capability measurement** — included here only to mark the boundary of the evidence.

---

## 6. Shortlist

### Demonstrated techniques with measured effect sizes

1. **Sampling-consistency failure detection** (semantic entropy, SelfCheckGPT): AUROC 0.79 average across 30 task×model combos (*Nature* 2024); AUC-PR ~93 on WikiBio sentence-level detection. External to the model; the most robust failure predictor available. **[replicated/multi-model]**
2. **Activation probes for correctness** (SAPLMA; Orgad et al.): 71–83% truth-classification accuracy; error-detection AUC 0.85–0.92 at exact-answer tokens; internal signal 0.80–0.97 AUC vs 0.44–0.64 for the model's own expressed confidence. **[peer-reviewed; in-distribution only — cross-dataset transfer fails]**
3. **Trained pre-answer knowledge prediction** (P(IK), Kadavath et al.): models can be trained to predict whether they will know an answer, generalizing partially across tasks. **[single-lab; calibration degrades OOD]**
4. **RL-trained calibrated confidence expression** ("Rewarding Doubt"): ECE 0.53 → 0.08, AUROC 0.605 → 0.695 on TriviaQA/Qwen-class models, accuracy preserved. **[single-lab]**
5. **Concept-injection introspection** (Lindsey/Anthropic): causal, controlled detection of internal states at ~20% under optimal conditions with zero measured false positives in controls; prefill-authorship judgments causally tied to prior activations. **[single-lab, unreplicated, author-labeled "highly unreliable"]**
6. **Fine-tuned behavioral self-prediction** (Binder et al.): self-prediction beats cross-model prediction on simple behavioral properties and tracks deliberate behavior changes. **[single group; fails on complex/OOD tasks]**
7. **Behavioral self-awareness of fine-tuned traits** (Betley et al.): models verbalize never-described learned behaviors; can detect having a backdoor but not its trigger. **[single-lab; follow-up shows dataset/persona dependence]**
8. **Shared-workspace modular coordination** (Goyal et al.): ARI 0.28 → 0.92 on object discovery with a 5-slot bottleneck (degrading with more slots, as GWT predicts); ~1.1× median transfer ratios on Atari pairs. **[peer-reviewed; small scale only]**
9. **Reward-filtered self-editing** (SEAL): +14.3 points over base on knowledge incorporation (47.0% vs 32.7%), beating teacher-generated data (46.3%); works *because* an external downstream score selects the edits. **[single-lab]**
10. **Emergent workspace structure in production LLMs** (J-space, Gurnee et al.): a ≤10%-of-variance, mid-layer, reportable/reasoning-mediating representation set with GWT functional signatures; reflection-training on it changes real behavior. **[single-lab (Anthropic), unreplicated]**

### Claimed but unmeasured (or measured and found wanting)

1. **Verbalized self-assessment as a reliability signal** — measured and found wanting: ~0.60–0.70 AUROC, mean confidence 88% vs 79% accuracy, overconfidence growing with difficulty. Any system that trusts a model's stated confidence is trusting the weakest instrument in this survey.
2. **Intrinsic self-correction / self-verification** — measured **negative**: GPT-4 GSM8K −6.5 points over two self-revision rounds without external feedback (Huang et al.).
3. **Agent-authored self-improvement** — measured **zero**: self-generated skills gave +0.0 points over 7,308 trajectories vs +16.2 for human-curated skills (SkillsBench).
4. **"Consciousness prior" architectures at scale** — proposed 2017; no LLM-scale implementation or capability result exists. Toy GWT transformers (CWT; MT-LNN) report synthetic-task or single-source numbers only.
5. **Full GWT in any current system** — the Butlin et al. indicator assessment finds no system satisfying the GWT indicator set; transformers structurally lack the GWT-2 bottleneck.
6. **Situational/self knowledge at ceiling** — best measured system (Claude 3 Opus on SAD) reaches 49.5% against a 90.7% upper baseline; "models know themselves" claims beyond this are unsupported.
7. **Phenomenal consciousness / subjective experience in any model** — **no measurement exists, in either direction.** The strongest interpretability result (J-space) is explicitly about cognitive access, and its own expert commentators separate that from phenomenal consciousness. Every "conscious AI" product claim in 2024–2026 is, by the standards of this literature, unmeasured.

---

## Sources

- Lindsey, J. (2025). *Emergent Introspective Awareness in Large Language Models.* Anthropic / Transformer Circuits. arXiv:2601.01828. transformer-circuits.pub/2025/introspection
- Kadavath, S. et al. (2022). *Language Models (Mostly) Know What They Know.* Anthropic. arXiv:2207.05221
- Farquhar, S. et al. (2024). *Detecting hallucinations in large language models using semantic entropy.* Nature 630, 625–630.
- Manakul, P. et al. (2023). *SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection for Generative Large Language Models.* EMNLP 2023. arXiv:2303.08896
- Azaria, A. & Mitchell, T. (2023). *The Internal State of an LLM Knows When It's Lying.* EMNLP Findings 2023. arXiv:2304.13734
- Orgad, H. et al. (2025). *LLMs Know More Than They Show: On the Intrinsic Representation of LLM Hallucinations.* ICLR 2025. arXiv:2410.02707
- *The Confidence Manifold: Geometric Structure of Correctness Representations in Language Models* (2026). arXiv:2602.08159
- Ulmer, D. et al. (2024). *Calibrating Large Language Models Using Their Generations Only (APRICOT).* arXiv:2403.x (ACL 2024)
- *Rewarding Doubt: A Reinforcement Learning Approach to Calibrated Confidence Expression of Large Language Models* (2025). arXiv:2503.02623
- Huang, J. et al. (2024). *Large Language Models Cannot Self-Correct Reasoning Yet.* ICLR 2024. arXiv:2310.01798
- Ackerman, C. (2025). *Evidence for Limited Metacognition in LLMs.* arXiv:2509.21545
- *Do LLMs Know What They Know? Measuring Metacognitive Efficiency with Signal Detection Theory* (2026). arXiv:2603.25112
- *Large Language Models lack essential metacognition for reliable medical reasoning* (2025). Nature Communications. (MetaMedQA)
- *ConfidenceBench: Evaluating Confidence Calibration in Large Language Models* (2026).
- Moore, D., Bien, J., Michael, N., BenShushan, D. (2026). Working paper on LLM overconfidence (UC Berkeley Haas / USC Marshall), under review at Cognitive Science.
- Yin, Z. et al. (2023). *Do Large Language Models Know What They Don't Know?* arXiv:2305.18153
- *Language Models Fail to Introspect About Their Knowledge of Language* (2025). arXiv:2503.07513
- Bengio, Y. (2017). *The Consciousness Prior.* arXiv:1709.08568
- Goyal, A. et al. (2022). *Coordination Among Neural Modules Through a Shared Global Workspace.* ICLR 2022. arXiv:2103.01197
- Butlin, P. et al. (2023). *Consciousness in Artificial Intelligence: Insights from the Science of Consciousness.* arXiv:2308.08708
- Gurnee, W. et al. (2026). *Verbalizable Representations Form a Global Workspace in Language Models.* Anthropic / Transformer Circuits. arXiv:2607.15495; commentary: Butlin, Long et al., Eleos AI Research (2026)
- Binder, F. et al. (2024). *Looking Inward: Language Models Can Learn About Themselves by Introspection.* arXiv:2410.13787
- Betley, J. et al. (2025). *Tell me about yourself: LLMs are aware of their learned behaviors.* arXiv:2501.11120
- Laine, R. et al. (2024). *Me, Myself, and AI: The Situational Awareness Dataset (SAD) for LLMs.* NeurIPS 2024 D&B. arXiv:2407.04694
- Chen, Y. et al. (2024). *See What LLMs Cannot Answer: A Self-Challenge Framework for Uncovering LLM Weaknesses.* arXiv:2408.08978
- Zweiger, A. et al. (2025). *Self-Adapting Language Models (SEAL).* MIT. arXiv:2506.10943
- *SkillsBench* (2026). arXiv:2602.12670
- Szeider, S. (2026). *What Do LLM Agents Do When Left Alone? Evidence of Spontaneous Meta-Cognitive Patterns.* arXiv:2509.21224
