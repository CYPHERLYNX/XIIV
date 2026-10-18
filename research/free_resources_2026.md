# Free Resources Audit for XIIV 1B Training
**Date:** 2026-10-15 | **Researched via web, verified against 2026 sources**

> TL;DR: **Kaggle is the answer for free GPU training** — 30 hrs/week of T4×2 (16GB each), fully automatable via CLI, background execution. SGBD's block-independent design maps perfectly to Kaggle sessions. **HuggingFace Datasets Hub is the answer for storage** — best-effort unlimited public storage. Everything below needs only one-time account setups from Fahad.

---

## RANKING: Impact for XIIV 1B Training Goal

| Rank | Resource | Why it matters | Fahad action needed |
|------|----------|----------------|---------------------|
| 1 | **Kaggle GPU** | 30 GPU-hrs/week, T4×2, API-automated, background exec | Account + phone verify + API token (~10 min, one-time) |
| 2 | **HuggingFace Datasets Hub** | Unlimited-ish public storage for our 66B-token pipeline | Account + write token (~5 min, one-time) |
| 3 | **Lightning AI** | A100/H100 possible on free credits, persistent env | Account signup (~5 min, one-time) |
| 4 | **Groq / Cerebras APIs** | Free LLM inference for eval + synthetic data gen | API keys (~5 min each, one-time) |
| 5 | **Google Colab** | Supplementary T4, manual only | Google account (likely has one) |
| 6 | **Backblaze B2** | 10GB free backup, no card | Account signup (~5 min, one-time) |
| 7 | **Modal $30/mo** | Serverless GPU for inference | Account + card (likely required) |

---

## 1. FREE GPU/TPU COMPUTE

### 1.1 Kaggle — ⭐ TOP PICK
| | |
|---|---|
| **GPU** | NVIDIA T4 ×2 (16GB each) or P100 (16GB) |
| **Quota** | **30 GPU-hours/week** + 20 TPU-hours/week (TPU v3-8) |
| **Session limit** | 9–12 hours per session |
| **System RAM** | ~13 GB |
| **Disk** | ~73 GB workspace, 20 GB persistent outputs |
| **Background exec** | **YES** — close browser, notebook keeps running |
| **Credit card** | **No** |
| **Automation** | **YES** — full CLI: `kaggle kernels push` → `kaggle kernels status` → `kaggle kernels output` |
| **Catch** | Phone/SMS verification required for GPU + internet. Notebooks public by default on free tier (use Secrets for keys). Quota resets ~Saturday 00:00 UTC. |

**Why it's perfect for XIIV:** SGBD trains 8 independent blocks (~115M params each). Each block = one Kaggle session. A 115M-param block on T4×2 trains in hours, not days. The 133M teacher activation cache can be uploaded as a Kaggle Dataset (mounted read-only at `/kaggle/input/`).

**Activation (Fahad, one-time ~10 min):**
1. Create account at kaggle.com
2. Settings → Phone verification (required for GPU + internet)
3. Account page → "Create New API Token" → downloads `kaggle.json`
4. Place at `~/.kaggle/kaggle.json` with `chmod 600`

**Automation:** See `scripts/free_resources/kaggle_setup.sh` and `kaggle_sgbd_block.py` template.

**Sources:** [Kaggle GPU docs](https://www.kaggle.com/docs/efficient-gpu-usage), [awesome-free-compute-2026](https://github.com/amitpandeytiktok/awesome-free-compute-2026)

---

### 1.2 Lightning AI Studio
| | |
|---|---|
| **GPU** | T4 typically; **A100 (40/80GB), H100, H200 available** on credit-based allocation |
| **Quota** | **15 free credits/month** (~15–22 hrs T4; up to ~80 hrs on interruptible per some reports) |
| **Storage** | 50–100 GB persistent |
| **Session** | 4-hour restart on free studios; background execution YES |
| **Access** | SSH, VS Code, full Linux environment |
| **Credit card** | **No** |
| **Catch** | Credit costs per GPU type vary; free studios restart every 4h (code must checkpoint) |

**Use case:** A100 access for critical runs (e.g., full-model eval, large-batch distillation). Persistent env means no re-setup.

**Activation:** Signup at lightning.ai (~5 min). No card.

---

### 1.3 Google Colab (Free)
| | |
|---|---|
| **GPU** | T4 (16GB), sometimes K80 or nothing during peak |
| **Quota** | ~10–15 hrs/week, **dynamic and not guaranteed** |
| **Session limit** | 12 hours hard cap; **90-minute idle timeout kills session** |
| **RAM** | ~12 GB |
| **Background exec** | **NO** on free tier |
| **Credit card** | No |
| **Automation** | **NO** — consumer Colab cannot be scripted (confirmed 2026) |

**Use case:** Manual prototyping only. Inferior to Kaggle in every way for our needs (no automation, no background exec, unpredictable quota).

---

### 1.4 HuggingFace ZeroGPU
| | |
|---|---|
| **GPU** | Shared A100 / RTX PRO 6000 Blackwell slices |
| **Quota** | A few minutes/day free |
| **Call limit** | <120 seconds per call |
| **Credit card** | No |

**Use case:** Inference demos only. **Useless for training.** Gradio-only, ephemeral.

---

### 1.5 AWS SageMaker Studio Lab
| | |
|---|---|
| **GPU** | T4 |
| **Quota** | 4 hrs/session, 8 hrs/day |
| **Storage** | 15 GB persistent |
| **Credit card** | No |

**Catch:** Requires application/approval. Backup option if Kaggle quota exhausted.

---

### 1.6 Modal
| | |
|---|---|
| **Quota** | **$30/month free compute credit** (recurring) |
| **Automation** | **YES** — Python SDK, serverless |
| **Catch** | Docs say payment method on file likely required. Best for inference, not long training. |

---

### 1.7 Paperspace Gradient — SKIP
Free tier effectively dead in 2026. Free machines ~never available; requires $8/mo Pro. Not worth it.

---

### 1.8 Oracle Cloud Always Free — CPU ONLY
**Important 2026 change:** Oracle **halved** the Always Free ARM allowance in August 2026. Now: **2 OCPU / 12 GB RAM** (was 4/24GB). No GPU. Still useful as a free CPU box (better than our current 7GB), but requires card for signup and suffers "out of host capacity" errors. Not a training solution.

---

## 2. FREE DATASET STORAGE

### 2.1 HuggingFace Datasets Hub — ⭐ TOP PICK
| | |
|---|---|
| **Public storage** | **Best-effort unlimited** (soft cap ~300GB/repo; contact HF if larger; PRO gets guaranteed 1TB) |
| **Private storage** | 100 GB free |
| **Per file** | 200 GB max (50 GB recommended) |
| **Per repo** | 100,000 files max; <10k per folder |
| **Upload** | Via `huggingface_hub` Python API or `hf` CLI |
| **Streaming** | `datasets` library streams directly — no download needed |

**Why it's perfect:** Our tokenized datasets (66B tokens ≈ 130GB) fit comfortably. Public repos are free. The `datasets` library already streams from HF — our pipeline already works this way.

**Strategy:** Push each completed tokenized dataset as `CYPHERLYNX/xiiv-frontier-<name>`. Push 200M-token coresets as `CYPHERLYNX/xiiv-coresets`. This gives us unlimited backup + public visibility for the open-source project.

**Activation (Fahad, one-time ~5 min):**
1. HF account (likely has one)
2. Settings → Access Tokens → Create with **Write** permission
3. Save token securely (never in chat)

**Automation:** See `scripts/free_resources/hf_upload_dataset.py`.

**Sources:** [HF storage limits](https://huggingface.co/docs/hub/storage-limits), [hub-docs](https://github.com/huggingface/hub-docs/blob/HEAD/docs/hub/datasets-upload-guide-llm.md)

---

### 2.2 Cloudflare R2
| | |
|---|---|
| **Free storage** | 10 GB |
| **Egress** | **$0 forever** (unlimited) |
| **Ops** | 1M Class A + 10M Class B /month |
| **Credit card** | **Required** |

**Use case:** Small file hosting (checkpoints, configs). 10GB is too small for datasets.

---

### 2.3 Backblaze B2
| | |
|---|---|
| **Free storage** | 10 GB |
| **Egress** | Free up to 3× stored data |
| **Credit card** | **Not required** |

**Use case:** Backup for critical small files (model checkpoints, configs). No card = easy signup.

---

## 3. FREE MODEL HOSTING / INFERENCE

| Platform | Free tier | Best for | Card? |
|----------|-----------|----------|-------|
| **HuggingFace Spaces** | Free CPU Spaces; 2 ZeroGPU Spaces free | Gradio/Streamlit demos | No |
| **Render** | 750 hrs/month (spins down after idle) | REST API servers | No |
| **Railway** | $5/month credit | Backend APIs | Likely |
| **Modal** | $30/month credit | Serverless GPU inference | Likely |
| **Replicate** | $5 one-time credit | Custom model Docker deploys | Yes |
| **Cloudflare Workers** | 100K req/day | Edge inference (small models) | No |

**For XIIV:** Once trained, push to HF Hub (free model hosting) + a Gradio Space for demo. Modal $30/mo for GPU inference if needed.

---

## 4. FREE LLM APIs (for eval, synthetic data, research)

All OpenAI-compatible, no credit card:

| Provider | Free quota | Rate limit | Best for |
|----------|------------|------------|----------|
| **Groq** | ~1k–14.4k req/day | 30 RPM | Fast eval, low latency |
| **Cerebras** | **1M tokens/day** | 5 RPM, 30k TPM | Bulk processing, classification |
| **Mistral** | ~1B tokens/month (Experiment tier) | ~1 RPM | Largest free allowance |
| **Google AI Studio** | Gemini Flash free | 5–30 RPM | Long context tasks |
| **NVIDIA NIM** | ~1000 req/day/model | 40 RPM | 100+ models incl. frontier |
| **DeepSeek** | 5M tokens one-time grant | — | One-shot bulk tasks |
| **OpenRouter** | 50 req/day free models | 20 RPM | Multi-model access |

**Use cases for XIIV:**
- **Eval:** Run benchmark comparisons (XIIV vs frontier) via Groq/Cerebras
- **Synthetic data:** Generate reasoning traces, quality labels via Cerebras (1M/day)
- **Research:** Query frontier models about techniques via NVIDIA NIM

**Activation:** Each needs a free API key from their console (~5 min each, no card).

---

## STRATEGIC PLAN: How This Changes XIIV

### The Kaggle + SGBD Play
SGBD was designed for CPU-only 7GB RAM. But it's actually **better** on GPU:
- 8 independent blocks, ~115M params each
- Each block = 1 Kaggle session (9-12h on T4×2)
- 30 GPU-hrs/week = ~2-3 blocks/week
- **Full 1B in ~3-4 weeks of free Kaggle time**

This doesn't replace the CPU path — it **accelerates** it massively.

### The HF Data Layer Play
- Push all tokenized datasets to HF Hub (public, free)
- Push coresets as versioned releases
- Training (Kaggle or local) streams directly from HF
- No more 50GB local disk constraint for data

### What Fahad Needs to Do (one-time, ~30 min total)
1. ☐ Kaggle: account + phone verify + API token (~10 min)
2. ☐ HuggingFace: write token (~5 min)
3. ☐ Lightning AI: signup (~5 min) [optional but recommended]
4. ☐ Groq + Cerebras: API keys (~5 min each) [optional, for eval]

**Everything else is already prepared.** Scripts are in `scripts/free_resources/`.

---

## READY-TO-RUN SCRIPTS

| Script | Purpose | Status |
|--------|---------|--------|
| `scripts/free_resources/kaggle_setup.sh` | Install kaggle CLI, verify `~/.kaggle/kaggle.json` | ✅ Ready |
| `scripts/free_resources/kaggle_sgbd_block.py` | Template: push one SGBD block training job to Kaggle | ✅ Ready |
| `scripts/free_resources/kernel-metadata.json` | Kaggle kernel config (T4×2, internet on) | ✅ Ready |
| `scripts/free_resources/hf_upload_dataset.py` | Upload tokenized `.bin` as HF dataset | ✅ Ready |

---

## SOURCES
- [awesome-free-compute-2026](https://github.com/amitpandeytiktok/awesome-free-compute-2026) (123 days ago)
- [free-llm-api-hub](https://github.com/pacocartones/free-llm-api-hub) (verified 2026-07-12)
- [HF storage limits](https://huggingface.co/docs/hub/storage-limits)
- [H100/A100 free access guide](https://medium.com/@mattral-lifelong-learning/h100-vs-a100-in-2026-the-real-differences-and-where-to-actually-get-one-for-free-or-almost-free-4764d37a2e82) (Jun 2026)
- [Kaggle hardware SKILL.md](https://github.com/multi-modal-rtm/uznlp_course/blob/HEAD/.claude/skills/kaggle-hardware/SKILL.md) (verified Jun 2026)
- [Spheron Colab alternatives](https://www.spheron.network/blog/google-colab-alternatives-8-gpu-clouds-compared-2026/) (15 days ago)
- [Oracle ARM halving](https://www.webpronews.com/oracle-slashes-always-free-arm-capacity-in-half/) (Aug 2026)
- [502 free AI APIs list](https://theaileverage.beehiiv.com/p/502-free-ai-apis-412-with-no-credit-card-the-full-list-and-how-to-chain-them) (24 days ago)
