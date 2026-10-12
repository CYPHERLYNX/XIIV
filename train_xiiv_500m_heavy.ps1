<#
.SYNOPSIS
  One-click XIIV-500M HEAVY training for NVIDIA RTX 4050 (6GB VRAM) —
  the real-data regime. No children's stories.
.DESCRIPTION
  Trains the grown 500M checkpoint (503.45M params, verified drift gate)
  on a REAL mixed corpus: FineWeb-Edu (general knowledge) + StarCoder
  (code) + books (long-form fluency), built by scripts/mix_corpus.py.

  Frozen config (do not hand-tune; see HEAVY_TRAINING_RUNBOOK.md):
    arch: d=1024, L=27, h=16, ffn=2816, vocab=32768 (tied)
    opt:  AdamW 8-bit, lr 2.0625e-4 (inverse-width from the 133M rung),
          cosine decay to 2.0625e-5, warmup 1000 (new data distribution)
    batch: micro-batch 2 x grad-accum 2 = 4 x 128 = 512 tok/step
    steps: 8218 -> 203218 (195,000 more steps ~= 100M tokens, 4-6 days)
    VRAM: bf16 + grad-checkpoint + 8-bit Adam ~ 4-5GB of 6GB
    guarantee-native stays ON: contract-prob 0.4, lock-prob 0.4,
    adversarial contracts 0.3 — the thesis trains on real data too.

  Data flow (all local, automatic):
    1. data_fineweb/ — required. Built on the spot (~30 min stream) if missing.
    2. data_code/    — optional. StarCoder is HF-gated: run `hf auth login`
                       (free HF account, 2 min) first; skipped with a warning if missing.
    3. data_books/   — optional. Built by scripts/prepare_books.py; skipped if missing.
    4. scripts/mix_corpus.py concatenates what exists -> data_heavy/
       (mix order + token counts + mix_id recorded in data_heavy/meta.json;
        skipped when the mix is already up to date).
    5. 20-step VRAM smoke test on the real mix, then the crash-resilient
       training loop. Resumes from checkpoints_500m_heavy/ automatically.

  Safe to re-run any time. STOPS (no auto-relaunch) on the non-finite-loss
  abort so a diverging run never burns VRAM silently.
.EXAMPLE
  .\train_xiiv_500m_heavy.ps1
  .\train_xiiv_500m_heavy.ps1 -SkipSmoke
#>
param(
  [switch]$SkipSmoke
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

# ---------- frozen heavy-regime config ----------
$DModel = 1024; $NLayers = 27; $NHeads = 16; $FFNDim = 2816
$LR = 2.0625e-4; $LRMin = 2.0625e-5; $Warmup = 1000
$TargetStep = 203218        # growth step 8218 + 195000 (~100M tokens @ 512 tok/step)
$MicroBatch = 2; $GradAccum = 2
$TokFile = "weights/tokenizer_32k.json"
$CkptDir = "checkpoints_500m_heavy"; $GrownCkpt = "checkpoints_500m_gn/ckpt_8218.pt"
$MixDir = "data_heavy"

function Fail($msg) {
  Write-Host ""
  Write-Host "ERROR: $msg" -ForegroundColor Red
  exit 1
}

function LatestCkptStep($dir) {
  $cks = Get-ChildItem "$dir/ckpt_*.pt" -ErrorAction SilentlyContinue |
    Sort-Object { [int]($_.BaseName -replace 'ckpt_','') } |
    Select-Object -Last 1
  if ($null -eq $cks) { return 0 }
  return [int]($cks.BaseName -replace 'ckpt_','')
}

# ---------- 1. prerequisites ----------
Write-Host "== [1/6] prerequisites ==" -ForegroundColor Cyan
try { $py = (Get-Command python).Source } catch { Fail "Python not found. Install Python 3.10+ from python.org and retry." }
Write-Host "  $(& python --version 2>&1)"
try { nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | ForEach-Object { Write-Host "  GPU: $_" } }
catch { Fail "nvidia-smi failed — is an NVIDIA driver installed?" }

# ---------- 2. venv + deps ----------
Write-Host "== [2/6] environment ==" -ForegroundColor Cyan
if (-not (Test-Path ".venv-gpu")) {
  Write-Host "  creating .venv-gpu ..."
  & python -m venv .venv-gpu
}
$VenvPy = Join-Path $Root ".venv-gpu\Scripts\python.exe"
try { & $VenvPy -c "import torch, bitsandbytes; assert torch.cuda.is_available()" 2>$null }
catch {
  Write-Host "  installing GPU dependencies (one-time, a few minutes) ..."
  & $VenvPy -m pip install --upgrade pip | Out-Null
  & $VenvPy -m pip install -r requirements-gpu.txt
  if ($LASTEXITCODE -ne 0) { Fail "pip install failed. Check your network and retry." }
}
& $VenvPy -c "import torch; print('  torch', torch.__version__, '| cuda', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0))"

# ---------- 3. data: build what's missing, then mix ----------
Write-Host "== [3/6] data ==" -ForegroundColor Cyan
if (-not (Test-Path $TokFile)) { Fail "missing $TokFile — unzip the full 500M kit into this folder first." }

if (-not (Test-Path "data_fineweb/train.bin")) {
  Write-Host "  data_fineweb missing — building (streams ~100M tokens, ~30 min) ..."
  & $VenvPy scripts/prepare_fineweb.py --source fineweb-edu --tokenizer $TokFile `
    --out-dir data_fineweb --max-tokens 100000000 --valid-frac 0.01
  if ($LASTEXITCODE -ne 0) { Fail "FineWeb-Edu build failed. Check the log above and retry." }
} else {
  Write-Host "  data_fineweb present"
}

$MixInputs = @("data_fineweb")
if (Test-Path "data_code/train.bin") {
  $MixInputs += "data_code"
  Write-Host "  data_code present (StarCoder)"
} else {
  Write-Host "  data_code missing — code skipped. For the full mix: pip install huggingface_hub, run" -ForegroundColor Yellow
  Write-Host "    hf auth login   (free HF account, 2 minutes — StarCoder is gated)" -ForegroundColor Yellow
  Write-Host "    then: python scripts/prepare_fineweb.py --source starcoder --tokenizer $TokFile --out-dir data_code --max-tokens 30000000" -ForegroundColor Yellow
}
if (Test-Path "data_books/train.bin") {
  $MixInputs += "data_books"
  Write-Host "  data_books present"
} else {
  Write-Host "  data_books missing — books skipped (optional; see scripts/prepare_books.py)" -ForegroundColor Yellow
}

Write-Host "  mixing -> $MixDir ..."
& $VenvPy scripts/mix_corpus.py --inputs ($MixInputs -join ",") --out $MixDir
if ($LASTEXITCODE -ne 0) { Fail "corpus mix failed." }
$MixMeta = Get-Content "$MixDir/meta.json" -Raw | ConvertFrom-Json
$DatasetId = "heavy-mix-" + $MixMeta.mix_id
Write-Host "  mix: $($MixMeta.train_tokens/1e6) M train tok, dataset-id=$DatasetId"

# ---------- 4. seed the grown checkpoint ----------
Write-Host "== [4/6] grown checkpoint ==" -ForegroundColor Cyan
if (-not (Test-Path $CkptDir)) { New-Item -ItemType Directory $CkptDir | Out-Null }
if ((LatestCkptStep $CkptDir) -eq 0) {
  if (-not (Test-Path $GrownCkpt)) { Fail "missing $GrownCkpt — unzip the full 500M kit first." }
  Copy-Item $GrownCkpt (Join-Path $CkptDir "ckpt_8218.pt")
  Write-Host "  seeded grown checkpoint -> $CkptDir/ckpt_8218.pt (optimizer starts fresh, step continues at 8218)"
} else {
  Write-Host "  resuming from step $(LatestCkptStep $CkptDir)"
}

$TrainArgs = @(
  "-m", "xiiv.train",
  "--data", $MixDir, "--tokenizer", $TokFile, "--ckpt-dir", $CkptDir,
  "--d-model", $DModel, "--n-layers", $NLayers, "--n-heads", $NHeads, "--ffn-dim", $FFNDim,
  "--lr", $LR, "--lr-min", $LRMin, "--warmup-steps", $Warmup, "--total-steps", $TargetStep,
  "--batch-size", $MicroBatch, "--grad-accum", $GradAccum, "--optimizer", "adamw8bit",
  "--bf16", "--grad-checkpoint", "--device", "cuda",
  "--lock-prob", "0.4",
  "--contract-enabled", "--contract-prob", "0.4",
  "--contract-adversarial-frac", "0.3", "--contract-max-tokens", "96",
  "--val-every", "500", "--val-batches", "8", "--log-every", "25",
  "--ckpt-every-min", "10", "--seed", "1337", "--threads", "4",
  "--run-name", "xiiv-500m-heavy", "--dataset-id", $DatasetId
)

# ---------- 5. VRAM smoke test (real measurement, 20 steps, on the real mix) ----------
if (-not $SkipSmoke) {
  Write-Host "== [5/6] VRAM smoke test ==" -ForegroundColor Cyan
  $ProbeArgs = @(
    "-m", "xiiv.train",
    "--data", $MixDir, "--tokenizer", $TokFile, "--ckpt-dir", "checkpoints-500m-heavy-probe",
    "--d-model", $DModel, "--n-layers", $NLayers, "--n-heads", $NHeads, "--ffn-dim", $FFNDim,
    "--lr", $LR, "--lr-min", $LRMin, "--warmup-steps", $Warmup, "--total-steps", 20,
    "--batch-size", $MicroBatch, "--grad-accum", $GradAccum, "--optimizer", "adamw8bit",
    "--bf16", "--grad-checkpoint", "--device", "cuda",
    "--lock-prob", "0.4",
    "--contract-enabled", "--contract-prob", "0.4",
    "--contract-adversarial-frac", "0.3", "--contract-max-tokens", "96",
    "--log-every", "20", "--seed", "1337", "--threads", "4"
  )
  $probe = & $VenvPy @ProbeArgs 2>&1 | Out-String
  ($probe -split "`n" | Select-Object -Last 5) | ForEach-Object { Write-Host "    $_" }
  $vramLine = ($probe -split "`n") | Where-Object { $_ -match "peak_vram|VRAM|vram" } | Select-Object -Last 1
  if ($vramLine) { Write-Host "  $vramLine" -ForegroundColor Yellow }
  Remove-Item -Recurse -Force "checkpoints-500m-heavy-probe" -ErrorAction SilentlyContinue
  $ans = Read-Host "  Start the full heavy run to step $TargetStep on mix $DatasetId ? [Y/n]"
  if ($ans -match "^[nN]") { Write-Host "Aborted by user."; exit 0 }
}

# ---------- 6. training (crash-resilient loop) ----------
Write-Host "== [6/6] training (heavy regime, no stories) ==" -ForegroundColor Cyan
Write-Host "  logs -> $CkptDir/train.log | target step $TargetStep | dataset $DatasetId"
while ($true) {
  $s = LatestCkptStep $CkptDir
  if ($s -ge $TargetStep) { Write-Host "Complete at step $s." -ForegroundColor Green; break }
  Write-Host "  launching from step $s ..."
  & $VenvPy @TrainArgs 2>&1 | Tee-Object -FilePath "$CkptDir/train.log" -Append
  if (Select-String -Path "$CkptDir/train.log" -Pattern "aborting: 20 consecutive non-finite losses" -Quiet) {
    Fail "non-finite-loss abort — run diverged. Do NOT relaunch blindly; inspect $CkptDir/train.log first."
  }
  Write-Host "  train.py exited; waiting 60s before relaunch ..." -ForegroundColor Yellow
  Start-Sleep -Seconds 60
}

Write-Host ""
Write-Host "Run finished. Best checkpoint: $CkptDir/best.pt" -ForegroundColor Green
