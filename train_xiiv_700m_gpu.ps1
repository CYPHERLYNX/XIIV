<#
.SYNOPSIS
  One-click XIIV-700M guarantee-native training for NVIDIA RTX 4050 (6GB VRAM).
.DESCRIPTION
  Continues the XIIV lineage: the 133.52M rung (step 8218) was grown
  Net2Net-style to 697.93M params, and this script trains that grown
  checkpoint on the 32k-vocabulary corpus.

  Frozen config (do not hand-tune; see MODEL_CARD_700M.md):
    arch: d=1152, L=30, h=18, ffn=3200, vocab=32768 (tied)
    opt:  AdamW 8-bit, lr 1.833e-4 (inverse-width from the 133M rung),
          cosine decay to 1.833e-5, warmup 500
    batch: micro-batch 2 x grad-accum 2 = 4 x 128 = 512 tok/step
           (identical tokens/step to the 133M rung)
    steps: 8218 -> 48218 (40,000 more steps = 20.48M tokens, same budget)
    VRAM: bf16 + grad-checkpoint + 8-bit Adam ≈ 5.3GB of 6GB (measured)

  Safe to re-run: resumes from the latest checkpoint automatically, and
  relaunches after crashes (but STOPS on the non-finite-loss abort so a
  diverging run never burns VRAM silently).
.EXAMPLE
  .\train_xiiv_700m_gpu.ps1
  .\train_xiiv_700m_gpu.ps1 -SkipSmoke
#>
param(
  [switch]$SkipSmoke
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

# ---------- frozen 700M rung config ----------
$DModel = 1152; $NLayers = 30; $NHeads = 18; $FFNDim = 3200
$LR = 1.833e-4; $LRMin = 1.833e-5; $Warmup = 500
$TargetStep = 48218          # growth step 8218 + 40000
$MicroBatch = 2; $GradAccum = 2
$DataDir = "data_32k"; $TokFile = "weights/tokenizer_32k.json"
$CkptDir = "checkpoints_700m_gn"; $GrownCkpt = "checkpoints_700m_gn/ckpt_8218.pt"

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
Write-Host "== [1/5] prerequisites ==" -ForegroundColor Cyan
try { $py = (Get-Command python).Source } catch { Fail "Python not found. Install Python 3.10+ from python.org and retry." }
Write-Host "  $(& python --version 2>&1)"
try { nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | ForEach-Object { Write-Host "  GPU: $_" } }
catch { Fail "nvidia-smi failed — is an NVIDIA driver installed?" }

# ---------- 2. venv + deps ----------
Write-Host "== [2/5] environment ==" -ForegroundColor Cyan
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

# ---------- 3. data + grown checkpoint ----------
Write-Host "== [3/5] data + grown checkpoint ==" -ForegroundColor Cyan
foreach ($p in @("$DataDir/train.bin", "$DataDir/valid.bin", "$DataDir/meta.json", $TokFile)) {
  if (-not (Test-Path $p)) { Fail "missing $p — unzip the full 700M kit into this folder first." }
}
if (-not (Test-Path $CkptDir)) { New-Item -ItemType Directory $CkptDir | Out-Null }
if ((LatestCkptStep $CkptDir) -eq 0) {
  if (-not (Test-Path $GrownCkpt)) { Fail "missing $GrownCkpt — unzip the full 700M kit first." }
  Write-Host "  grown checkpoint present: $GrownCkpt"
} else {
  Write-Host "  resuming from step $(LatestCkptStep $CkptDir)"
}

$TrainArgs = @(
  "-m", "xiiv.train",
  "--data", $DataDir, "--tokenizer", $TokFile, "--ckpt-dir", $CkptDir,
  "--d-model", $DModel, "--n-layers", $NLayers, "--n-heads", $NHeads, "--ffn-dim", $FFNDim,
  "--lr", $LR, "--lr-min", $LRMin, "--warmup-steps", $Warmup, "--total-steps", $TargetStep,
  "--batch-size", $MicroBatch, "--grad-accum", $GradAccum, "--optimizer", "adamw8bit",
  "--bf16", "--grad-checkpoint", "--device", "cuda",
  "--lock-prob", "0.4",
  "--contract-enabled", "--contract-prob", "0.4",
  "--contract-adversarial-frac", "0.3", "--contract-max-tokens", "96",
  "--val-every", "500", "--val-batches", "8", "--log-every", "25",
  "--ckpt-every-min", "10", "--seed", "1337", "--threads", "4",
  "--run-name", "xiiv-700m-gn", "--dataset-id", "data_32k-322916b"
)

# ---------- 4. VRAM smoke test (real measurement, 20 steps) ----------
if (-not $SkipSmoke) {
  Write-Host "== [4/5] VRAM smoke test ==" -ForegroundColor Cyan
  # rebuild cleanly: same config, probe ckpt dir, 20 steps
  $ProbeArgs = @(
    "-m", "xiiv.train",
    "--data", $DataDir, "--tokenizer", $TokFile, "--ckpt-dir", "checkpoints-700m-probe",
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
  Remove-Item -Recurse -Force "checkpoints-700m-probe" -ErrorAction SilentlyContinue
  $ans = Read-Host "  Start the full run to step $TargetStep with these settings? [Y/n]"
  if ($ans -match "^[nN]") { Write-Host "Aborted by user."; exit 0 }
}

# ---------- 5. training (crash-resilient loop) ----------
Write-Host "== [5/5] training ==" -ForegroundColor Cyan
Write-Host "  logs -> $CkptDir/train.log | target step $TargetStep"
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
