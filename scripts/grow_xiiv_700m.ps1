<#
.SYNOPSIS
  Step 0 of the XIIV-700M rung: grow the 133M parent checkpoint to 700M on CPU.
.DESCRIPTION
  Grows transfer/xiiv-133m-gn-step8218-weights.pt (133.52M, step 8218,
  optimizer stripped for transfer — growth never reads optimizer state)
  into checkpoints_700m_gn/ckpt_8218.pt (697.93M) via scripts/grow_checkpoint.py.

  Frozen growth config (do not hand-tune; see MODEL_CARD_700M.md):
    d 704->1152, layers 15->30, heads 11->18, ffn 2048->3200,
    vocab 8192->32768 (vocab map: weights/vocab_map_8k_32k.json),
    drift gate mean|d| < 0.20 nats, noise 1%, seed 0.

  Growth runs on CPU (~4-6 GB RAM peak, a few minutes). After this,
  run train_xiiv_700m_gpu.ps1 for the GPU training phase.

  Safe to re-run: skips if the grown checkpoint already exists (use -Force to redo).
.EXAMPLE
  .\grow_xiiv_700m.ps1
#>
param(
  [switch]$Force
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

$ParentCkpt = "transfer/xiiv-133m-gn-step8218-weights.pt"
$GrownCkpt  = "checkpoints_700m_gn/ckpt_8218.pt"
$VocabMap   = "weights/vocab_map_8k_32k.json"
$GrowScript = "scripts/grow_checkpoint.py"

function Fail($msg) {
  Write-Host ""
  Write-Host "ERROR: $msg" -ForegroundColor Red
  exit 1
}

Write-Host "== [1/4] prerequisites ==" -ForegroundColor Cyan
try { $py = (Get-Command python).Source } catch { Fail "Python not found. Install Python 3.10+ from python.org and retry." }
Write-Host "  $(& python --version 2>&1)"
foreach ($p in @($ParentCkpt, $VocabMap, $GrowScript)) {
  if (-not (Test-Path $p)) { Fail "missing $p — unzip the full 700M kit into this folder first." }
  Write-Host "  found $p"
}
if ((Test-Path $GrownCkpt) -and (-not $Force)) {
  Write-Host "  grown checkpoint already exists: $GrownCkpt" -ForegroundColor Green
  Write-Host "  nothing to do. Run train_xiiv_700m_gpu.ps1 next, or re-run with -Force to regrow."
  exit 0
}

Write-Host "== [2/4] environment (CPU torch is enough for growth) ==" -ForegroundColor Cyan
if (-not (Test-Path ".venv-grow")) {
  Write-Host "  creating .venv-grow ..."
  & python -m venv .venv-grow
}
$VenvPy = Join-Path $Root ".venv-grow\Scripts\python.exe"
try { & $VenvPy -c "import torch; print('  torch', torch.__version__)" 2>$null }
catch {
  Write-Host "  installing CPU torch (one-time, a few minutes) ..."
  & $VenvPy -m pip install --upgrade pip | Out-Null
  & $VenvPy -m pip install torch --index-url https://download.pytorch.org/whl/cpu
  if ($LASTEXITCODE -ne 0) { Fail "pip install failed. Check your network and retry." }
}

Write-Host "== [3/4] growing 133M -> 700M (frozen config) ==" -ForegroundColor Cyan
if (-not (Test-Path "checkpoints_700m_gn")) { New-Item -ItemType Directory "checkpoints_700m_gn" | Out-Null }
$GrowArgs = @(
  "scripts/grow_checkpoint.py",
  "--src-ckpt", $ParentCkpt,
  "--out-ckpt", $GrownCkpt,
  "--d-new", "1152", "--layers-new", "30", "--heads-new", "18",
  "--ffn-new", "3200",
  "--vocab-size-new", "32768", "--vocab-map", $VocabMap,
  "--noise-scale", "0.01", "--seed", "0"
)
& $VenvPy @GrowArgs
if ($LASTEXITCODE -ne 0) { Fail "growth failed or drift gate tripped — see output above. Do NOT train; report the drift numbers." }

Write-Host "== [4/4] verify ==" -ForegroundColor Cyan
if (-not (Test-Path $GrownCkpt)) { Fail "grow script exited 0 but $GrownCkpt is missing." }
$sizeGB = [math]::Round((Get-Item $GrownCkpt).Length / 1GB, 2)
Write-Host "  grown checkpoint: $GrownCkpt ($sizeGB GB)" -ForegroundColor Green
Write-Host ""
Write-Host "Growth complete. Copy the drift numbers printed above (max|d|, mean|d|)" -ForegroundColor Yellow
Write-Host "into MODEL_CARD_700M.md's verification section, then run:"
Write-Host "  .\train_xiiv_700m_gpu.ps1"
