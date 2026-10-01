<#
.SYNOPSIS
  One-click XIIV GPU training for NVIDIA RTX 4050 (6GB VRAM).
.DESCRIPTION
  Sets up a venv, installs the CUDA build, verifies data, runs a short
  VRAM smoke test with REAL measurements, then launches the full run.
  Safe to re-run: it resumes from the latest checkpoint automatically.
.EXAMPLE
  .\train_xiiv_gpu.ps1
  .\train_xiiv_gpu.ps1 -Preset max -TotalSteps 20000
#>
param(
  [ValidateSet("safe", "fast", "max")]
  [string]$Preset = "fast",
  [int]$TotalSteps = 5000,
  [int]$BatchSize = 0,          # 0 = preset default
  [switch]$SkipSmoke,
  [switch]$NoCheckpointing
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

function Fail($msg) {
  Write-Host ""
  Write-Host "ERROR: $msg" -ForegroundColor Red
  Write-Host "If you are stuck, open an issue at https://github.com/CYPHERLYNX/XIIV"
  exit 1
}

# ---------- 1. prerequisites ----------
Write-Host "== [1/5] prerequisites ==" -ForegroundColor Cyan
try { $py = (Get-Command python).Source } catch { Fail "Python not found. Install Python 3.10+ from python.org and retry." }
$pyv = & python --version 2>&1
Write-Host "  $pyv"
try { nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | ForEach-Object { Write-Host "  GPU: $_" } }
catch { Fail "nvidia-smi failed — is an NVIDIA driver installed?" }

# ---------- 2. venv + deps ----------
Write-Host "== [2/5] environment ==" -ForegroundColor Cyan
if (-not (Test-Path ".venv-gpu")) {
  Write-Host "  creating .venv-gpu ..."
  & python -m venv .venv-gpu
}
$VenvPy = Join-Path $Root ".venv-gpu\Scripts\python.exe"
try { & $VenvPy -c "import torch; assert torch.cuda.is_available()" 2>$null }
catch {
  Write-Host "  installing GPU dependencies (one-time, a few minutes) ..."
  & $VenvPy -m pip install --upgrade pip | Out-Null
  & $VenvPy -m pip install -r requirements-gpu.txt
  if ($LASTEXITCODE -ne 0) { Fail "pip install failed. Check your network and retry." }
}
& $VenvPy -c "import torch; print('  torch', torch.__version__, '| cuda', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0))"

# ---------- 3. data ----------
Write-Host "== [3/5] data ==" -ForegroundColor Cyan
if (-not ((Test-Path "data/train.bin") -and (Test-Path "data/valid.bin") -and (Test-Path "data/meta.json"))) {
  Write-Host "  preparing dataset (one-time, downloads TinyStories) ..."
  & $VenvPy scripts/prepare_data.py
  if ($LASTEXITCODE -ne 0) { Fail "data preparation failed." }
}
& $VenvPy scripts/dataset_audit.py | Select-Object -Last 2

# ---------- 4. preset -> batch size ----------
$presets = @{
  safe = @{ bs = 64;  gc = $true  }
  fast = @{ bs = 128; gc = $true  }
  max  = @{ bs = 256; gc = $true  }
}
if ($BatchSize -le 0) { $BatchSize = $presets[$Preset].bs }
$UseGC = $presets[$Preset].gc -and (-not $NoCheckpointing)
Write-Host "== [4/5] VRAM smoke test ==" -ForegroundColor Cyan
Write-Host "  preset=$Preset batch=$BatchSize grad_checkpoint=$UseGC"

if (-not $SkipSmoke) {
  $gcFlag = if ($UseGC) { "--grad-checkpoint" } else { "" }
  # 20-step probe that prints REAL peak VRAM — no guessing.
  $probe = & $VenvPy -m xiiv.train --data data --ckpt-dir checkpoints-gpu-probe `
    --total-steps 20 --batch-size $BatchSize --device cuda $gcFlag `
    --threads 4 --log-every 20 2>&1 | Out-String
  $vramLine = ($probe -split "`n") | Where-Object { $_ -match "peak_vram|VRAM|vram" } | Select-Object -Last 1
  Write-Host "  probe output tail:"
  ($probe -split "`n" | Select-Object -Last 4) | ForEach-Object { Write-Host "    $_" }
  # ask before committing to the full run
  $ans = Read-Host "  Start the full $TotalSteps-step run with these settings? [Y/n]"
  if ($ans -match "^[nN]") { Write-Host "Aborted by user. Probe checkpoints are in checkpoints-gpu-probe/."; exit 0 }
  Remove-Item -Recurse -Force checkpoints-gpu-probe -ErrorAction SilentlyContinue
}

# ---------- 5. full run ----------
Write-Host "== [5/5] training ==" -ForegroundColor Cyan
$gcFlag = if ($UseGC) { "--grad-checkpoint" } else { "" }
Write-Host "  launching: batch=$BatchSize steps=$TotalSteps device=cuda"
Write-Host "  logs -> train_gpu.log | checkpoints -> checkpoints/ (resumable)"
Write-Host "  NOTE: do NOT combine --grad-checkpoint with --compile (they conflict)."
& $VenvPy -m xiiv.train --data data --ckpt-dir checkpoints `
  --total-steps $TotalSteps --batch-size $BatchSize `
  --device cuda $gcFlag --threads 4 `
  2>&1 | Tee-Object -FilePath train_gpu.log
Write-Host ""
Write-Host "Run finished. Best checkpoint: checkpoints/best.pt" -ForegroundColor Green
Write-Host "Export weights: .\.venv-gpu\Scripts\python.exe scripts/export_weights.py --ckpt checkpoints/best.pt --out-dir weights"
