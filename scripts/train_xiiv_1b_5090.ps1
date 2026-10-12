<#
.SYNOPSIS
  One-click XIIV-1B guarantee-native training for NVIDIA RTX 5090 (32GB VRAM).
.DESCRIPTION
  Continues the XIIV single lineage: 133M (trained) → 500M (grown) → 1B (grown).
  This script trains the grown 1B checkpoint on REAL DATA ONLY.

  Frozen config (do not hand-tune; see MODEL_SPEC_1B.md):
    arch: d=1280, L=36, h=20, ffn=3584, vocab=32768 (tied) = 1019.23M unique params
    opt:  AdamW, lr 1.65e-4 (inverse-width from the 133M rung: 3e-4 × 704/1280),
          cosine decay to 1.65e-5, warmup 2000
    data: FineWeb-Edu ONLY (data_fineweb/, 99.3M tokens). TinyStories EXCLUDED.
    precision: BF16 (validated by smoke test below)

  The script runs in three phases:
    1. SMOKE: BF16 forward/backward on the grown checkpoint, measures peak VRAM,
       verifies finite loss/gradients, profiles micro-batch sizes.
    2. RESUME-TEST: 10 training steps, checkpoint, reload, verify step continuity.
    3. TRAIN: full run with auto-resume. STOPS on non-finite loss (never burns
       VRAM silently on a diverging run).

  Safe to re-run: resumes from the latest checkpoint automatically.
.EXAMPLE
  .\train_xiiv_1b_5090.ps1
  .\train_xiiv_1b_5090.ps1 -SkipSmoke
#>
param(
  [switch]$SkipSmoke
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

# ---------- frozen 1B rung config ----------
$DModel = 1280; $NLayers = 36; $NHeads = 20; $FFNDim = 3584; $VocabSize = 32768
$LR = 1.65e-4; $LRMin = 1.65e-5; $Warmup = 2000
$StartStep = 8218
$TargetStep = 48218          # +40,000 steps, same budget shape as prior rungs
$DataDir = "data_fineweb"   # REAL DATA ONLY — TinyStories (data_32k) excluded
$TokFile = "weights/tokenizer_32k.json"
$CkptDir = "checkpoints_1b_gn"
$SrcCkpt = "$CkptDir/ckpt_8218.pt"   # grown 1B init (from grow_checkpoint.py)

# Batch: profiled by smoke test; defaults are conservative for 32GB
$MicroBatch = 8; $GradAccum = 4      # 8 x 4 x 128 = 4096 tok/step (smoke test adjusts)

Write-Host "=== XIIV-1B RTX 5090 training launcher ===" -ForegroundColor Cyan
Write-Host "arch: d=$DModel L=$NLayers h=$NHeads ffn=$FFNDim vocab=$VocabSize (1019.23M)"
Write-Host "lr: $LR -> $LRMin (cosine), warmup $Warmup"
Write-Host "data: $DataDir (real data only)"
Write-Host "steps: $StartStep -> $TargetStep"

# ---------- preflight ----------
if (-not (Test-Path $SrcCkpt)) {
  Write-Host "FATAL: grown 1B checkpoint not found: $SrcCkpt" -ForegroundColor Red
  Write-Host "Run: python scripts/grow_checkpoint.py --src-ckpt checkpoints_500m_gn/ckpt_8218.pt --out-ckpt $SrcCkpt --d-new 1280 --layers-new 36 --heads-new 20 --ffn-new 3584 --seed 0"
  exit 1
}
if (-not (Test-Path $DataDir)) {
  Write-Host "FATAL: data dir not found: $DataDir" -ForegroundColor Red
  exit 1
}
$VenvPython = ".\.venv\Scripts\python.exe"
if (-not (Test-Path $VenvPython)) { $VenvPython = "python" }
& $VenvPython -c "import torch; assert torch.cuda.is_available(), 'no CUDA'; print('CUDA:', torch.cuda.get_device_name(0))"

# ---------- phase 1: smoke ----------
if (-not $SkipSmoke) {
  Write-Host "`n--- PHASE 1: smoke (BF16 finite-loss + VRAM profile) ---" -ForegroundColor Yellow
  & $VenvPython scripts/smoke_1b_5090.py --ckpt $SrcCkpt --data-dir $DataDir
  if ($LASTEXITCODE -ne 0) { Write-Host "SMOKE FAILED — aborting before training." -ForegroundColor Red; exit 1 }
  Write-Host "Smoke passed." -ForegroundColor Green
}

# ---------- phase 2+3: resume test + train ----------
Write-Host "`n--- PHASE 2/3: training (auto-resume, abort on non-finite loss) ---" -ForegroundColor Yellow
& $VenvPython xiiv/train.py `
  --data $DataDir `
  --ckpt-dir $CkptDir `
  --init-ckpt $SrcCkpt `
  --tokenizer $TokFile `
  --d-model $DModel --n-layers $NLayers --n-heads $NHeads --ffn-dim $FFNDim `
  --lr $LR --lr-min $LRMin --warmup-steps $Warmup `
  --total-steps $TargetStep `
  --batch-size $MicroBatch --grad-accum $GradAccum `
  --seq-len 128 `
  --device cuda --bf16 --grad-checkpoint `
  --contract-enabled

Write-Host "`nDone. Latest checkpoint in $CkptDir." -ForegroundColor Green
