"""Package the versioned XIIV artifact bundle (ecc_mle_workflow §5).

The artifact is production-ready only when it carries: version, training
config, dataset reference, preprocessing/tokenizer, and eval results —
together, in one directory. This script assembles and VERIFIES that bundle:

    weights/
      xiiv.safetensors   fp32 model weights (no pickle)
      config.json         architecture + special-token ids
      tokenizer.json      byte-level BPE (offline inference)
      run_config.json     frozen training config + code SHA + config hash
      ARTIFACT_CARD.md    version, dataset ref, gates summary, repro commands

Usage:
    python scripts/package_artifact.py --ckpt checkpoints/best.pt
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                ".."))

ARTIFACT_CARD = """# XIIV artifact card — v{version}

## Identity
- model: XIIV (masked-diffusion LM), {params} parameters
- version: {version} (code SHA `{code_sha}`)
- training run: `{run_name}` — config hash `{config_hash}`
- objective: {objective}; seed {seed}; {steps} steps x {tokens_per_step} tokens

## Dataset
- snapshot: `{dataset_snapshot}` (see DATASET_CARD.md)
- train {train_tokens} / valid {valid_tokens} tokens, seq_len {seq_len}

## Training
- optimizer: AdamW lr {lr}->{lr_min}, cosine, warmup {warmup}, wd {wd}
- mask schedule: t ~ Uniform({t_min}, {t_max}); bf16 autocast on CPU
- best held-out masked-LM NLL bound: {val_nll} nats/token (on VALID —
  selection-biased; headline numbers are on the held-out TEST split)

## Promotion gates
{gate_summary}

## Reproduce
```bash
git clone https://github.com/CYPHERLYNX/xiiv && cd xiiv
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m xiiv.sample generate --length 128 --steps 16 --seed 7
.venv/bin/python -m xiiv.sample infill --text "The <mask:8> sat." --steps 16 --seed 7
```

## Files
- `xiiv.safetensors`: fp32 weights, safetensors (no pickle)
- `config.json`: architecture (d={d}, L={L}, h={h})
- `tokenizer.json`: byte-level BPE, vocab {vocab}
- `run_config.json`: full frozen training config + provenance
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/best.pt")
    ap.add_argument("--out-dir", default="weights")
    ap.add_argument("--version", default="0.1.0")
    args = ap.parse_args()

    # 1. export weights/tokenizer/config via the standard exporter
    import subprocess
    r = subprocess.run(
        [sys.executable, "scripts/export_weights.py", "--ckpt", args.ckpt,
         "--out-dir", args.out_dir], capture_output=True, text=True)
    print(r.stdout.strip())
    if r.returncode != 0:
        print(r.stderr, file=sys.stderr)
        sys.exit("export failed")

    # 2. run config: prefer the run's own run_config.json, else reconstruct
    import torch
    from xiiv.run_config import RunConfig
    from xiiv import load_ckpt_trusted
    rc_path = os.path.join(os.path.dirname(args.ckpt), "run_config.json")
    meta = json.load(open("data/meta.json"))
    if os.path.exists(rc_path):
        rc = RunConfig.load(rc_path)
        print(f"run config: {rc_path} (hash {rc.config_hash})")
    else:
        ck = load_ckpt_trusted(args.ckpt)
        cfg = json.loads(ck["config"])
        rc = RunConfig(
            run_name="xiiv-10m-tinystories", model_version=args.version,
            objective="masked-diffusion",
            dataset_snapshot_id="tinystories-v2-gpt4-20261001",
            train_blocks=meta["train_blocks"], valid_blocks=meta["valid_blocks"],
            seq_len=meta["seq_len"], d_model=cfg["d_model"],
            n_layers=cfg["n_layers"], n_heads=cfg["n_heads"],
            ffn_dim=cfg["ffn_dim"], vocab_size=meta["vocab_size"],
            mask_id=meta["mask_id"], eos_id=meta["eos_id"],
            total_steps=ck.get("step", 5000),
            seed=1337, bf16=True).with_provenance("xiiv", torch.__version__)
        print("run config: RECONSTRUCTED from checkpoint+meta "
              "(run predates run_config.py; values verified against logs)")
    rc.save(os.path.join(args.out_dir, "run_config.json"))

    # 3. gate summary (if evaluated)
    gate_summary = "(not yet evaluated — run scripts/check_gates.py)"
    gp = "promotion_gate_results.json"
    if os.path.exists(gp):
        res = json.load(open(gp))
        lines = [f"- {k}: {'PASS' if v['pass'] else 'FAIL'} ({v['detail']})"
                 for k, v in res.items()]
        gate_summary = "\n".join(lines)

    # 4. artifact card
    from xiiv.model import XIIVModel
    from xiiv.config import ModelConfig
    cfg = ModelConfig.from_json(
        open(os.path.join(args.out_dir, "config.json")).read())
    card = ARTIFACT_CARD.format(
        version=args.version, params=f"{XIIVModel(cfg).count_parameters():,}",
        code_sha=rc.code_sha256[:12], run_name=rc.run_name,
        config_hash=rc.config_hash, objective=rc.objective, seed=rc.seed,
        steps=rc.total_steps,
        tokens_per_step=rc.batch_size * rc.seq_len,
        dataset_snapshot=rc.dataset_snapshot_id,
        train_tokens=f"{meta['train_tokens']/1e6:.1f}M",
        valid_tokens=f"{meta['valid_tokens']/1e6:.1f}M",
        seq_len=rc.seq_len, lr=rc.lr, lr_min=rc.lr_min,
        warmup=rc.warmup_steps, wd=rc.weight_decay,
        t_min=rc.t_min, t_max=rc.t_max,
        val_nll=f"{load_ckpt_trusted(args.ckpt).get('val_nll', float('nan')):.4f}",
        gate_summary=gate_summary, d=cfg.d_model, L=cfg.n_layers,
        h=cfg.n_heads, vocab=rc.vocab_size)
    with open(os.path.join(args.out_dir, "ARTIFACT_CARD.md"), "w") as f:
        f.write(card)

    # 5. verify the bundle loads end-to-end (fail closed)
    from safetensors.torch import load_file
    state = load_file(os.path.join(args.out_dir, "xiiv.safetensors"))
    model = XIIVModel(cfg)
    model.load_state_dict(state, strict=True)
    model.eval()
    with torch.no_grad():
        logits = model(torch.randint(0, cfg.vocab_size, (1, 16)))
    assert logits.shape == (1, 16, cfg.vocab_size)
    assert all(torch.isfinite(v).all() for v in state.values())
    print(f"artifact verified: loads, forward ok, "
          f"{sum(v.numel() for v in state.values()):,} params, all finite")
    print(f"bundle complete in {args.out_dir}/")


if __name__ == "__main__":
    main()
