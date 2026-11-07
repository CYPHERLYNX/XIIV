"""XIIV command-line interface: generate text or fill in masked spans.

Examples:
    python -m xiiv.sample generate --length 128 --steps 16 --seed 7
    python -m xiiv.sample infill --text "Lily found a <mask> in the garden."
    python -m xiiv.sample generate --prompt "Once upon a time" --length 160

Zero network calls. Deterministic given --seed.
"""
import argparse
import json
import os
import sys
import time

import torch

from .config import ModelConfig
from .model import XIIVModel
from .tokenizer import XIIVTokenizer
from .diffusion import sample_trajectory


def _build_guarantee(args, tok):
    """Build the GCDD lookahead for --guarantee-json, else None."""
    if not args.guarantee_json:
        return None
    from .guarantee_decode import GuaranteeLookahead, JsonViabilityChecker
    return GuaranteeLookahead(
        json_checker=JsonViabilityChecker(),
        token_text=lambda tid: tok.decode([int(tid)]),
        mask_id=tok.mask_id)


def load_model(weights_path: str, device="cpu"):
    """Load weights + config. Zero network calls.

    Handles two formats:
      .pt         — training checkpoint: torch.save dict with "model",
                    "config" (ModelConfig JSON), "step".
      .safetensors — export weights with a sibling config.json.
    """
    if weights_path.endswith(".pt"):
        ck = torch.load(weights_path, map_location=device,
                        weights_only=False)
        cfg = ModelConfig.from_json(ck["config"])
        model = XIIVModel(cfg)
        missing, unexpected = model.load_state_dict(ck["model"],
                                                    strict=False)
        assert not unexpected, \
            f"unexpected keys in {weights_path}: {unexpected}"
        assert set(missing) <= {"lm_head.weight"}, \
            f"missing keys in {weights_path}: {missing}"
    else:
        from safetensors.torch import load_file
        cfg_path = os.path.join(
            os.path.dirname(os.path.abspath(weights_path)), "config.json")
        if not os.path.exists(cfg_path):
            raise FileNotFoundError(
                f"no config.json next to {weights_path}; for training "
                f"checkpoints (.pt) the config is embedded — pass the .pt "
                f"path directly, e.g. --weights checkpoints_500m_gn/ckpt_8218.pt")
        cfg = ModelConfig.from_json(open(cfg_path).read())
        state = load_file(weights_path, device=device)
        model = XIIVModel(cfg)
        # strict=False: safetensors stores tied weights once (embed.weight covers
        # lm_head.weight — same tensor object by construction in XIIVModel).
        missing, unexpected = model.load_state_dict(state, strict=False)
        assert not unexpected, \
            f"unexpected keys in {weights_path}: {unexpected}"
        assert set(missing) <= {"lm_head.weight"}, \
            f"missing keys in {weights_path}: {missing}"
    model.to(device).eval()
    return model, cfg


def resolve_device(requested):
    """Resolve a --device request to a concrete torch device string.

    "cpu" -> "cpu". "cuda" -> "cuda", but LOUDLY refuse (SystemExit,
    no fallback) when CUDA is unavailable, so a GPU run can never
    silently degrade to CPU timing presented as GPU timing.
    "auto" -> "cuda" when torch.cuda.is_available(), else "cpu".
    Anything else is a caller error.
    """
    if requested == "cpu":
        return "cpu"
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit(
                "ERROR: --device cuda requested but CUDA is unavailable "
                "(torch.cuda.is_available() == False); refusing to fall "
                "back to CPU")
        return "cuda"
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    raise SystemExit(f"ERROR: unknown --device {requested!r} "
                     "(expected cpu, cuda, or auto)")


def cmd_generate(args, model, cfg, tok):
    from xiiv.diffusion import generate_with_scratchpad, self_refine
    prompt_ids = None
    if args.prompt:
        prompt_ids = torch.tensor(tok.encode(args.prompt), dtype=torch.long)
        if len(prompt_ids) >= args.length:
            raise ValueError(
                f"prompt is {len(prompt_ids)} tokens but --length is only "
                f"{args.length}: the canvas must be longer than the prompt")
        prompt_ids = torch.tensor(tok.encode(args.prompt), dtype=torch.long)
    t0 = time.time()
    scratch_txt, traj = None, None
    used_cgec = False
    used_cdr = False
    used_ptcs = False
    used_scv = False
    used_cbrs = False
    used_ccca = False
    used_cwtc = False
    if args.scratchpad > 0:
        # Two-phase reasoning: scratchpad first, answer conditioned on it.
        ids, conf, scratch = generate_with_scratchpad(
            model, args.length, tok.mask_id, tok.eos_id,
            scratch_len=args.scratchpad, steps=args.steps,
            temperature=args.temperature, top_p=args.top_p, seed=args.seed,
            prompt_ids=prompt_ids, device=args.device)
        scratch_txt = tok.decode(scratch.tolist())
    elif args.refine_rounds > 0:
        # Draft + self-correction: remask lowest-confidence spans,
        # re-denoise in place (diffusion-native revision).
        ids, conf = self_refine(
            model, args.length, tok.mask_id, tok.eos_id, steps=args.steps,
            refine_rounds=args.refine_rounds, remask_frac=args.refine_frac,
            temperature=args.temperature, top_p=args.top_p, seed=args.seed,
            prompt_ids=prompt_ids, device=args.device)
    else:
        if args.dcmr:
            # DCMR path: certificate-gated unmasking with decoupled
            # regeneration. Prompt tokens are frozen (never remasked).
            from .dcmr import DCMRConfig, dcmr_sample
            frozen = {}
            if prompt_ids is not None:
                frozen = {i: int(t)
                          for i, t in enumerate(prompt_ids.tolist())}
            dev = next(model.parameters()).device
            def _dcmr_model(x):
                with torch.no_grad():
                    return model(x.to(dev)).cpu()
            ids, trace = dcmr_sample(
                _dcmr_model, length=args.length, mask_id=tok.mask_id,
                config=DCMRConfig(), seed=args.seed, frozen=frozen)
            conf, traj = None, None
            print(f"[xiiv] dcmr: commits={trace.commits} "
                  f"gate_rejects={trace.gate_rejects} "
                  f"remasks={trace.remasks} regen_passes={trace.regen_passes} "
                  f"final_no_repeat={trace.final_no_repeat}",
                  file=sys.stderr)
        elif args.cdr:
            # CDR path: certificate-drift remasking -- vanilla commits,
            # then every check cycle re-verifies the proof margin
            # (commit-time vs current) and remasks decayed positions.
            # Emits a cdr-v1 certificate per generation.
            from .cdr import CDRConfig, cdr_sample
            frozen = {}
            if prompt_ids is not None:
                frozen = {i: int(t)
                          for i, t in enumerate(prompt_ids.tolist())}
            dev = next(model.parameters()).device

            def _cdr_model(x):
                with torch.no_grad():
                    return model(x.to(dev)).cpu()
            ids, trace, cert = cdr_sample(
                _cdr_model, length=args.length, mask_id=tok.mask_id,
                config=CDRConfig(), seed=args.seed, frozen=frozen)
            conf, traj = None, None
            used_cdr = True
            print(f"[xiiv] cdr: commits={trace.commits} "
                  f"drift_remasks={trace.drift_remasks} "
                  f"stillborn_remasks={trace.stillborn_remasks} "
                  f"regen_passes={trace.regen_passes} "
                  f"probe_passes={trace.probe_passes} "
                  f"forward_passes={trace.forward_passes} "
                  f"final_no_repeat={trace.final_no_repeat} "
                  f"completed={trace.completed}",
                  file=sys.stderr)
            if trace.completed:
                import json as _json
                print("[xiiv] cdr certificate: " + _json.dumps(
                    {k: cert[k] for k in
                     ("signal", "n_drift_remasks", "n_stillborn_remasks",
                      "check_cycles", "final_no_repeat")}),
                    file=sys.stderr)
        elif args.ptcs:
            # PTCS path: planned-trajectory certificate scheduler --
            # global budget planner + calibrated informativeness +
            # verifier-side settling + pre-commit certificate gate.
            # Emits a ptcs-v1 certificate per generation.
            from .ptcs import PTCSConfig, ptcs_sample
            frozen = {}
            if prompt_ids is not None:
                frozen = {i: int(t)
                          for i, t in enumerate(prompt_ids.tolist())}
            dev = next(model.parameters()).device

            def _ptcs_model(x):
                with torch.no_grad():
                    return model(x.to(dev)).cpu()
            ids, trace, cert = ptcs_sample(
                _ptcs_model, length=args.length, mask_id=tok.mask_id,
                config=PTCSConfig(), seed=args.seed, frozen=frozen)
            conf, traj = None, None
            used_ptcs = True
            print(f"[xiiv] ptcs: commits={trace.commits} "
                  f"settles={trace.settles} unsets={trace.unsets} "
                  f"gate_rejects={trace.gate_rejects} "
                  f"est_query_skips={trace.est_query_skips} "
                  f"forward_passes={trace.forward_passes} "
                  f"final_no_repeat={trace.final_no_repeat} "
                  f"completed={trace.completed}",
                  file=sys.stderr)
            if trace.completed:
                import json as _json
                print("[xiiv] ptcs certificate: " + _json.dumps(
                    {k: cert[k] for k in
                     ("n_settles", "n_unsets", "n_gate_rejects",
                      "est_query_skips", "final_no_repeat")}),
                    file=sys.stderr)
        elif args.scv:
            # SCV path: speculative certificate verification -- one
            # forward pass per round drafts top-k candidates per frontier
            # position; the INDEPENDENT certificate verifier arbitrates
            # the speculation and commits the longest certificate-clean
            # prefix (verifier-decided batch size, no fixed budget).
            # Emits a scv-v1 certificate per generation.
            from .scv import SCVConfig, scv_sample
            frozen = {}
            if prompt_ids is not None:
                frozen = {i: int(t)
                          for i, t in enumerate(prompt_ids.tolist())}
            dev = next(model.parameters()).device

            def _scv_model(x):
                with torch.no_grad():
                    return model(x.to(dev)).cpu()
            ids, trace, cert = scv_sample(
                _scv_model, length=args.length, mask_id=tok.mask_id,
                config=SCVConfig(), seed=args.seed, frozen=frozen)
            conf, traj = None, None
            used_scv = True
            print(f"[xiiv] scv: accepts={trace.accepts} "
                  f"rejects={trace.rejects} tail_stops={trace.tail_stops} "
                  f"forward_passes={trace.forward_passes} "
                  f"final_no_repeat={trace.final_no_repeat} "
                  f"completed={trace.completed}",
                  file=sys.stderr)
            if trace.completed:
                import json as _json
                print("[xiiv] scv certificate: " + _json.dumps(
                    {k: cert[k] for k in
                     ("n_accepts", "n_rejects", "n_tail_stops",
                      "forward_passes", "final_no_repeat")}),
                    file=sys.stderr)
        elif args.cbrs:
            # CBRS path: certificate-backed repair scheduling -- fast
            # UNVERIFIED draft (frontier top-1 commits, no certificate
            # claim), then verifier-localized repair: the INDEPENDENT
            # verifier finds every fault window, the fault envelope
            # (windows + halo, frozen excluded) is remasked while the
            # rest of the draft stays frozen, and repair rounds resample
            # the envelope with verifier-gated candidates. The cbrs-v1
            # certificate is a claim about the FINAL canvas only.
            from .cbrs import CBRSConfig, cbrs_sample
            frozen = {}
            if prompt_ids is not None:
                frozen = {i: int(t)
                          for i, t in enumerate(prompt_ids.tolist())}
            dev = next(model.parameters()).device

            def _cbrs_model(x):
                with torch.no_grad():
                    return model(x.to(dev)).cpu()
            ids, trace, cert = cbrs_sample(
                _cbrs_model, length=args.length, mask_id=tok.mask_id,
                config=CBRSConfig(), seed=args.seed, frozen=frozen)
            conf, traj = None, None
            used_cbrs = True
            print(f"[xiiv] cbrs: draft_commits={trace.draft_commits} "
                  f"fault_windows_draft={trace.n_fault_windows_draft} "
                  f"repair_passes={trace.repair_rounds} "
                  f"positions_repaired={trace.n_positions_repaired} "
                  f"rejects={trace.rejects} "
                  f"locality={trace.locality:.2f} "
                  f"forward_passes={trace.forward_passes} "
                  f"final_no_repeat={trace.final_no_repeat} "
                  f"completed={trace.completed}",
                  file=sys.stderr)
            if trace.completed:
                import json as _json
                print("[xiiv] cbrs certificate: " + _json.dumps(
                    {k: cert[k] for k in
                     ("draft_passes", "repair_passes",
                      "n_fault_windows_draft", "n_positions_repaired",
                      "n_rejects", "locality",
                      "forward_passes", "final_no_repeat")}),
                    file=sys.stderr)
        elif args.ccca:
            # CCCA path: cross-chain consensus arbitration -- K parallel
            # chains denoise with different reveal orders; a position
            # reaches the shared canvas only when >= quorum chains agree
            # on the token AND the independent verifier accepts it.
            # Disputed positions are rolled back and re-proposed; lone
            # proposals promote after the confirm window as explicitly
            # non-consensus commits. Emits a ccca-v1 certificate.
            from .ccca import CCCAConfig, ccca_sample
            frozen = {}
            if prompt_ids is not None:
                frozen = {i: int(t)
                          for i, t in enumerate(prompt_ids.tolist())}
            dev = next(model.parameters()).device

            def _ccca_model(x):
                with torch.no_grad():
                    return model(x.to(dev)).cpu()
            ids, trace, cert = ccca_sample(
                _ccca_model, length=args.length, mask_id=tok.mask_id,
                config=CCCAConfig(), seed=args.seed, frozen=frozen)
            conf, traj = None, None
            used_ccca = True
            print(f"[xiiv] ccca: commits={trace.commits} "
                  f"consensus={trace.consensus_commits} "
                  f"nonconsensus={trace.nonconsensus_commits} "
                  f"disagreements={trace.disagreements} "
                  f"rollbacks={trace.rollbacks} "
                  f"rounds={trace.rounds} "
                  f"forward_passes={trace.forward_passes} "
                  f"final_no_repeat={trace.final_no_repeat} "
                  f"completed={trace.completed}",
                  file=sys.stderr)
            if trace.completed:
                import json as _json
                print("[xiiv] ccca certificate: " + _json.dumps(
                    {k: cert[k] for k in
                     ("consensus_rate", "n_consensus_commits",
                      "n_nonconsensus_commits", "nonconsensus_positions",
                      "n_disagreements", "n_rollbacks",
                      "forward_passes", "final_no_repeat")}),
                    file=sys.stderr)
        elif args.cwtc:
            # CWTC path: certificate-weighted trace credit -- a
            # per-position credit ledger. The leading certificate-safe
            # candidate earns credit equal to its probability each step;
            # a position commits on confidence OR on accumulated credit
            # (a token repeatedly endorsed safely across steps). No
            # logits are ever modified or fused. Emits a cwtc-v1
            # certificate.
            from .cwtc import CWTCConfig, cwtc_sample
            frozen = {}
            if prompt_ids is not None:
                frozen = {i: int(t)
                          for i, t in enumerate(prompt_ids.tolist())}
            dev = next(model.parameters()).device

            def _cwtc_model(x):
                with torch.no_grad():
                    return model(x.to(dev)).cpu()
            ids, trace, cert = cwtc_sample(
                _cwtc_model, length=args.length, mask_id=tok.mask_id,
                config=CWTCConfig(), frozen=frozen)
            conf, traj = None, None
            used_cwtc = True
            print(f"[xiiv] cwtc: commits={trace.commits} "
                  f"confidence={trace.confidence_commits} "
                  f"credit_driven={trace.credit_driven_commits} "
                  f"gate_rejects={trace.gate_rejects} "
                  f"steps={trace.steps} "
                  f"forward_passes={trace.forward_passes} "
                  f"final_no_repeat={trace.final_no_repeat} "
                  f"completed={trace.completed}",
                  file=sys.stderr)
            if trace.completed:
                import json as _json
                print("[xiiv] cwtc certificate: " + _json.dumps(
                    {k: cert[k] for k in
                     ("confidence_commits", "credit_driven_commits",
                      "gate_rejects", "forward_passes",
                      "final_no_repeat")}),
                    file=sys.stderr)
        else:
            if args.cgec:
                # CGEC path: certificate-guided early commit over the
                # arith-v1 domain (arithmetic expressions). Emits a
                # machine-checkable certificate per generation, replayable
                # by xiiv/cert_verify.py without model weights.
                from .cgec import (CGECConfig, arithmetic_alphabet_32k,
                                   cgec_sample)
                alphabet = arithmetic_alphabet_32k()
                frozen = {}
                if prompt_ids is not None:
                    toks = prompt_ids.tolist()
                    bad = [t for t in toks if t not in alphabet]
                    if bad:
                        raise ValueError(
                            "--cgec certifies arithmetic expressions only: "
                            f"prompt contains {len(bad)} non-domain token(s) "
                            f"(e.g. id {bad[0]}); use plain generate for "
                            "free text")
                    frozen = {i: int(t) for i, t in enumerate(toks)}
                dev = next(model.parameters()).device

                def _cgec_model(x):
                    with torch.no_grad():
                        return model(x.to(dev)).cpu()
                ids, trace, cert = cgec_sample(
                    _cgec_model, length=args.length, mask_id=tok.mask_id,
                    alphabet=alphabet, config=CGECConfig(), seed=args.seed,
                    frozen=frozen)
                conf, traj = None, None
                used_cgec = True
                print(f"[xiiv] cgec: commits={trace.commits} "
                      f"early_commits={trace.early_commits} "
                      f"gate_rejects={trace.gate_rejects} "
                      f"defers={trace.defers} "
                      f"forward_passes={trace.forward_passes} "
                      f"min_gap={trace.min_gap} "
                      f"grammar_valid={trace.grammar_valid} "
                      f"completed={trace.completed}",
                      file=sys.stderr)
                if trace.completed:
                    import json as _json
                    print("[xiiv] cgec certificate: " + _json.dumps(
                        {k: cert[k] for k in
                         ("domain", "min_gap", "n_commits", "early_commits",
                          "gate_rejects")}), file=sys.stderr)
            else:
                out = sample_trajectory(
                    model, args.length, tok.mask_id, tok.eos_id, steps=args.steps,
                    temperature=args.temperature, top_p=args.top_p, seed=args.seed,
                    prompt_ids=prompt_ids, remask_ratio=args.remask,
                    return_trajectory=args.trajectory, device=args.device,
                    adaptive_decode=args.adaptive_decode,
                    guarantee=_build_guarantee(args, tok))
                if args.trajectory:
                    ids, conf, traj = out
                else:
                    ids, conf = out
    dt = time.time() - t0
    if scratch_txt is not None:
        print(f"[scratchpad] {scratch_txt}\n")
    text = tok.decode(ids.tolist())
    # trim at EOS / trailing masks (already skipped by decode)
    print(text)
    n_tok = int((ids != tok.mask_id).sum())
    mode = ("scratchpad+answer" if scratch_txt is not None
            else f"refine x{args.refine_rounds}" if args.refine_rounds > 0
            else "cdr" if used_cdr
            else "ptcs" if used_ptcs
            else "scv" if used_scv
            else "cbrs" if used_cbrs
            else "ccca" if used_ccca
            else "cwtc" if used_cwtc
            else "cgec" if used_cgec
            else "generate")
    print(f"\n[xiiv] {mode}: {n_tok} tokens, {args.steps} denoise steps, "
          f"{dt:.2f}s ({n_tok / max(dt, 1e-6):.1f} tok/s), seed={args.seed}, "
          f"device={args.device}",
          file=sys.stderr)
    if args.trajectory:
        return text, traj, tok
    return text, None, tok


def cmd_infill(args, model, cfg, tok):
    ids, frozen = tok.encode_with_masks(args.text)
    length = len(ids)
    if args.length and args.length > length:
        pad = args.length - length
        import torch as _t
        ids = _t.cat([ids, _t.full((pad,), tok.mask_id, dtype=_t.long)])
        frozen = _t.cat([frozen, _t.zeros(pad, dtype=_t.bool)])
        length = args.length
    t0 = time.time()
    out_ids, conf = sample_trajectory(
        model, length, tok.mask_id, tok.eos_id, steps=args.steps,
        temperature=args.temperature, top_p=args.top_p, seed=args.seed,
        prompt_ids=ids, frozen=frozen, remask_ratio=args.remask,
        device=args.device, adaptive_decode=args.adaptive_decode,
        guarantee=_build_guarantee(args, tok))
    # re-apply frozen context exactly (safety: sampler never touches it,
    # but belt-and-braces for the demo)
    out_ids[frozen] = ids[frozen]
    dt = time.time() - t0
    text = tok.decode(out_ids.tolist())
    print(text)
    print(f"\n[xiiv] infilled {int((~frozen).sum())} masked tokens in "
          f"{dt:.2f}s, seed={args.seed}, device={args.device}", file=sys.stderr)
    return text


def main():
    ap = argparse.ArgumentParser(prog="xiiv",
                                 description="XIIV masked-diffusion LM")
    ap.add_argument("--weights", default="weights/xiiv.safetensors")
    ap.add_argument("--tokenizer", default="weights/tokenizer.json")
    ap.add_argument("--device", default="auto", choices=["cpu", "cuda", "auto"],
                    help="compute device (before the subcommand). "
                         "auto (default): cuda if available, else cpu. "
                         "cuda: fail loudly if CUDA is unavailable, "
                         "never silently fall back to CPU.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="generate from noise (or a prompt)")
    # SUPPRESS default so a --device given before the subcommand is not
    # overwritten by the subparser default; only an explicit
    # `generate --device ...` sets it here.
    g.add_argument("--device", default=argparse.SUPPRESS,
                   choices=["cpu", "cuda", "auto"])
    g.add_argument("--prompt", default="")
    g.add_argument("--length", type=int, default=128)
    g.add_argument("--steps", type=int, default=16)
    g.add_argument("--temperature", type=float, default=0.8,
                   help="sampling temperature (0.8 measured good)")
    g.add_argument("--top-p", type=float, default=0.9)
    g.add_argument("--remask", type=float, default=0.0)
    g.add_argument("--seed", type=int, default=7)
    g.add_argument("--trajectory", action="store_true")
    g.add_argument("--scratchpad", type=int, default=0,
                   help="reasoning-first decoding: denoise SCRATCHPAD tokens "
                        "as a scratchpad, freeze it, then denoise the answer "
                        "on the full trace (0 = off)")
    g.add_argument("--refine-rounds", type=int, default=0,
                   help="self-correction passes: remask lowest-confidence "
                        "spans and re-denoise in place (0 = off)")
    g.add_argument("--refine-frac", type=float, default=0.15,
                   help="fraction of open positions remasked per refine round")
    g.add_argument("--adaptive-decode", action="store_true",
                   help="GACD: adaptive confidence-parallel unmasking — "
                        "batch size follows a confidence threshold that "
                        "slides with canvas confidence (fast when confident, "
                        "careful when uncertain); contract guarantees unchanged")
    g.add_argument("--guarantee-json", action="store_true",
                   help="GCDD: LAVE-style lookahead-then-verify — veto any "
                        "token commit that provably destroys JSON validity "
                        "or contract satisfiability (CPU-only, opt-in)")
    g.add_argument("--dcmr", action="store_true",
                   help="DCMR: decoupled certificate-margin remasking — "
                        "certifies no repeated 4-gram in the output via a "
                        "per-commit gate + decoupled regeneration passes "
                        "(experimental: tested on scripted models only, "
                        "not yet evaluated on a real checkpoint)")
    g.add_argument("--cgec", action="store_true",
                   help="CGEC: certificate-guided early commit — block-wise "
                        "diffusion decoding with a grammar gate "
                        "(arith-v1: arithmetic expressions) + "
                        "confidence-gap all-in fill, emitting a "
                        "machine-checkable certificate per generation "
                        "(experimental: tested on scripted models only, "
                        "not yet evaluated on a real checkpoint)")
    g.add_argument("--cdr", action="store_true",
                   help="CDR: certificate-drift remasking — vanilla "
                        "confidence commits, then every check cycle "
                        "re-verifies the independent proof margin "
                        "commit-time vs current and remasks decayed "
                        "positions; repairs what confidently-wrong models "
                        "hide from confidence remasking, emitting a "
                        "cdr-v1 certificate per generation "
                        "(experimental: tested on scripted models only, "
                        "not yet evaluated on a real checkpoint)")
    g.add_argument("--ptcs", action="store_true",
                   help="PTCS: planned-trajectory certificate scheduler — "
                        "plans the unmasking trajectory (how many positions "
                        "per step, in what order) from calibrated "
                        "informativeness, settles positions whose "
                        "independent-verifier margin stabilizes "
                        "(SureLock-style compute saving), and gates every "
                        "commit on the no-repeat certificate, emitting a "
                        "ptcs-v1 certificate per generation "
                        "(experimental: tested on scripted models only, "
                        "not yet evaluated on a real checkpoint)")
    g.add_argument("--scv", action="store_true",
                   help="SCV: speculative certificate verification — "
                        "draft top-k candidates per frontier position in "
                        "one forward pass, then let the INDEPENDENT "
                        "certificate verifier arbitrate: commit the "
                        "longest certificate-clean prefix (verifier-"
                        "decided batch size, no fixed budget), emitting "
                        "a scv-v1 certificate per generation "
                        "(experimental: tested on scripted models only, "
                        "not yet evaluated on a real checkpoint)")
    g.add_argument("--cbrs", action="store_true",
                   help="CBRS: certificate-backed repair scheduling — "
                        "fast UNVERIFIED draft (frontier top-1 commits), "
                        "then verifier-localized repair: the independent "
                        "verifier finds every fault window, the fault "
                        "envelope (windows + halo) is remasked while the "
                        "rest of the draft stays frozen, and repair "
                        "rounds resample the envelope with verifier-"
                        "gated candidates; emits a cbrs-v1 certificate "
                        "about the FINAL canvas only "
                        "(experimental: tested on scripted models only, "
                        "not yet evaluated on a real checkpoint)")
    g.add_argument("--ccca", action="store_true",
                   help="CCCA: cross-chain consensus arbitration — "
                        "K parallel chains denoise with different reveal "
                        "orders (confidence / left-to-right / right-to-"
                        "left); a position reaches the shared canvas only "
                        "when >= quorum chains agree AND the independent "
                        "verifier accepts it; disputed positions roll back "
                        "and re-propose, lone proposals promote after the "
                        "confirm window as explicitly non-consensus commits; "
                        "emits a ccca-v1 certificate "
                        "(experimental: tested on scripted models only, "
                        "not yet evaluated on a real checkpoint)")
    g.add_argument("--cwtc", action="store_true",
                   help="CWTC: certificate-weighted trace credit — "
                        "a per-position credit ledger: the leading "
                        "certificate-safe candidate earns credit equal to "
                        "its probability each step (gate failures earn "
                        "zero), and a position commits on confidence OR "
                        "on accumulated credit, so a token repeatedly "
                        "endorsed safely across steps can commit while "
                        "still below the confidence threshold; credit "
                        "decays, so stale endorsement never commits; "
                        "emits a cwtc-v1 certificate "
                        "(experimental: tested on scripted models only, "
                        "not yet evaluated on a real checkpoint)")

    f = sub.add_parser("infill",
                       help="fill <mask> / <mask:N> spans, e.g. --text 'a <mask:5> sat'")
    f.add_argument("--device", default=argparse.SUPPRESS,
                   choices=["cpu", "cuda", "auto"])
    f.add_argument("--text", required=True)
    f.add_argument("--length", type=int, default=0,
                   help="pad total length with extra masks (0 = exact)")
    f.add_argument("--steps", type=int, default=16)
    f.add_argument("--temperature", type=float, default=0.8)
    f.add_argument("--top-p", type=float, default=0.9)
    f.add_argument("--remask", type=float, default=0.0)
    f.add_argument("--seed", type=int, default=7)
    f.add_argument("--adaptive-decode", action="store_true",
                   help="GACD: adaptive confidence-parallel unmasking "
                        "(see generate --help)")
    f.add_argument("--guarantee-json", action="store_true",
                   help="GCDD: LAVE-style lookahead-then-verify — veto any "
                        "token commit that provably destroys JSON validity "
                        "or contract satisfiability (CPU-only, opt-in)")

    args = ap.parse_args()
    # GCDD is wired into the plain generate / infill paths only.
    if (args.guarantee_json and args.cmd == "generate"
            and (args.scratchpad > 0 or args.refine_rounds > 0)):
        ap.error("--guarantee-json is not wired into the --scratchpad / "
                 "--refine-rounds paths yet; use it with plain generate "
                 "or infill")
    if (args.dcmr and args.cmd == "generate"
            and (args.scratchpad > 0 or args.refine_rounds > 0
                 or args.trajectory)):
        ap.error("--dcmr is not wired into the --scratchpad / "
                 "--refine-rounds / --trajectory paths yet; use it with "
                 "plain generate")
    if (args.cgec and args.cmd == "generate"
            and (args.scratchpad > 0 or args.refine_rounds > 0
                 or args.trajectory)):
        ap.error("--cgec is not wired into the --scratchpad / "
                 "--refine-rounds / --trajectory paths yet; use it with "
                 "plain generate")
    if (args.cdr and args.cmd == "generate"
            and (args.scratchpad > 0 or args.refine_rounds > 0
                 or args.trajectory)):
        ap.error("--cdr is not wired into the --scratchpad / "
                 "--refine-rounds / --trajectory paths yet; use it with "
                 "plain generate")
    if (args.ptcs and args.cmd == "generate"
            and (args.scratchpad > 0 or args.refine_rounds > 0
                 or args.trajectory)):
        ap.error("--ptcs is not wired into the --scratchpad / "
                 "--refine-rounds / --trajectory paths yet; use it with "
                 "plain generate")
    if sum([bool(args.dcmr), bool(args.cgec), bool(args.cdr),
            bool(args.ptcs)]) > 1:
        ap.error("--dcmr, --cgec, --cdr and --ptcs are alternative "
                 "samplers; pick one")
    # generate needs a positive canvas; infill allows --length 0 (= exact,
    # no padding — the canvas comes from the masked text itself).
    if args.cmd == "generate" and args.length < 1:
        ap.error("--length must be >= 1")
    if args.cmd == "infill" and args.length < 0:
        ap.error("--length must be >= 0")
    if args.cmd == "generate":
        if args.scratchpad < 0:
            ap.error("--scratchpad must be >= 0")
        if args.scratchpad >= args.length:
            ap.error("--scratchpad must be < --length")
        if args.refine_rounds < 0:
            ap.error("--refine-rounds must be >= 0")
        if not 0.0 <= args.refine_frac <= 1.0:
            ap.error("--refine-frac must be in [0, 1]")
    if args.steps < 1:
        ap.error("--steps must be >= 1")
    # Resolve once, before any file load: --device cuda on a CPU-only
    # box exits here with a loud error instead of running on CPU.
    args.device = resolve_device(args.device)
    tok = XIIVTokenizer(args.tokenizer)
    model, cfg = load_model(args.weights, device=args.device)
    assert tok.mask_id == cfg.mask_id and tok.eos_id == cfg.eos_id, \
        "tokenizer/weights special-token mismatch"
    if tok.vocab_size != cfg.vocab_size:
        ap.error(
            f"tokenizer vocab ({tok.vocab_size}) != model vocab "
            f"({cfg.vocab_size}) — the 500M rung needs "
            f"--tokenizer weights/tokenizer_32k.json")

    if args.cmd == "generate":
        cmd_generate(args, model, cfg, tok)
    else:
        cmd_infill(args, model, cfg, tok)


if __name__ == "__main__":
    main()
