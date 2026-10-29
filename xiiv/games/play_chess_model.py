"""Play chess with the ASSEMBLED XIIV-1B as White vs a system opponent.

Caller glue for the Adaptive protocol (2026-10-13 design). This module does
NOT invent a framework: it wires the live model into the documented
injection point (a move-generation callable) and records every game with
the system's own LearningHistory + OpponentPool + rewards.

Model: checkpoints_1b_sgbd/xiiv_1b_sgbd.safetensors ONLY (assembled,
contract off, tied). NEVER the untrained parent ckpt_8218.safetensors.

Rules (system-defined, rewards.py):
  - One model attempt per turn. Illegal or unparseable move = game lost
    immediately (reward_chess_game gives 0.0 for a model illegal move).
  - Only a real checkmate of the opponent counts as won=True. Draws,
    ply-cap endings, and material leads are NOT wins.
  - Opponent tiers (opponents.py): random (0.05) -> greedy (0.25) ->
    checkpoint (0.60). Starting tier is random; the scheduler gate
    (70% over 20 games) governs tier advance, and the protocol's 9/10
    rule governs game advance. Neither is bypassed here.

Usage:
  probe:  .venv/bin/python -m xiiv.games.play_chess_model probe
  games:  .venv/bin/python -m xiiv.games.play_chess_model play --games N
          [--seed-base S] [--steps S] [--gen-tokens N] [--max-plies P]
"""

import argparse
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import chess
import torch

CORSAC = Path.home() / "workspace" / "ai-tools" / "corsac"
sys.path.insert(0, str(CORSAC))

from xiiv.diffusion import sample_trajectory  # noqa: E402
from xiiv.sample import load_model  # noqa: E402
from xiiv.tokenizer import XIIVTokenizer  # noqa: E402

from xiiv.games.protocol import LearningHistory, GameResult  # noqa: E402
from xiiv.games.opponents import (  # noqa: E402
    OpponentPool, RandomOpponent, GreedyOpponent, DifficultyScheduler,
)
from xiiv.games import rewards as R  # noqa: E402

WEIGHTS = CORSAC / "checkpoints_1b_sgbd" / "xiiv_1b_sgbd.safetensors"
TOKENIZER = CORSAC / "weights" / "tokenizer_32k.json"
GOAL_HIDDEN = (Path.home() / "workspace" / "goals" /
               "xiiv-adaptive-intelligence-training-system-chess-self-play-games" /
               "hidden_files")
GOAL_HIDDEN.mkdir(parents=True, exist_ok=True)
RUN_LOG = GOAL_HIDDEN / "adaptive_run_2026-10-28.jsonl"

UCI_RE = re.compile(r"\b([a-h][1-8][a-h][1-8][qrbn]?)\b")


def build_prompt(board: chess.Board, move_history: list, style: int) -> str:
    fen = board.fen()
    hist = " ".join(move_history) if move_history else "none"
    if style == 1:
        return (f"Chess position (FEN): {fen} White to move. "
                f"Best move in UCI notation:")
    return (f"Chess game. Moves so far: {hist}. "
            f"Position (FEN): {fen} White to move. Move:")


def model_move(model, tok, board: chess.Board, move_history: list,
               seed: int, steps: int = 16, gen_tokens: int = 24,
               temperature: float = 0.8, style: int = 1):
    """One model attempt at a move. Returns dict with raw text + verdict."""
    t0 = time.time()
    prompt = build_prompt(board, move_history, style)
    prompt_ids = torch.tensor(tok.encode(prompt), dtype=torch.long)
    length = len(prompt_ids) + gen_tokens
    ids, _conf = sample_trajectory(
        model, length, tok.mask_id, tok.eos_id, steps=steps,
        temperature=temperature, top_p=0.95, seed=seed,
        prompt_ids=prompt_ids, remask_ratio=0.0,
        return_trajectory=False, device="cpu", adaptive_decode=False,
        guarantee=None)
    cont_ids = ids[len(prompt_ids):].tolist()
    raw = tok.decode(cont_ids)
    m = UCI_RE.search(raw)
    uci = m.group(1) if m else None
    legal = False
    reward = 0.0
    if uci is not None:
        try:
            mv = chess.Move.from_uci(uci)
            legal = mv in board.legal_moves
        except ValueError:
            legal = False
        reward = R.reward_chess_move(board.fen(), uci, player_color=chess.WHITE)
    return {
        "prompt": prompt,
        "raw_continuation": raw,
        "uci": uci,
        "legal": legal,
        "reward": reward,
        "seconds": round(time.time() - t0, 2),
    }


def play_game(model, tok, pool: OpponentPool, opponent_name: str,
              history: LearningHistory, game_no: int, seed_base: int,
              steps: int, gen_tokens: int, max_plies: int):
    """Play one game, model = White. Returns the recorded GameResult."""
    board = chess.Board()
    opponent = pool.get(opponent_name)
    moves: list = []
    move_log: list = []
    mistakes: list = []
    t0 = time.time()
    result_note = ""
    won = False
    for ply in range(max_plies):
        if board.is_game_over():
            break
        if board.turn == chess.WHITE:
            attempt = model_move(
                model, tok, board, moves, seed=seed_base + game_no * 1000 + ply,
                steps=steps, gen_tokens=gen_tokens)
            attempt["ply"] = ply
            attempt["fen_before"] = board.fen()
            move_log.append(attempt)
            if not attempt["legal"]:
                mistakes.append(
                    f"ply {ply}: illegal/unparseable model move "
                    f"(raw={attempt['raw_continuation']!r})")
                result_note = (f"model illegal/unparseable move at ply {ply}: "
                               f"uci={attempt['uci']} raw={attempt['raw_continuation']!r}")
                break
            mv = chess.Move.from_uci(attempt["uci"])
            moves.append(attempt["uci"])
            board.push(mv)
        else:
            mv = opponent.select_move(board)
            if mv is None:
                result_note = "opponent resigned (no move)"
                won = True
                break
            moves.append(mv.uci())
            board.push(mv)
    else:
        result_note = f"ply cap {max_plies} reached, game unfinished"
    if board.is_checkmate():
        # Side to move is mated. Model wins only if Black is mated.
        won = (board.turn == chess.BLACK)
        result_note = f"checkmate: {'model wins' if won else 'model mated'}"
    elif board.is_stalemate():
        result_note = "stalemate (draw, not a win)"
    elif board.is_insufficient_material():
        result_note = "insufficient material (draw, not a win)"
    elif not result_note:
        result_note = f"game over: {board.result()} (not a model win)"
    dur = time.time() - t0
    res = GameResult(
        game="chess", difficulty="random" if opponent_name == "random" else opponent_name,
        opponent=opponent_name, won=won, moves=len(moves),
        duration_secs=round(dur, 2), timestamp=datetime.now(timezone.utc).isoformat(),
        notes=result_note, mistakes=mistakes,
        key_decisions=[f"model_legal_moves={sum(1 for a in move_log if a['legal'])}/{len(move_log)}"])
    history.record(res)
    pool.record(opponent_name, won)
    with open(RUN_LOG, "a") as f:
        f.write(json.dumps({
            "game_no": game_no, "opponent": opponent_name, "won": won,
            "moves_uci": moves, "duration_secs": round(dur, 2),
            "notes": result_note, "move_log": move_log,
            "final_fen": board.fen(), "final_result": board.result(),
        }) + "\n")
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["probe", "play"])
    ap.add_argument("--games", type=int, default=1)
    ap.add_argument("--seed-base", type=int, default=7)
    ap.add_argument("--steps", type=int, default=16)
    ap.add_argument("--gen-tokens", type=int, default=24)
    ap.add_argument("--max-plies", type=int, default=200)
    args = ap.parse_args()

    assert WEIGHTS.exists(), f"assembled weights missing: {WEIGHTS}"
    print(f"[load] assembled 1B: {WEIGHTS}", flush=True)
    model, cfg = load_model(str(WEIGHTS), device="cpu")
    tok = XIIVTokenizer(str(TOKENIZER))
    print(f"[load] contract_enabled={cfg.contract_enabled} (must be False)",
          flush=True)

    if args.mode == "probe":
        board = chess.Board()
        for style in (1, 2):
            att = model_move(model, tok, board, [], seed=7,
                             steps=args.steps, gen_tokens=args.gen_tokens,
                             style=style)
            print(f"--- probe style {style} ---")
            print(f"prompt: {att['prompt']}")
            print(f"raw: {att['raw_continuation']!r}")
            print(f"uci={att['uci']} legal={att['legal']} "
                  f"reward={att['reward']} secs={att['seconds']}")
        return 0

    history = LearningHistory("chess")
    pool = OpponentPool()
    pool.add(RandomOpponent())
    pool.add(GreedyOpponent())
    sched = DifficultyScheduler(pool)
    print(f"[start] chess history: {len(history.results)} prior games; "
          f"tier={sched.current_tier}; "
          f"last10={history.recent_record(10)}; target 9/10 met={history.meets_target()}",
          flush=True)
    for g in range(1, args.games + 1):
        res = play_game(model, tok, pool, sched.current_tier, history,
                        g, args.seed_base, args.steps, args.gen_tokens,
                        args.max_plies)
        print(f"[game {g}] won={res.won} moves={res.moves} "
              f"dur={res.duration_secs}s notes={res.notes}", flush=True)
        print(f"[status] last10={history.recent_record(10)} "
              f"target_met={history.meets_target()} "
              f"scheduler={sched.maybe_advance()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
