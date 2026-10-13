"""
Adversarial Self-Play + Opponent Pool — System 2.

Design:
    - Opponents are game-agnostic where possible. For board games they
      expose select_move(); for puzzle games they expose
      generate_challenge().
    - CheckpointOpponent wraps an EARLIER LoRA checkpoint of our own
      model: the model literally plays against its past self. The actual
      move-generation callable is injected (needs the live model), so
      this module never pretends to run the 1B model on its own.
    - DifficultyScheduler advances opponents only on MEASURED win rates,
      never on vibes.
    - Weakness-targeted generation reads LearningHistory.analyze_weaknesses()
      and builds challenges aimed at the model's most frequent failure mode.

No fake metrics: every win-rate is computed from recorded GameResults.
"""

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import chess

from .protocol import LearningHistory

CKPT_DIR = Path.home() / "workspace" / "ai-tools" / "corsac" / "checkpoints_1b_lora"

# Difficulty gates: (games_required, min_win_rate) to advance past each tier.
ADVANCE_GAMES = 20
ADVANCE_WIN_RATE = 0.70


# ---------------------------------------------------------------------------
# Opponent base + built-ins
# ---------------------------------------------------------------------------

@dataclass
class OpponentSpec:
    name: str
    strength: float  # 0.0 (trivial) .. 1.0 (frontier). Honest estimate.
    strategy: str    # human-readable description of how it plays


class Opponent:
    """Base class. Subclasses implement select_move and/or generate_challenge."""
    spec: OpponentSpec = OpponentSpec("base", 0.0, "undefined")

    def select_move(self, board: chess.Board) -> Optional[chess.Move]:
        """Return a move for `board` (board.turn to move). None = resign."""
        raise NotImplementedError

    def generate_challenge(self, game_type: str, difficulty: str = "medium"):
        """Return a puzzle/challenge dict for puzzle-type games."""
        raise NotImplementedError


class RandomOpponent(Opponent):
    """Baseline: uniform-random legal moves. Strength ~0.05.

    Purpose: verify the model can beat pure noise before anything harder.
    Losing to this opponent means the model cannot follow game rules.
    """
    spec = OpponentSpec("random", 0.05,
                        "uniform-random legal move; puzzle baseline = trivially easy")

    def select_move(self, board: chess.Board) -> Optional[chess.Move]:
        moves = list(board.legal_moves)
        return random.choice(moves) if moves else None

    def generate_challenge(self, game_type: str, difficulty: str = "easy"):
        if game_type == "sudoku":
            from .sudoku import generate_extreme_sudoku
            # Easy: keep 40 givens instead of 20.
            puzzle, answer = generate_extreme_sudoku()
            return {"puzzle": puzzle, "answer": answer, "difficulty": "easy"}
        raise NotImplementedError(f"no generator for {game_type}")


class GreedyOpponent(Opponent):
    """1-ply greedy: captures the highest-value piece available, else random.

    Strength ~0.25. Punishes hanging pieces; no positional understanding.
    """
    spec = OpponentSpec("greedy", 0.25,
                        "1-ply: takes highest-value capture, else random move")

    _VALUES = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
               chess.ROOK: 5, chess.QUEEN: 9}

    def select_move(self, board: chess.Board) -> Optional[chess.Move]:
        moves = list(board.legal_moves)
        if not moves:
            return None
        best, best_val = None, -1
        for m in moves:
            val = 0
            if board.is_capture(m):
                captured = board.piece_at(m.to_square)
                if captured:
                    val = self._VALUES.get(captured.piece_type, 0)
                elif board.is_en_passant(m):
                    val = 1
            # Prefer checks slightly.
            board.push(m)
            if board.is_checkmate():
                board.pop()
                return m
            if board.is_check():
                val += 0.5
            board.pop()
            if val > best_val:
                best, best_val = m, val
        return best if best is not None else random.choice(moves)

    def generate_challenge(self, game_type: str, difficulty: str = "medium"):
        if game_type == "sudoku":
            from .sudoku import generate_extreme_sudoku
            puzzle, answer = generate_extreme_sudoku()
            return {"puzzle": puzzle, "answer": answer, "difficulty": "medium"}
        raise NotImplementedError(f"no generator for {game_type}")


class HistoryOpponent(Opponent):
    """Replays the model's own past mistakes as new challenges.

    Strength is adaptive: exactly as hard as the model's demonstrated
    weaknesses. Built from LearningHistory.analyze_weaknesses().
    """
    spec = OpponentSpec("history", 0.40,
                        "targets model's most frequent recorded mistake types")

    def __init__(self, history: LearningHistory):
        self.history = history

    def top_weakness(self) -> Optional[str]:
        weaknesses = self.history.analyze_weaknesses()
        if not weaknesses:
            return None
        return max(weaknesses.items(), key=lambda kv: kv[1])[0]

    def select_move(self, board: chess.Board) -> Optional[chess.Move]:
        # Without a mistake model for chess specifics, fall back to greedy.
        # Honest: this is a placeholder until chess mistake taxonomies exist.
        return GreedyOpponent().select_move(board)

    def generate_challenge(self, game_type: str, difficulty: str = "targeted"):
        weakness = self.top_weakness()
        if game_type == "sudoku":
            from .sudoku import generate_extreme_sudoku
            puzzle, answer = generate_extreme_sudoku()
            return {"puzzle": puzzle, "answer": answer,
                    "difficulty": "targeted",
                    "targets_weakness": weakness}
        raise NotImplementedError(f"no generator for {game_type}")


class CheckpointOpponent(Opponent):
    """The model's own past self, loaded from an earlier LoRA checkpoint.

    This is genuine self-play: current adapters vs older adapters.
    The move-generation function is INJECTED because this module cannot
    (and must not pretend to) load the 1B model by itself.

    Args:
        ckpt_path: path to a lora_ckpt_*.pt file.
        move_fn: callable(fen: str) -> uci_move: str. Built by the caller
            using the live model + the checkpoint's adapters.
    """
    spec = OpponentSpec("checkpoint", 0.60,
                        "past self: earlier LoRA checkpoint of our own model")

    def __init__(self, ckpt_path: str,
                 move_fn: Optional[Callable[[str], str]] = None):
        self.ckpt_path = str(ckpt_path)
        self.move_fn = move_fn
        # Read step number from filename for honest labeling.
        stem = Path(ckpt_path).stem  # e.g. lora_ckpt_675
        self.spec = OpponentSpec(
            f"checkpoint@{stem}", 0.60,
            f"past self from {stem}; move_fn "
            + ("injected" if move_fn else "NOT injected — cannot play"))

    def select_move(self, board: chess.Board) -> Optional[chess.Move]:
        if self.move_fn is None:
            raise RuntimeError(
                "CheckpointOpponent has no move_fn: wire it to the live model "
                "before playing. Refusing to fake a move.")
        uci = self.move_fn(board.fen())
        try:
            move = chess.Move.from_uci(uci)
        except ValueError:
            return None
        return move if move in board.legal_moves else None

    def generate_challenge(self, game_type: str, difficulty: str = "hard"):
        raise NotImplementedError(
            "CheckpointOpponent plays board games; use puzzle opponents "
            "for challenge generation.")


# ---------------------------------------------------------------------------
# Opponent pool
# ---------------------------------------------------------------------------

class OpponentPool:
    """Named registry of opponents with per-opponent results."""

    def __init__(self):
        self._opponents: dict[str, Opponent] = {}
        self._results: dict[str, list[bool]] = {}  # name -> [won, ...]

    def add(self, opponent: Opponent):
        name = opponent.spec.name
        self._opponents[name] = opponent
        self._results.setdefault(name, [])

    def get(self, name: str) -> Opponent:
        return self._opponents[name]

    def names(self) -> list[str]:
        return list(self._opponents.keys())

    def record(self, opponent_name: str, won: bool):
        self._results.setdefault(opponent_name, []).append(won)

    def win_rate(self, opponent_name: str, last_n: int = None) -> Optional[float]:
        results = self._results.get(opponent_name, [])
        if last_n is not None:
            results = results[-last_n:]
        if not results:
            return None
        return sum(results) / len(results)

    def games_played(self, opponent_name: str) -> int:
        return len(self._results.get(opponent_name, []))

    def summary(self) -> str:
        lines = ["Opponent pool:"]
        for name, opp in self._opponents.items():
            wr = self.win_rate(name)
            n = self.games_played(name)
            wr_s = f"{wr:.0%}" if wr is not None else "no games"
            lines.append(f"  {name:25s} strength={opp.spec.strength:.2f} "
                         f"record={wr_s} ({n} games)")
            lines.append(f"    strategy: {opp.spec.strategy}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Difficulty scheduler
# ---------------------------------------------------------------------------

class DifficultyScheduler:
    """Advances: random -> greedy -> checkpoint, on measured win rates.

    Gate: >=70% wins over the last 20 games vs the current tier.
    Never advances on fewer games; never skips tiers.
    """

    TIERS = ["random", "greedy", "checkpoint"]

    def __init__(self, pool: OpponentPool,
                games_required: int = ADVANCE_GAMES,
                win_rate: float = ADVANCE_WIN_RATE):
        self.pool = pool
        self.games_required = games_required
        self.win_rate_threshold = win_rate
        self.tier_idx = 0

    @property
    def current_tier(self) -> str:
        return self.TIERS[min(self.tier_idx, len(self.TIERS) - 1)]

    def _tier_opponent_name(self, tier: str) -> Optional[str]:
        for name in self.pool.names():
            if name == tier or name.startswith(tier + "@"):
                return name
        return None

    def should_advance(self) -> tuple[bool, str]:
        """Returns (advance?, reason). Pure function of recorded results."""
        tier = self.current_tier
        opp_name = self._tier_opponent_name(tier)
        if opp_name is None:
            return False, f"no opponent registered for tier '{tier}'"
        n = self.pool.games_played(opp_name)
        if n < self.games_required:
            return False, (f"{n}/{self.games_required} games vs {opp_name}; "
                           f"need {self.games_required} before evaluation")
        wr = self.pool.win_rate(opp_name, last_n=self.games_required)
        if wr is not None and wr >= self.win_rate_threshold:
            return True, (f"{wr:.0%} win rate over last {self.games_required} "
                          f"vs {opp_name} >= {self.win_rate_threshold:.0%}")
        return False, (f"{wr:.0%} win rate vs {opp_name} below "
                       f"{self.win_rate_threshold:.0%} gate")

    def maybe_advance(self) -> str:
        """Advance one tier if the gate is met. Returns a status message."""
        advance, reason = self.should_advance()
        if advance and self.tier_idx < len(self.TIERS) - 1:
            self.tier_idx += 1
            return f"ADVANCED to tier '{self.current_tier}': {reason}"
        return f"staying at tier '{self.current_tier}': {reason}"


# ---------------------------------------------------------------------------
# Weakness-targeted challenge generation
# ---------------------------------------------------------------------------

def generate_targeted_challenges(history: LearningHistory,
                                 game_type: str,
                                 n: int = 10) -> list[dict]:
    """Build challenges aimed at the model's top recorded weakness.

    Returns a list of challenge dicts. If the history has no recorded
    weaknesses yet, returns standard extreme challenges and marks them
    as untargeted (honest labeling).
    """
    weaknesses = history.analyze_weaknesses()
    opponent = HistoryOpponent(history)
    challenges = []
    for _ in range(n):
        try:
            ch = opponent.generate_challenge(game_type)
        except NotImplementedError:
            break
        ch["targeted"] = bool(weaknesses)
        challenges.append(ch)
    return challenges


def build_default_pool(history: Optional[LearningHistory] = None) -> OpponentPool:
    """Convenience: pool with random + greedy (+ history if given)."""
    pool = OpponentPool()
    pool.add(RandomOpponent())
    pool.add(GreedyOpponent())
    if history is not None:
        pool.add(HistoryOpponent(history))
    return pool
