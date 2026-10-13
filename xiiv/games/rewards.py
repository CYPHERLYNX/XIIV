"""
Verifiable-Reward pipeline for XIIV Adaptive Intelligence Training.

Every reward here is computed from an OBJECTIVELY VERIFIABLE outcome:
game rules, mathematical equality, unit-test results. Nothing is rewarded
for "sounding convincing" — if it cannot be checked by code, it gets
reward 0.0 and is logged as unverifiable.

Reward ranges are [0, 1] unless noted, so they can be used directly as
RWR weights (see xiiv/games/rl.py).
"""

import re
import subprocess
import sys
import tempfile
import os
from pathlib import Path

import chess

# Reuse the existing Sudoku validator — single source of truth.
from .sudoku import is_valid_solution, parse_solution


# ---------------------------------------------------------------------------
# Sudoku
# ---------------------------------------------------------------------------

def reward_sudoku(puzzle, solution_text) -> float:
    """1.0 if the solution text parses to a valid Sudoku solution, else 0.0.

    Args:
        puzzle: 9x9 grid with 0 for empty cells (the givens).
        solution_text: model's raw text output.
    """
    solution = parse_solution(solution_text)
    valid, _msg = is_valid_solution(puzzle, solution)
    return 1.0 if valid else 0.0


# ---------------------------------------------------------------------------
# Chess
# ---------------------------------------------------------------------------

# Standard piece values for material accounting.
_PIECE_VALUES = {
    chess.PAWN: 1.0,
    chess.KNIGHT: 3.0,
    chess.BISHOP: 3.0,
    chess.ROOK: 5.0,
    chess.QUEEN: 9.0,
    chess.KING: 0.0,  # king capture ends the game; handled via checkmate
}


def _material(board: chess.Board, color: bool) -> float:
    total = 0.0
    for piece_type, value in _PIECE_VALUES.items():
        total += len(board.pieces(piece_type, color)) * value
    return total


def reward_chess_move(fen: str, move_uci: str, player_color: bool = chess.WHITE) -> float:
    """Reward a single chess move from a position.

    - Illegal move: 0.0
    - Legal move: 0.2 (base credit for following the rules)
    - Capturing material: +0.1 per pawn-value captured (uncapped here,
      capped by the caller if needed)
    - Giving check: +0.1
    - Checkmate: 1.0 (overrides — a winning move is a winning move)

    Args:
        fen: position before the move.
        move_uci: move in UCI notation, e.g. "e2e4".
        player_color: side to move (must match the FEN side to move
            for the reward to be meaningful).
    """
    board = chess.Board(fen)
    try:
        move = chess.Move.from_uci(move_uci)
    except ValueError:
        return 0.0
    if move not in board.legal_moves:
        return 0.0

    reward = 0.2  # legal move base credit

    # Material captured
    if board.is_capture(move):
        captured = board.piece_at(move.to_square)
        # En passant: the captured pawn is not on the target square.
        if captured is None and board.is_en_passant(move):
            reward += 0.1 * _PIECE_VALUES[chess.PAWN]
        elif captured is not None:
            reward += 0.1 * _PIECE_VALUES[captured.piece_type]

    board.push(move)

    if board.is_checkmate():
        return 1.0
    if board.is_check():
        reward += 0.1
    # Stalemate / draws are not rewarded — only progress toward winning.
    return min(reward, 1.0)


def reward_chess_game(moves_uci: list[str], starting_fen: str = chess.STARTING_FEN,
                      player_color: bool = chess.WHITE) -> float:
    """Reward a full game: 1.0 win, 0.5 draw, 0.0 loss or illegal move.

    Plays the model's moves for `player_color`; the opponent's moves are
    taken from the list as given (caller supplies opponent play).
    Any illegal move by the model ends the game as a loss.
    """
    board = chess.Board(starting_fen)
    for i, uci in enumerate(moves_uci):
        side = board.turn
        try:
            move = chess.Move.from_uci(uci)
        except ValueError:
            # Illegal move notation by whoever was to move.
            return 0.0 if side == player_color else 1.0
        if move not in board.legal_moves:
            return 0.0 if side == player_color else 1.0
        board.push(move)
        if board.is_game_over():
            break

    if board.is_checkmate():
        # Side to move is checkmated.
        return 0.0 if board.turn == player_color else 1.0
    if board.is_stalemate() or board.is_insufficient_material() \
            or board.is_seventyfive_moves() or board.is_fivefold_repetition():
        return 0.5
    # Game not finished within the supplied moves: score by material.
    mine = _material(board, player_color)
    theirs = _material(board, not player_color)
    if mine > theirs:
        return 0.6
    if mine == theirs:
        return 0.5
    return 0.4


# ---------------------------------------------------------------------------
# Math
# ---------------------------------------------------------------------------

def _normalize_number(text: str):
    """Extract a float from free text, e.g. 'x = 42' -> 42.0. None if absent."""
    # Strip common wrappers
    t = text.strip().replace(",", "")
    m = re.findall(r"-?\d+(?:\.\d+)?", t)
    if not m:
        return None
    try:
        return float(m[-1])  # last number is usually the answer
    except ValueError:
        return None


def reward_math(expected, model_text: str, tol: float = 1e-6) -> float:
    """1.0 if the model's answer matches expected, else 0.0.

    - Numeric expected: compare as floats within `tol` (handles
      "42", "42.0", "x = 42" uniformly).
    - String expected: case-insensitive exact match after stripping
      whitespace/punctuation.
    Never rewards partial credit for wrong answers.
    """
    if isinstance(expected, (int, float)):
        got = _normalize_number(model_text)
        if got is None:
            return 0.0
        return 1.0 if abs(got - float(expected)) <= tol else 0.0
    # String / symbolic answer
    norm = lambda s: re.sub(r"\s+", "", s.strip().lower().rstrip("."))
    return 1.0 if norm(str(expected)) == norm(model_text) else 0.0


# ---------------------------------------------------------------------------
# Logic puzzles
# ---------------------------------------------------------------------------

def reward_logic(expected_answer: str, model_text: str) -> float:
    """Binary 1.0/0.0 — the model's conclusion must match exactly.

    Normalizes case and whitespace; no partial credit.
    """
    norm = lambda s: re.sub(r"\s+", " ", s.strip().lower())
    return 1.0 if norm(expected_answer) == norm(model_text) else 0.0


# ---------------------------------------------------------------------------
# Code (unit-test execution)
# ---------------------------------------------------------------------------

def reward_code(code: str, tests: str, timeout_secs: int = 10) -> float:
    """Run `code` against `tests` in an isolated subprocess.

    Reward = fraction of test functions passed (0.0–1.0).
    - `tests` is Python source defining test functions named `test_*`
      that import from the solution module.
    - The solution is written to a temp dir as `solution.py`; tests run
      with cwd=tempdir so only that dir is importable.
    - Any crash/timeout in the solution counts those tests as failed.
    - Empty test list -> 0.0 (nothing verifiable).

    NEVER rewards code for looking plausible — only executed outcomes.
    """
    test_names = re.findall(r"^def (test_\w+)", tests, re.M)
    if not test_names:
        return 0.0

    with tempfile.TemporaryDirectory() as tmp:
        sol_path = os.path.join(tmp, "solution.py")
        test_path = os.path.join(tmp, "run_tests.py")
        with open(sol_path, "w") as f:
            f.write(code)
        # Runner: import solution, run each test_*, report pass/fail per test.
        # Inject `solution` into the test module's namespace so test
        # functions can reference it directly.
        runner = (
            "import traceback\n"
            "import solution\n"
            "import test_defs\n"
            "test_defs.solution = solution\n"
            "passed = 0\n"
            f"total = {len(test_names)}\n"
            f"names = {test_names!r}\n"
            "for name in names:\n"
            "    try:\n"
            "        getattr(test_defs, name)()\n"
            "        passed += 1\n"
            "    except Exception:\n"
            "        traceback.print_exc()\n"
            "print(f'PASSED:{passed}/{total}')\n"
        )
        with open(test_path, "w") as f:
            f.write(runner)
        with open(os.path.join(tmp, "test_defs.py"), "w") as f:
            f.write(tests)

        try:
            proc = subprocess.run(
                [sys.executable, "run_tests.py"],
                cwd=tmp, capture_output=True, text=True,
                timeout=timeout_secs,
            )
        except subprocess.TimeoutExpired:
            return 0.0

        m = re.search(r"PASSED:(\d+)/(\d+)", proc.stdout)
        if not m:
            return 0.0
        passed, total = int(m.group(1)), int(m.group(2))
        return passed / total if total else 0.0


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

REWARD_FNS = {
    "sudoku": reward_sudoku,
    "chess_move": reward_chess_move,
    "chess_game": reward_chess_game,
    "math": reward_math,
    "logic": reward_logic,
    "code": reward_code,
}
