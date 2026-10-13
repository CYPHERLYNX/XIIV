"""Tests for Systems 1+2: verifiable rewards, RWR, opponents.

Run:  .venv/bin/python xiiv/games/test_systems_1_2.py

Every test uses dummy/fixed inputs — no model required. Anything that
needs the live 1B model is marked NOT TESTABLE HERE and skipped with
an explicit message (never faked).
"""

import sys
import chess

sys.path.insert(0, __import__("os").path.expanduser("~/workspace/ai-tools/corsac"))

from xiiv.games import rewards as R
from xiiv.games.opponents import (
    OpponentPool, RandomOpponent, GreedyOpponent, HistoryOpponent,
    CheckpointOpponent, DifficultyScheduler, generate_targeted_challenges,
    build_default_pool,
)
from xiiv.games.protocol import LearningHistory, GameResult
from xiiv.games.rl import reward_to_weight, RWRConfig
from xiiv.games.sudoku import generate_extreme_sudoku, grid_to_text

passed, failed = 0, 0

def check(name, cond):
    global passed, failed
    if cond:
        passed += 1
        print(f"  PASS {name}")
    else:
        failed += 1
        print(f"  FAIL {name}")


print("== reward_sudoku ==")
puzzle, answer = generate_extreme_sudoku()
solved_text = grid_to_text(answer)
check("valid solution -> 1.0", R.reward_sudoku(puzzle, solved_text) == 1.0)
check("garbage -> 0.0", R.reward_sudoku(puzzle, "not a grid") == 0.0)
# Corrupt one cell of the answer
bad = [row[:] for row in answer]
bad[0][0] = (bad[0][0] % 9) + 1
if bad[0][0] == answer[0][0]:
    bad[0][0] = (bad[0][0] % 9) + 1
check("wrong digit -> 0.0", R.reward_sudoku(puzzle, grid_to_text(bad)) == 0.0)

print("== reward_chess_move ==")
check("e2e4 legal -> 0.2", R.reward_chess_move(chess.STARTING_FEN, "e2e4") == 0.2)
check("illegal e2e5 -> 0.0", R.reward_chess_move(chess.STARTING_FEN, "e2e5") == 0.0)
check("bad UCI -> 0.0", R.reward_chess_move(chess.STARTING_FEN, "zzzz") == 0.0)
# Fool's mate: 1.f3 e5 2.g4 Qh4#
b = chess.Board()
for uci in ["f2f3", "e7e5", "g2g4"]:
    b.push(chess.Move.from_uci(uci))
check("Qh4# checkmate -> 1.0",
      R.reward_chess_move(b.fen(), "d8h4", player_color=chess.BLACK) == 1.0)
# Capture: white pawn d4 takes e5 (diagonal pawn capture)
b2 = chess.Board("rnbqkbnr/ppp1pppp/8/4p3/3P4/8/PPP1PPPP/RNBQKBNR w KQkq - 0 2")
r = R.reward_chess_move(b2.fen(), "d4e5")
check("dxe5 capture -> 0.3", abs(r - 0.3) < 1e-9)

print("== reward_chess_game ==")
# Scholar's mate as white: 1.e4 e5 2.Qh5 Nc6 3.Bc4 Nf6 4.Qxf7#
scholar = ["e2e4", "e7e5", "d1h5", "b8c6", "f1c4", "g8f6", "h5f7"]
check("scholar's mate win -> 1.0",
      R.reward_chess_game(scholar, player_color=chess.WHITE) == 1.0)
check("illegal move -> 0.0 loss",
      R.reward_chess_game(["e2e5"], player_color=chess.WHITE) == 0.0)

print("== reward_math ==")
check("42 exact -> 1.0", R.reward_math(42, "42") == 1.0)
check("'x = 42' -> 1.0", R.reward_math(42, "x = 42") == 1.0)
check("43 -> 0.0", R.reward_math(42, "43") == 0.0)
check("no number -> 0.0", R.reward_math(42, "hello") == 0.0)
check("float tol -> 1.0", R.reward_math(3.14159, "3.14159") == 1.0)
check("string match -> 1.0", R.reward_math("Paris", "paris") == 1.0)
check("string mismatch -> 0.0", R.reward_math("Paris", "London") == 0.0)

print("== reward_logic ==")
check("match -> 1.0", R.reward_logic("True", "true") == 1.0)
check("mismatch -> 0.0", R.reward_logic("True", "False") == 0.0)

print("== reward_code ==")
code_ok = "def add(a, b):\n    return a + b\n"
tests = ("def test_add():\n    assert solution.add(2, 3) == 5\n"
         "def test_neg():\n    assert solution.add(-1, 1) == 0\n")
check("2/2 tests -> 1.0", R.reward_code(code_ok, tests) == 1.0)
code_half = "def add(a, b):\n    return 5\n"  # passes first, fails second
check("1/2 tests -> 0.5", R.reward_code(code_half, tests) == 0.5)
code_bad = "def add(a, b):\n    raise ValueError('nope')\n"
check("crash -> 0.0", R.reward_code(code_bad, tests) == 0.0)
check("timeout -> 0.0",
      R.reward_code("while True:\n    pass\n", tests, timeout_secs=2) == 0.0)
check("no tests -> 0.0", R.reward_code(code_ok, "# nothing") == 0.0)

print("== reward_to_weight ==")
check("r=1,t=1 -> e", abs(reward_to_weight(1.0, 1.0) - 2.718281828) < 1e-6)
check("r=0 -> 1.0", reward_to_weight(0.0) == 1.0)
check("monotonic", reward_to_weight(0.8) > reward_to_weight(0.2))

print("== opponents: select_move ==")
board = chess.Board()
rm = RandomOpponent().select_move(board)
check("random returns legal move", rm in board.legal_moves)
gm = GreedyOpponent().select_move(board)
check("greedy returns legal move", gm in board.legal_moves)
# Greedy must take a hanging queen
b3 = chess.Board("rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2")
# white pawn e4 takes d5
mv = GreedyOpponent().select_move(b3)
check("greedy takes hanging pawn", mv is not None and b3.is_capture(mv))

print("== opponents: pool + scheduler ==")
pool = build_default_pool()
check("pool has random+greedy", set(pool.names()) == {"random", "greedy"})
sched = DifficultyScheduler(pool)
check("starts at random tier", sched.current_tier == "random")
adv, reason = sched.should_advance()
check("no advance with 0 games", adv is False)
# Simulate 20 wins vs random -> should advance
for _ in range(20):
    pool.record("random", True)
msg = sched.maybe_advance()
check("advances after 20/20 vs random",
      sched.current_tier == "greedy" and "ADVANCED" in msg)
# 10 wins 10 losses vs greedy -> stays
for i in range(20):
    pool.record("greedy", i % 2 == 0)
msg2 = sched.maybe_advance()
check("stays at greedy on 50% win rate",
      sched.current_tier == "greedy" and "staying" in msg2)

print("== CheckpointOpponent ==")
ckpt = CheckpointOpponent("/tmp/fake_ckpt.pt")
try:
    ckpt.select_move(chess.Board())
    check("raises without move_fn", False)
except RuntimeError:
    check("raises without move_fn", True)
ckpt2 = CheckpointOpponent(
    str(__import__("pathlib").Path.home() /
        "workspace/ai-tools/corsac/checkpoints_1b_lora/lora_ckpt_675.pt"),
    move_fn=lambda fen: "e2e4")
m2 = ckpt2.select_move(chess.Board())
check("injected move_fn plays e2e4",
      m2 == chess.Move.from_uci("e2e4"))

print("== HistoryOpponent + targeted challenges ==")
hist = LearningHistory("_test_tmp_game")
# Fabricate a short history with a known weakness (in-memory only;
# use a throwaway game name so we don't pollute real histories).
from datetime import datetime
for i in range(5):
    hist.record(GameResult(game="x", difficulty="extreme", opponent="random",
                           won=False, moves=81, duration_secs=1.0,
                           timestamp=datetime.now().isoformat(),
                           mistakes=["Row 3 invalid", "Box (1,1) invalid"]
                           if i % 2 == 0 else ["Row 3 invalid"]))
ho = HistoryOpponent(hist)
check("top weakness detected", ho.top_weakness() == "Row 3 invalid")
chals = generate_targeted_challenges(hist, "sudoku", n=3)
check("3 targeted challenges", len(chals) == 3 and all(c["targeted"] for c in chals))
# Cleanup throwaway history file
try:
    hist.history_file.unlink()
except OSError:
    pass

print("== RWRConfig defaults ==")
cfg = RWRConfig()
check("default temperature 1.0", cfg.temperature == 1.0)
check("default ckpt_every 25", cfg.ckpt_every == 25)

print()
print(f"RESULT: {passed} passed, {failed} failed")
print("NOT TESTABLE HERE (needs live 1B model):")
print("  - RWRTrainer.step() forward/backward (needs XIIVModel + LoRA)")
print("  - CheckpointOpponent with real move_fn (needs model inference)")
print("  - rwr_loss_for_sample masking (needs model forward)")
sys.exit(1 if failed else 0)
