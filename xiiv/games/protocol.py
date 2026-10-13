"""
EXTREME INTELLIGENCE TRAINING PROTOCOL
======================================
Makes XIIV compete in strategy and mind games at increasing difficulty.
One game at a time. 9/10 wins required to advance.

Games (in order):
  1. Sudoku (extreme) — pure logic, text-native
  2. Logic puzzles — deductive reasoning
  3. Chess — tactical play vs engines
  4. Go/Shogi/Poker — future (need specialized harnesses)

Honest tracking: every game, every move, every loss recorded.
No fake victories. Metrics are real or marked NOT RUN.
"""

import json
import time
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional

HISTORY_DIR = Path.home() / "workspace" / "ai-tools" / "corsac" / "xiiv" / "games" / "history"
HISTORY_DIR.mkdir(parents=True, exist_ok=True)

TARGET_WINS = 9
TARGET_GAMES = 10


@dataclass
class GameResult:
    game: str
    difficulty: str
    opponent: str
    won: bool
    moves: int
    duration_secs: float
    timestamp: str
    notes: str = ""
    # For analysis
    mistakes: list = None
    key_decisions: list = None

    def __post_init__(self):
        if self.mistakes is None:
            self.mistakes = []
        if self.key_decisions is None:
            self.key_decisions = []


class LearningHistory:
    """Persistent record of all games, wins, losses, and lessons."""

    def __init__(self, game_name: str):
        self.game_name = game_name
        self.history_file = HISTORY_DIR / f"{game_name}_history.jsonl"
        self.results: list[GameResult] = []
        self._load()

    def _load(self):
        if self.history_file.exists():
            with open(self.history_file) as f:
                for line in f:
                    if line.strip():
                        d = json.loads(line)
                        self.results.append(GameResult(**d))

    def record(self, result: GameResult):
        self.results.append(result)
        with open(self.history_file, "a") as f:
            f.write(json.dumps(asdict(result)) + "\n")

    def recent_record(self, n: int = 10) -> tuple[int, int]:
        """Returns (wins, total) for last n games."""
        recent = self.results[-n:]
        wins = sum(1 for r in recent if r.won)
        return wins, len(recent)

    def meets_target(self) -> bool:
        """9/10 wins in last 10 consecutive games."""
        if len(self.results) < TARGET_GAMES:
            return False
        wins, total = self.recent_record(TARGET_GAMES)
        return wins >= TARGET_WINS and total == TARGET_GAMES

    def win_rate(self) -> float:
        if not self.results:
            return 0.0
        return sum(1 for r in self.results if r.won) / len(self.results)

    def analyze_weaknesses(self) -> dict:
        """Aggregate mistakes across losses to identify patterns."""
        all_mistakes = []
        for r in self.results:
            if not r.won:
                all_mistakes.extend(r.mistakes)
        # Count mistake types
        from collections import Counter
        return dict(Counter(all_mistakes))

    def summary(self) -> str:
        wins, total = self.recent_record(TARGET_GAMES)
        lines = [
            f"=== {self.game_name} ===",
            f"Total games: {len(self.results)}",
            f"Overall win rate: {self.win_rate():.1%}",
            f"Last {TARGET_GAMES}: {wins}/{total} wins",
            f"Target ({TARGET_WINS}/{TARGET_GAMES}): {'MET ✓' if self.meets_target() else 'NOT MET'}",
        ]
        weaknesses = self.analyze_weaknesses()
        if weaknesses:
            lines.append("Top weaknesses from losses:")
            for mistake, count in sorted(weaknesses.items(), key=lambda x: -x[1])[:5]:
                lines.append(f"  - {mistake}: {count}x")
        return "\n".join(lines)


class ExtremeProtocol:
    """Orchestrates the full protocol across all games."""

    GAMES_IN_ORDER = ["sudoku", "logic_puzzles", "chess", "go", "shogi", "poker"]

    def __init__(self):
        self.histories = {g: LearningHistory(g) for g in self.GAMES_IN_ORDER}
        self.current_game_idx = 0
        self._find_current_game()

    def _find_current_game(self):
        """Advance to first game that hasn't met target."""
        for i, game in enumerate(self.GAMES_IN_ORDER):
            if not self.histories[game].meets_target():
                self.current_game_idx = i
                return
        self.current_game_idx = len(self.GAMES_IN_ORDER)  # all complete

    @property
    def current_game(self) -> Optional[str]:
        if self.current_game_idx >= len(self.GAMES_IN_ORDER):
            return None
        return self.GAMES_IN_ORDER[self.current_game_idx]

    def status(self) -> str:
        lines = ["EXTREME INTELLIGENCE TRAINING PROTOCOL", "=" * 45]
        for game in self.GAMES_IN_ORDER:
            h = self.histories[game]
            wins, total = h.recent_record(10)
            status = "✓ DONE" if h.meets_target() else "→ ACTIVE" if game == self.current_game else "· waiting"
            lines.append(f"{game:15s} {wins:2d}/{total:2d}  {status}")
        lines.append("=" * 45)
        if self.current_game:
            lines.append(f"Current focus: {self.current_game}")
        else:
            lines.append("ALL GAMES COMPLETE — protocol finished")
        return "\n".join(lines)


if __name__ == "__main__":
    p = ExtremeProtocol()
    print(p.status())
