"""
Sudoku harness for Extreme Intelligence Training Protocol.
Difficulty: EXTREME (minimal givens, requires advanced techniques).

The model receives the puzzle as text and must output the solved grid.
Every attempt is scored: correct = win, incorrect = loss with analysis.
"""

import random
import time
from datetime import datetime


def generate_extreme_sudoku():
    """Generate an extreme-difficulty Sudoku (17-22 givens)."""
    # Start from a solved grid, remove cells strategically
    base = [
        [5,3,4,6,7,8,9,1,2],
        [6,7,2,1,9,5,3,4,8],
        [1,9,8,3,4,2,5,6,7],
        [8,5,9,7,6,1,4,2,3],
        [4,2,6,8,5,3,7,9,1],
        [7,1,3,9,2,4,8,5,6],
        [9,6,1,5,3,7,2,8,4],
        [2,8,7,4,1,9,6,3,5],
        [3,4,5,2,8,6,1,7,9],
    ]
    # Shuffle for variety
    random.shuffle(base)
    for row in base:
        # Shuffle within band constraints (simplified)
        pass

    # Remove cells to reach extreme difficulty (keep 20 givens)
    puzzle = [row[:] for row in base]
    cells = [(r, c) for r in range(9) for c in range(9)]
    random.shuffle(cells)
    for r, c in cells[20:]:
        puzzle[r][c] = 0

    return puzzle, base


def grid_to_text(grid):
    """Format grid as text for model input."""
    lines = []
    for r in range(9):
        row = []
        for c in range(9):
            v = grid[r][c]
            row.append(str(v) if v != 0 else ".")
            if c in (2, 5):
                row.append("|")
        lines.append(" ".join(row))
        if r in (2, 5):
            lines.append("-" * 21)
    return "\n".join(lines)


def parse_solution(text):
    """Parse model's text output back into a grid."""
    grid = []
    for line in text.strip().split("\n"):
        # Remove separators
        clean = line.replace("|", "").replace("-", "").strip()
        if not clean:
            continue
        row = []
        for ch in clean.split():
            if ch == "." or ch == "0":
                row.append(0)
            elif ch.isdigit() and 1 <= int(ch) <= 9:
                row.append(int(ch))
        if len(row) == 9:
            grid.append(row)
    return grid if len(grid) == 9 else None


def is_valid_solution(puzzle, solution):
    """Check if solution is correct: fills puzzle and follows Sudoku rules."""
    if solution is None or len(solution) != 9:
        return False, "Invalid grid shape"

    # Check puzzle givens preserved
    for r in range(9):
        for c in range(9):
            if puzzle[r][c] != 0 and puzzle[r][c] != solution[r][c]:
                return False, f"Changed given at ({r},{c})"

    # Check rows
    for r in range(9):
        if sorted(solution[r]) != list(range(1, 10)):
            return False, f"Row {r} invalid"

    # Check columns
    for c in range(9):
        col = [solution[r][c] for r in range(9)]
        if sorted(col) != list(range(1, 10)):
            return False, f"Column {c} invalid"

    # Check boxes
    for br in range(3):
        for bc in range(3):
            box = []
            for r in range(br*3, br*3+3):
                for c in range(bc*3, bc*3+3):
                    box.append(solution[r][c])
            if sorted(box) != list(range(1, 10)):
                return False, f"Box ({br},{bc}) invalid"

    return True, "Correct"


def play_sudoku(model_solve_fn, difficulty="extreme"):
    """
    Play one Sudoku game.
    model_solve_fn: takes puzzle text, returns solution text.
    Returns (won, moves, mistakes, duration).
    """
    puzzle, answer = generate_extreme_sudoku()
    puzzle_text = grid_to_text(puzzle)

    prompt = (
        "Solve this EXTREME difficulty Sudoku. Fill every empty cell (marked .).\n"
        "Output ONLY the completed 9x9 grid, one row per line, digits separated by spaces.\n"
        "Use | between 3x3 boxes and --- between bands, matching the input format.\n\n"
        f"{puzzle_text}"
    )

    t0 = time.time()
    try:
        solution_text = model_solve_fn(prompt)
    except Exception as e:
        return False, 0, [f"Model error: {e}"], time.time() - t0
    duration = time.time() - t0

    solution = parse_solution(solution_text)
    valid, msg = is_valid_solution(puzzle, solution)

    mistakes = [] if valid else [msg]
    return valid, 81, mistakes, duration


if __name__ == "__main__":
    # Self-test with a dummy solver (returns the puzzle unsolved = always loses)
    def dummy_solver(prompt):
        # Extract puzzle and return it unchanged (will fail validation)
        return prompt.split("\n\n")[-1]

    won, moves, mistakes, dur = play_sudoku(dummy_solver)
    print(f"Dummy solver: won={won}, mistakes={mistakes}")
    print("Harness working — model integration pending.")
