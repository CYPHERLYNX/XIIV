"""Tests for xiiv.eb_gate (Entropy-Bounded certificate-gated commitment).

CPU-only, deterministic, stdlib-only. The module is loaded by file path so
these tests run on a bare box without importing the torch-dependent
``xiiv/__init__`` -- the selector itself has no model dependency by design.

Pre-registered falsifiers (see xiiv/eb_gate.py):
- F1 certificate soundness: no committed token is uncertifiable.
- F2 bound: sum(H) - max(H) <= gamma for every returned commit set.
- F3 progress: a certifiable top-1 with gamma >= 0 always yields >= 1 commit.
- F4 prefix property: commits are the lowest-entropy certifiable positions,
  in ascending-entropy order, with no gaps.
KILL: any failure here kills the module; it must not ship.
"""

import importlib.util
import math
import os
import random
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
_SPEC = importlib.util.spec_from_file_location(
    "eb_gate_under_test",
    os.path.join(_HERE, "..", "xiiv", "eb_gate.py"),
)
eb_gate = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(eb_gate)


def _dist_from_logits(logits):
    m = max(logits)
    exps = [math.exp(x - m) for x in logits]
    s = sum(exps)
    return [e / s for e in exps]


def _check_result(test, dists, cert, gamma, result):
    """Assert F1/F2/F4 structurally for one result."""
    entropies = {}
    for pos, probs in enumerate(dists):
        top = max(range(len(probs)), key=lambda t: (probs[t], -t))
        entropies[pos] = (eb_gate.entropy(probs), top)
    # F1: every commit is certifiable.
    for c in result.commits:
        test.assertTrue(cert(c.position, c.token), f"uncertifiable commit {c}")
        test.assertEqual(c.token, entropies[c.position][1])
        test.assertAlmostEqual(c.entropy, entropies[c.position][0], places=9)
    # F2: bound holds.
    if result.commits:
        hs = [c.entropy for c in result.commits]
        test.assertLessEqual(sum(hs) - max(hs), gamma + 1e-9)
        test.assertAlmostEqual(result.bound_value, sum(hs) - max(hs), places=9)
    else:
        test.assertEqual(result.bound_value, 0.0)
    # F4: commits are a prefix of the certifiable candidates in ascending
    # entropy order (ties by position).
    certifiable_sorted = sorted(
        (h, pos) for pos, (h, top) in entropies.items() if cert(pos, top)
    )
    committed_positions = [c.position for c in result.commits]
    test.assertEqual(
        committed_positions,
        [pos for _, pos in certifiable_sorted[: len(committed_positions)]],
    )
    # Commit order itself is ascending entropy.
    hs = [c.entropy for c in result.commits]
    test.assertEqual(hs, sorted(hs))


class TestEBGate(unittest.TestCase):
    def test_empty_input(self):
        r = eb_gate.select_commits([], lambda p, t: True, 1.0)
        self.assertEqual(r.commits, ())
        self.assertEqual(r.bound_value, 0.0)

    def test_negative_gamma_commits_nothing(self):
        dists = [[0.9, 0.1], [0.6, 0.4]]
        r = eb_gate.select_commits(dists, lambda p, t: True, -0.5)
        self.assertEqual(r.commits, ())

    def test_gamma_zero_commits_exactly_lowest_entropy_certifiable(self):
        # First commit always has sum - max == 0, so gamma=0 admits exactly one
        # commit (unless it is one-hot, covered below).
        dists = [[0.9, 0.1], [0.6, 0.4], [0.99, 0.01]]
        r = eb_gate.select_commits(dists, lambda p, t: True, 0.0)
        self.assertEqual([c.position for c in r.commits], [2])
        self.assertEqual(r.skipped_by_bound, (0, 1))

    def test_one_hot_distributions_all_commit_at_gamma_zero(self):
        dists = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        r = eb_gate.select_commits(dists, lambda p, t: True, 0.0)
        self.assertEqual([c.position for c in r.commits], [0, 1, 2])
        self.assertAlmostEqual(r.bound_value, 0.0)

    def test_uncertifiable_top1_never_committed(self):
        # Adversarial: the model's favourite token is forbidden everywhere.
        dists = [[0.9, 0.1], [0.8, 0.2], [0.7, 0.3]]
        cert = lambda p, t: t != 0  # noqa: E731 -- top-1 is always token 0
        r = eb_gate.select_commits(dists, cert, 10.0)
        self.assertEqual(r.commits, ())
        self.assertEqual(r.skipped_uncertifiable, (0, 1, 2))

    def test_mixed_certificates_and_bound(self):
        dists = [
            [0.99, 0.01],  # pos 0: lowest entropy, certifiable
            [0.9, 0.1],  # pos 1: uncertifiable top-1
            [0.6, 0.4],  # pos 2: high entropy
            [0.8, 0.2],  # pos 3: mid entropy
        ]
        cert = lambda p, t: not (p == 1 and t == 0)  # noqa: E731
        r = eb_gate.select_commits(dists, cert, 0.05)
        _check_result(self, dists, cert, 0.05, r)
        # pos 0 commits (bound 0); pos 3 would raise the bound to H(pos 0),
        # which exceeds 0.05, so only pos 0 commits.
        self.assertEqual([c.position for c in r.commits], [0])
        self.assertIn(1, r.skipped_uncertifiable)

    def test_invalid_distributions_raise(self):
        with self.assertRaises(ValueError):
            eb_gate.select_commits([[]], lambda p, t: True, 1.0)
        with self.assertRaises(ValueError):
            eb_gate.select_commits([[0.5, -0.5]], lambda p, t: True, 1.0)
        with self.assertRaises(ValueError):
            eb_gate.select_commits([[0.0, 0.0]], lambda p, t: True, 1.0)

    def test_fuzz_falsifiers(self):
        rng = random.Random(20261028)
        for trial in range(300):
            n_pos = rng.randint(1, 12)
            vocab = rng.randint(2, 8)
            dists = [
                _dist_from_logits([rng.uniform(-3.0, 3.0) for _ in range(vocab)])
                for _ in range(n_pos)
            ]
            forbidden = {p for p in range(n_pos) if rng.random() < 0.3}
            cert = lambda p, t, forbidden=forbidden: p not in forbidden  # noqa: E731
            gamma = rng.choice([0.0, 0.05, 0.2, 0.5, 1.0, 5.0])
            r = eb_gate.select_commits(dists, cert, gamma)
            _check_result(self, dists, cert, gamma, r)
            # F3: progress whenever any certifiable top-1 exists.
            any_cert = any(p not in forbidden for p in range(n_pos))
            if any_cert:
                self.assertGreaterEqual(len(r.commits), 1)


if __name__ == "__main__":
    unittest.main()
