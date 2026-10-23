"""
Interval abstract domain for the Compositional Proof Machine
(spec section 5.6, task 3: noisy inputs).

An input is no longer a point x0 but an interval I0 = [x0 - r, x0 + r]
(noise radius r >= 0). The abstract state at each step is an integer closed
interval (lo, hi), lo <= hi. The certificate is the interval trace
I0 -> I1 -> ... -> IL.

SOUNDNESS: each interval transformer below computes a SOUND over-approximation
of the op's image of the input interval (every concrete output lies inside).
EXACTNESS (corrected 2026-10-23 after the kill-gate falsified the stronger
claim): ADD_k, MUL_k, NEG are EXACT (monotone/antitone: image of a full
integer interval is a full integer interval). MOD_m is exact in the no-wrap
case, but in the WRAP case (guard interval straddles a multiple of m) it
returns the HULL [0, m-1] -- the true image is {0..b} union {a..m-1}, a
strict subset with gaps. The hull is the smallest sound interval, and both
endpoints are attained, but it is NOT the true set. Consequence: precision
loss under noise comes from (a) MOD-wrap hulls -- even on flat chains -- and
(b) conditional joins (hull of branch intervals). Both are measured, not
assumed, by killgate_noise.py.

Representation: plain (lo, hi) int tuples. JSON: [lo, hi].
"""

# ---------------------------------------------------------------------------
# Domain operations
# ---------------------------------------------------------------------------

def iv(lo, hi):
    lo, hi = int(lo), int(hi)
    assert lo <= hi, f"empty interval [{lo}, {hi}]"
    return (lo, hi)


def width(interval):
    return interval[1] - interval[0]


def contains(interval, x):
    return interval[0] <= int(x) <= interval[1]


def interval_eq(a, b):
    return a[0] == b[0] and a[1] == b[1]


def join(a, b):
    """Hull (least upper bound): the smallest interval containing both."""
    return (min(a[0], b[0]), max(a[1], b[1]))


def straddles_zero(g):
    """Guard-interval case split for IFZ: 0 in G but G != {0}."""
    return g[0] <= 0 <= g[1] and not (g[0] == 0 and g[1] == 0)


def guard_decision(g):
    """
    Decide an IFZ branch from a guard INTERVAL (not a point).
    Returns "then" | "else" | "straddle".
      then     iff G == {0}
      else     iff 0 not in G
      straddle iff 0 in G and G != {0}  (both branches reachable)
    """
    if g[0] == 0 and g[1] == 0:
        return "then"
    if g[1] < 0 or g[0] > 0:
        return "else"
    return "straddle"


# ---------------------------------------------------------------------------
# Fixed interval abstract transformers (the rule table). Sound by construction;
# exact for this op set (proofs in comments). NOT learned.
# ---------------------------------------------------------------------------

def _i_add1(v): return (v[0] + 1, v[1] + 1)
def _i_add3(v): return (v[0] + 3, v[1] + 3)
def _i_add7(v): return (v[0] + 7, v[1] + 7)
def _i_mul2(v): return (v[0] * 2, v[1] * 2)   # k > 0: monotone, endpoints exact
def _i_mul5(v): return (v[0] * 5, v[1] * 5)
def _i_neg(v): return (-v[1], -v[0])           # antitone: endpoints swap, exact


def _i_mod7(v): return _i_mod(v, 7)
def _i_mod13(v): return _i_mod(v, 13)


def _i_mod(v, m):
    """
    SOUND hull of the image of integer interval [lo, hi] under x -> x mod m
    (Python semantics: 0 <= result < m). Exact (image == output interval) in
    the no-wrap case; in the wrap case returns [0, m-1], the smallest sound
    interval -- the true image {0..b} u {a..m-1} has gaps, so this is a
    strict over-approximation (endpoints 0 and m-1 are both attained, proved
    below, so the hull is tight as an interval).
    Proof of the wrap characterization: let a = lo % m, b = hi % m. Residues
    of consecutive integers increase by 1 mod m, so a > b iff the interval
    crosses a multiple of m. If hi - lo >= m the interval covers a full
    residue system -> image = [0, m-1] exactly. Else if a <= b (no crossing):
    image = [a, b] exactly (consecutive residues, all attained). Else (wrap):
    the multiple q*m lies in (lo, hi]; q*m gives residue 0 and q*m - 1 >= lo
    gives residue m-1 -- both attained, hull [0, m-1] sound and endpoint-tight.
    """
    lo, hi = v
    assert m > 0
    if hi - lo >= m:
        return (0, m - 1)
    a, b = lo % m, hi % m
    if a <= b:
        return (a, b)
    return (0, m - 1)


INTERVAL_RULES = {
    "ADD_1": _i_add1,
    "ADD_3": _i_add3,
    "ADD_7": _i_add7,
    "MUL_2": _i_mul2,
    "MUL_5": _i_mul5,
    "MOD_7": _i_mod7,
    "MOD_13": _i_mod13,
    "NEG": _i_neg,
}


# ---------------------------------------------------------------------------
# JSON
# ---------------------------------------------------------------------------

def iv_to_json(v):
    return [int(v[0]), int(v[1])]


def iv_from_json(j):
    return iv(j[0], j[1])
