#!/usr/bin/env python3
"""
PROBE 2b — Conservation Reaction Substrate: discrete rule discovery.

Redirect after Probe 2 KILLED gradient-based discovery (gain explosion into a
chaotic attractor; NaN divergence learning mass-action exponents — see
reaction_probe2_results.md). Tests DreamCoder-style DISCRETE enumerative
search + compression instead: no gradients on rule matrices, no ODE rollout,
so both Probe-2 failure modes are sidestepped by construction.

Prior art (honest): DreamCoder (Ellis et al.) discovers library abstractions
via enumerative search over a DSL + compression (find common substructure,
add to library). This probe transfers that METHOD to reaction-rule
discovery; no novelty is claimed over DreamCoder.

Task: discover the catalyzed doubling rule `a + c -> 2a + c` (c = catalyst,
required but unconsumed) from (soup_before, soup_after) pairs.
Target rule known -> success is checkable (same honesty standard as Probe 2).

SPEC CORRECTION (flagged): the probe spec said "at most 3 products", but its
own tripling example (`a + c -> 3a + c`) needs 4 product terms. Grammar used:
|reactants| <= 2, |products| <= 5 (uniform; admits tripling and the
quadrupling reuse task).

CHARGE DESIGN (documented limitation): q(a) = 0-vector is FORCED by the
target rule form: q(a)+q(c) = 2q(a)+q(c) ==> q(a) = 0. Consequence: the
certificate constrains catalyst/distractor conservation, NOT the doubling
itself; enumeration + behavioral data carry the doubling structure. This is
stated plainly — the memo separates what the certificate contributes from
what search+data contribute. Distractors carry distinct charges so rules
that create/destroy them are pruned by the certificate.

KILL if: the rule is not recovered exactly on held-out pairs within budget;
enumeration explodes combinatorially past pruning (branching factor reported
honestly); the compressor finds spurious common substructure.
PASS if: exact rule recovered + reuse via compression on a second task
family with measured search-cost reduction.

stdlib only. Deterministic (seed fixed below). No post-hoc tuning.
"""
import itertools
import random
import time
from collections import Counter

SEED = 20261018
ALPHABET = ['a', 'c', 'd1', 'd2', 'd3', 'd4']
# charge vectors: a neutral (forced, see docstring); catalyst/distractors distinct
CHARGE = {
    'a':  (0, 0, 0, 0, 0),
    'c':  (1, 0, 0, 0, 0),
    'd1': (0, 1, 0, 0, 0),
    'd2': (0, 0, 1, 0, 0),
    'd3': (0, 0, 0, 1, 0),
    'd4': (0, 0, 0, 0, 1),
}
MAX_R, MAX_P = 2, 5          # grammar bounds (see spec correction above)
N_TRAIN, N_TEST = 64, 32
REACTIVE_FRAC = 0.7
DISTRACTOR_P = 0.5


def multisets(types, max_n):
    """All multisets of size 0..max_n over types, as sorted tuples."""
    out = []
    for k in range(max_n + 1):
        for combo in itertools.combinations_with_replacement(types, k):
            out.append(combo)
    return out


def charge(ms):
    q = [0] * 5
    for t in ms:
        v = CHARGE[t]
        for i in range(5):
            q[i] += v[i]
    return tuple(q)


def conserves(R, P):
    return charge(R) == charge(P)


def norm(c):
    return Counter({t: n for t, n in c.items() if n > 0})


def apply_rule(rule, soup):
    """Single deterministic application. Returns resulting soup Counter,
    or None if the rule does not fire (reactants not present)."""
    R, P = rule
    need = Counter(R)
    for t, n in need.items():
        if soup.get(t, 0) < n:
            return None
    out = Counter(soup)
    out.subtract(need)
    out.update(Counter(P))
    return norm(out)


def matches(rule, before, after):
    """Exact behavioral match on one pair. A non-firing rule matches only
    pairs where nothing should change."""
    res = apply_rule(rule, before)
    if res is None:
        return before == after
    return res == after


def gen_pairs(rng, catalyst, factor, n):
    """(before, after) pairs. Reactive (factor a:1 -> a:factor, catalyst
    required+conserved); non-reactive (catalyst absent OR substrate absent)
    -> unchanged. Distractors vary independently per pair."""
    distractors = [t for t in ALPHABET if t not in ('a', catalyst)]
    pairs = []
    for _ in range(n):
        D = Counter({d: 1 for d in distractors if rng.random() < DISTRACTOR_P})
        if rng.random() < REACTIVE_FRAC:
            before = Counter({'a': 1, catalyst: 1})
            before.update(D)
            after = Counter({'a': factor, catalyst: 1})
            after.update(D)
        else:
            before = Counter({'a': 1}) if rng.random() < 0.5 else Counter({catalyst: 1})
            before.update(D)
            after = Counter(before)
        pairs.append((norm(before), norm(after)))
    return pairs


def enumerate_rules():
    Rs = multisets(ALPHABET, MAX_R)
    Ps = multisets(ALPHABET, MAX_P)
    raw = [(R, P) for R in Rs for P in Ps]
    pruned = [(R, P) for (R, P) in raw if conserves(R, P)]
    return raw, pruned


def discover(pairs_train, pairs_test, candidates):
    """Enumerate + behavioral filter. Select on TRAIN only (pre-declared
    tie-break: fewest atoms, then lexicographic). Evaluate selected once
    on held-out. Returns (rule_or_None, n_train_survivors, n_evaluated)."""
    survivors = [r for r in candidates
                 if all(matches(r, b, a) for b, a in pairs_train)]
    survivors.sort(key=lambda r: (len(r[0]) + len(r[1]), r[0], r[1]))
    if not survivors:
        return None, 0, len(candidates)
    best = survivors[0]
    if not all(matches(best, b, a) for b, a in pairs_test):
        return None, len(survivors), len(candidates)  # selected rule fails held-out
    return best, len(survivors), len(candidates)


def anti_unify_seq(r1, r2):
    """Plotkin-style anti-unification on (R_tuple, P_tuple). Difference
    pairs map to consistent variables. Lengths must match exactly;
    returns None if structurally incompatible."""
    (R1, P1), (R2, P2) = r1, r2
    if len(R1) != len(R2) or len(P1) != len(P2):
        return None
    varmap = {}

    def au(s1, s2):
        out = []
        for x, y in zip(s1, s2):
            if x == y:
                out.append(x)
            else:
                key = (x, y)
                if key not in varmap:
                    varmap[key] = 'X%d' % len(varmap)
                out.append(varmap[key])
        return tuple(out)

    return (au(R1, R2), au(P1, P2))


def instantiate_seq(schema, alphabet):
    """All charge-conserving, grammar-bounded instantiations of a sequence schema."""
    Rs, Ps = schema
    vars_ = sorted({t for t in Rs + Ps if t.startswith('X')})
    rules = []
    for vals in itertools.product(alphabet, repeat=len(vars_)):
        m = dict(zip(vars_, vals))
        R = tuple(sorted(m.get(t, t) for t in Rs))
        P = tuple(sorted(m.get(t, t) for t in Ps))
        if len(R) <= MAX_R and len(P) <= MAX_P and conserves(R, P):
            rules.append((R, P))
    return list(dict.fromkeys(rules))


def anti_unify_numeric(r1, r2):
    """Count-based anti-unification. Requires identical R skeleton;
    P may differ in counts -> numeric variables. Returns
    (R_dict, P_schema_dict) or None if skeletons incompatible."""
    R1, P1 = Counter(r1[0]), Counter(r1[1])
    R2, P2 = Counter(r2[0]), Counter(r2[1])
    if R1 != R2:
        return None
    if set(P1) != set(P2):
        return None
    schema_P, varmap = {}, {}
    for t in sorted(set(P1) | set(P2)):
        v1, v2 = P1.get(t, 0), P2.get(t, 0)
        if v1 == v2:
            schema_P[t] = v1
        else:
            key = (t, v1, v2)
            if key not in varmap:
                varmap[key] = 'N%d' % len(varmap)
            schema_P[t] = varmap[key]
    return (dict(R1), schema_P)


def instantiate_numeric(schema, max_val):
    """Instantiate numeric schema; variable ranges 0..max_val."""
    R, Ps = schema
    vars_ = sorted({v for v in Ps.values() if isinstance(v, str)})
    rules = []
    for vals in itertools.product(range(max_val + 1), repeat=len(vars_)):
        m = dict(zip(vars_, vals))
        P = norm(Counter({t: (m[v] if isinstance(v, str) else v)
                          for t, v in Ps.items()}))
        Rt = tuple(sorted(Counter(R).elements()))
        Pt = tuple(sorted(P.elements()))
        if len(Rt) <= MAX_R and len(Pt) <= MAX_P and conserves(Rt, Pt):
            rules.append((Rt, Pt))
    return list(dict.fromkeys(rules))


def rule_str(r):
    R, P = r
    s = lambda ms: '+'.join(ms) if ms else '0'
    return '%s -> %s' % (s(R), s(P))


def distractor_audit(pairs_train, catalyst):
    """Fraction of reactive train pairs containing each distractor.
    The distractor discipline is only real if every distractor is
    sometimes present AND sometimes absent."""
    distractors = [t for t in ALPHABET if t not in ('a', catalyst)]
    reactive = [(b, a) for b, a in pairs_train if b != a]
    out = {}
    for d in distractors:
        frac = sum(1 for b, a in reactive if b.get(d, 0) > 0) / max(1, len(reactive))
        out[d] = frac
    return out, len(reactive)


def main():
    t0 = time.time()
    rng = random.Random(SEED)
    report = []

    raw, pruned = enumerate_rules()
    report.append('raw candidates: %d' % len(raw))
    report.append('charge-pruned candidates: %d (%.1fx pruning)'
                  % (len(pruned), len(raw) / len(pruned)))

    # ---- Task 1: c-catalyzed doubling (the spec'd discovery target) ----
    tr1 = gen_pairs(rng, 'c', 2, N_TRAIN)
    te1 = gen_pairs(rng, 'c', 2, N_TEST)
    audit1, nreac1 = distractor_audit(tr1, 'c')
    r1, surv1, eval1 = discover(tr1, te1, pruned)
    report.append('T1 c-doubling: survivors(train)=%d evaluated=%d rule=%s'
                  % (surv1, eval1, rule_str(r1) if r1 else 'NONE'))
    report.append('T1 distractor presence in reactive pairs: %s'
                  % {k: round(v, 2) for k, v in audit1.items()})

    # ---- Task 2: d1-catalyzed doubling (second family, new catalyst) ----
    tr2 = gen_pairs(rng, 'd1', 2, N_TRAIN)
    te2 = gen_pairs(rng, 'd1', 2, N_TEST)
    r2, surv2, eval2 = discover(tr2, te2, pruned)
    report.append('T2 d1-doubling: survivors(train)=%d evaluated=%d rule=%s'
                  % (surv2, eval2, rule_str(r2) if r2 else 'NONE'))

    # ---- Task 3: c-catalyzed tripling (second family, new factor) ----
    tr3 = gen_pairs(rng, 'c', 3, N_TRAIN)
    te3 = gen_pairs(rng, 'c', 3, N_TEST)
    r3, surv3, eval3 = discover(tr3, te3, pruned)
    report.append('T3 c-tripling: survivors(train)=%d evaluated=%d rule=%s'
                  % (surv3, eval3, rule_str(r3) if r3 else 'NONE'))

    ok = True
    exp1 = (('a', 'c'), ('a', 'a', 'c'))
    exp2 = (('a', 'd1'), ('a', 'a', 'd1'))
    exp3 = (('a', 'c'), ('a', 'a', 'a', 'c'))
    for name, got, exp in [('T1', r1, exp1), ('T2', r2, exp2), ('T3', r3, exp3)]:
        if got != exp:
            report.append('MISMATCH %s: got %s, expected %s'
                          % (name, rule_str(got) if got else None,
                             rule_str(exp)))
            ok = False
    for name, s in [('T1', surv1), ('T2', surv2), ('T3', surv3)]:
        if s != 1:
            report.append('NOTE %s: %d train survivors (expected exactly 1)' % (name, s))

    # ---- Compression: schema S1 (catalyst-general doubling) ----
    S1 = anti_unify_seq(r1, r2) if (r1 and r2) else None
    report.append('S1 schema: %s' % (S1,))
    expS1 = (('a', 'X0'), ('a', 'a', 'X0'))
    if S1 != expS1:
        report.append('S1 MISMATCH: got %s expected %s' % (S1, expS1))
        ok = False
    inst1 = instantiate_seq(S1, ALPHABET) if S1 else []
    report.append('S1 instantiations (charge+grammar valid): %d' % len(inst1))
    # sanity: schema must regenerate both source rules
    if r1 not in inst1 or r2 not in inst1:
        report.append('S1 FAILS to regenerate source rules')
        ok = False

    # ---- Reuse Task 4: d2-catalyzed doubling via S1 ----
    tr4 = gen_pairs(rng, 'd2', 2, N_TRAIN)
    te4 = gen_pairs(rng, 'd2', 2, N_TEST)
    r4, surv4, eval4 = discover(tr4, te4, inst1)
    exp4 = (('a', 'd2'), ('a', 'a', 'd2'))
    report.append('T4 d2-doubling via S1: survivors=%d evaluated=%d rule=%s'
                  % (surv4, eval4, rule_str(r4) if r4 else 'NONE'))
    if r4 != exp4:
        report.append('T4 MISMATCH: got %s expected %s'
                      % (rule_str(r4) if r4 else None, rule_str(exp4)))
        ok = False
    # full-space baseline for the cost comparison
    _, _, eval4full = discover(tr4, te4, pruned)
    report.append('T4 cost: schema-guided %d vs full-space %d (%.1fx reduction)'
                  % (eval4, eval4full, eval4full / max(1, eval4)))

    # ---- Compression: schema S2 (c-catalyzed N-doubling, numeric) ----
    S2 = anti_unify_numeric(r1, r3) if (r1 and r3) else None
    report.append('S2 schema: %s' % (S2,))
    if not (S2 and S2[0] == {'a': 1, 'c': 1}
            and S2[1].get('c') == 1 and isinstance(S2[1].get('a'), str)):
        report.append('S2 MISMATCH: %s' % (S2,))
        ok = False
    inst2 = instantiate_numeric(S2, 4) if S2 else []
    report.append('S2 instantiations (N=0..4): %d' % len(inst2))

    # ---- Reuse Task 5: c-catalyzed quadrupling via S2 ----
    tr5 = gen_pairs(rng, 'c', 4, N_TRAIN)
    te5 = gen_pairs(rng, 'c', 4, N_TEST)
    r5, surv5, eval5 = discover(tr5, te5, inst2)
    exp5 = (('a', 'c'), ('a', 'a', 'a', 'a', 'c'))
    report.append('T5 c-quadrupling via S2: survivors=%d evaluated=%d rule=%s'
                  % (surv5, eval5, rule_str(r5) if r5 else 'NONE'))
    if r5 != exp5:
        report.append('T5 MISMATCH: got %s expected %s'
                      % (rule_str(r5) if r5 else None, rule_str(exp5)))
        ok = False
    _, _, eval5full = discover(tr5, te5, pruned)
    report.append('T5 cost: schema-guided %d vs full-space %d (%.1fx reduction)'
                  % (eval5, eval5full, eval5full / max(1, eval5)))

    dt = time.time() - t0
    report.append('wall time: %.1fs' % dt)
    report.append('OVERALL: %s' % ('PASS' if ok else 'FAIL'))
    print('\n'.join(report))
    return ok


if __name__ == '__main__':
    raise SystemExit(0 if main() else 1)
