#!/usr/bin/env python3
"""
PROBE 3 — Conservation Reaction Substrate: the SCALE kill gate.

Follows Probe 2b (PASS at toy scale: alphabet 6, 12,936 raw -> 128 pruned
(101x), exact rule recovery on held-out, compression + measured reuse).
2b's explicit caveat was scale: "existence proof, not a scaling claim."
This probe tests exactly that.

KILL CRITERION (pre-declared 2026-10-18, honored exactly as written):
  if at alphabet size <= 14 the search exceeds 1,000,000 RAW candidates
  (the full |R|<=2 x |P|<=5 Cartesian product — the 2b definition of "raw"),
  OR exceeds 10 minutes wall time without finding the exact rule,
  verdict = KILL.

Experiments (stdlib only):
  1. Alphabet scale: n = 6 (2b replication), 10, 14, 20. Exact raw/pruned
     counts (closed-form combinatorics, asserted against brute force),
     prune factor, three enumeration strategies timed (S1 naive
     enumerate-then-filter as in 2b; S2 streaming filter; S3 charge-guided
     direct enumeration), T1 (c-catalyzed doubling) exact recovery.
  2a. Two-rule target system: JOINT search over rule pairs under single-pass
      firing semantics (each rule fires at most once, canonical order).
  2b. Two-catalyst rule: a+c1+c2 -> 2a+c1+c2 under |R|<=3 grammar.
  2c. Three-rule target system: joint triple enumeration measured for
      feasibility; then COMPOSITIONAL discovery (change-set clustering is
      unsupervised — no generator labels — per-substrate single-rule
      discovery, joint verification), because flat joint search is pruned^k.
  3. Prune-ratio curve as numbers + closed-form formulas.

SEMANTICS NOTE (found during this probe): naive FIXPOINT firing ("repeat
rules until no change") DIVERGES on autocatalytic rules like a+c->2a+c —
the product still contains the reactants, so the rule re-fires to the
iteration cap. This is not a bug in the probe; it is why the substrate needs
an explicit scheduling discipline (exactly Probe 1's subject: optimal-sharing
scheduling of reactions). Multi-rule experiments therefore use SINGLE-PASS
firing semantics (each rule fires at most once per step, canonical rule
order — deterministic and order-independent).

Prior art (honest, mandatory): DreamCoder (Ellis et al.) — enumerative
search + library compression over a DSL; this probe transfers that METHOD
to reaction-rule discovery and to its scaling analysis; no novelty claimed
over DreamCoder. Optimal-reduction lineage (HVM/HVM2/HVM3, BOHM, Optiscope)
is non-ML. Chemlambda: graph-rewriting chemistry, no learning. Battaglia
et al. 2016 "Interaction Networks" is an unrelated GNN — name disambiguated.
Claim: "clean for ML-native use; mature non-ML lineage," never "zero prior
art."

CHARGE DESIGN (same documented limitation as 2b): substrate types are
charge-neutral (forced by the doubling rule form); catalyst/distractors
carry distinct unit charges. The certificate prunes catalyst/distractor
flow; enumeration + behavioral data carry the doubling structure.

Deterministic (seed 20261018 + experiment offset). No post-hoc tuning.
Run: nice -n 19 python3 research/reaction_probe3.py
"""
import itertools
import math
import random
import time
from collections import Counter

SEED = 20261018
KILL_RAW = 1_000_000
KILL_WALL_S = 600  # per-discovery wall budget
N_TRAIN, N_TEST = 64, 32


# ---------------- exact combinatorics ----------------
def Cnk(n, k):
    return math.comb(n, k) if 0 <= k <= n else 0


def nmult(n, kmax):
    """Number of multisets of size 0..kmax from n types."""
    return sum(Cnk(n + j - 1, j) for j in range(kmax + 1))


def raw_count(n, max_r=2, max_p=5):
    return nmult(n, max_r) * nmult(n, max_p)


def pruned_count_1neutral(n, max_r=2, max_p=5):
    """Closed form, ONE neutral substrate type, n-1 distinct unit charges.

    charge(R)==charge(P) iff the charged multisets are equal (distinct unit
    vectors); the neutral count is free on each side. For charged part of
    size j: C((n-1)+j-1, j) charged multisets x (max_r-j+1) neutral
    fillings of R x (max_p-j+1) fillings of P.
    """
    return sum(Cnk(n + j - 2, j) * (max_r - j + 1) * (max_p - j + 1)
               for j in range(max_r + 1))


# ---------------- alphabets ----------------
def make_alphabet(neutral, charged):
    types = list(neutral) + list(charged)
    dim = len(charged)
    charge = {t: (0,) * dim for t in neutral}
    for i, t in enumerate(charged):
        v = [0] * dim
        v[i] = 1
        charge[t] = tuple(v)
    return types, charge


def std_alphabet(n):
    """2b-compatible: 'a' neutral; 'c','d1',... carry distinct unit charges."""
    return make_alphabet(('a',), ['c'] + ['d%d' % i for i in range(1, n - 1)])


# ---------------- multisets & charges ----------------
def all_multisets(types, kmax):
    out = []
    for k in range(kmax + 1):
        out.extend(itertools.combinations_with_replacement(types, k))
    return out


def charge_of(ms, charge):
    dim = len(next(iter(charge.values())))
    q = [0] * dim
    for t in ms:
        v = charge[t]
        for i in range(dim):
            q[i] += v[i]
    return tuple(q)


# ---------------- enumeration strategies ----------------
def enum_S1(types, charge, max_r=2, max_p=5):
    """Naive, faithful to 2b: materialize the full raw Cartesian product,
    then filter by charge conservation. Time AND memory scale with raw."""
    Rs = all_multisets(types, max_r)
    Ps = all_multisets(types, max_p)
    ch = {}
    for ms in Rs:
        ch[(0,) + ms] = charge_of(ms, charge)
    for ms in Ps:
        ch[(1,) + ms] = charge_of(ms, charge)
    raw = [(R, P) for R in Rs for P in Ps]
    nraw = len(raw)
    pruned = [(R, P) for (R, P) in raw if ch[(0,) + R] == ch[(1,) + P]]
    return nraw, pruned


def enum_S2(types, charge, max_r=2, max_p=5):
    """Streaming filter: touch every raw pair, keep only pruned.
    Time scales with raw (~n^7); memory scales with pruned (~n^2)."""
    Rs = all_multisets(types, max_r)
    Ps = all_multisets(types, max_p)
    chR = [charge_of(R, charge) for R in Rs]
    chP = [charge_of(P, charge) for P in Ps]
    nraw = 0
    pruned = []
    for i, R in enumerate(Rs):
        cr = chR[i]
        for j, P in enumerate(Ps):
            nraw += 1
            if cr == chP[j]:
                pruned.append((R, P))
    return nraw, pruned


def enum_S3(types, charge, max_r=2, max_p=5):
    """Charge-guided direct enumeration: group P multisets by charge, join
    each R only with charge-matching P's. Never materializes or touches the
    raw product. Time and memory scale with pruned (~n^2). Standard
    database-join technique; no new machinery."""
    Rs = all_multisets(types, max_r)
    Ps = all_multisets(types, max_p)
    by_charge = {}
    for P in Ps:
        by_charge.setdefault(charge_of(P, charge), []).append(P)
    pruned = []
    for R in Rs:
        for P in by_charge.get(charge_of(R, charge), ()):
            pruned.append((R, P))
    return pruned


# ---------------- rule semantics ----------------
def norm(c):
    return Counter({t: n for t, n in c.items() if n > 0})


def apply_rule(rule, soup):
    """Single application (the 2b semantics)."""
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
    res = apply_rule(rule, before)
    if res is None:
        return before == after
    return res == after


def apply_system(rules, soup):
    """SINGLE-PASS firing semantics for multi-rule systems: rules sorted to
    canonical order (deterministic, order-independent), each rule fires at
    most once. Fixpoint firing is NOT used: it diverges on autocatalytic
    rules (product re-triggers the rule), which is why the substrate needs
    an explicit scheduling discipline (cf. Probe 1)."""
    cur = Counter(soup)
    for (R, P) in sorted(rules):
        need = Counter(R)
        fire = True
        for t, n in need.items():
            if cur.get(t, 0) < n:
                fire = False
                break
        if fire:
            cur.subtract(need)
            cur.update(Counter(P))
    return norm(cur)


def matches_system(rules, before, after):
    return apply_system(rules, before) == after


def rule_str(r):
    s = lambda ms: '+'.join(ms) if ms else '0'
    return '%s -> %s' % (s(r[0]), s(r[1]))


def change_set(before, after):
    """Types whose counts changed. Distractors/catalysts are conserved by
    the generator, so this is the set of changed substrates — observable
    from data alone, no generator labels."""
    return frozenset(t for t in set(before) | set(after)
                     if before.get(t, 0) != after.get(t, 0))


# ---------------- data generation ----------------
def gen_pairs(rng, types, substrates, catalysts, factor, n):
    """Single-system pairs: reactive (all substrates + all catalysts ->
    substrates x factor); non-reactive (one required component dropped)."""
    distractors = [t for t in types
                   if t not in substrates and t not in catalysts]
    pairs = []
    for _ in range(n):
        D = Counter({d: 1 for d in distractors if rng.random() < 0.5})
        if rng.random() < 0.7:
            before = Counter()
            for s in substrates:
                before[s] = 1
            for c in catalysts:
                before[c] = 1
            before.update(D)
            after = Counter()
            for s in substrates:
                after[s] = factor
            for c in catalysts:
                after[c] = 1
            after.update(D)
        else:
            drop = rng.choice(list(substrates) + list(catalysts))
            before = Counter()
            for s in substrates:
                if s != drop:
                    before[s] = 1
            for c in catalysts:
                if c != drop:
                    before[c] = 1
            before.update(D)
            after = Counter(before)
        pairs.append((norm(before), norm(after)))
    return pairs


def gen_pairs_multi(rng, types, substrates, catalysts, factor, n):
    """Multi-substrate pairs: full-reactive (0.45), single-substrate partial
    (0.30) — these pin each rule individually and kill merged single rules —
    non-reactive (0.25) in TWO kinds: substrates-without-catalyst (kills
    catalyst-free rules like a->2a) and catalyst-without-substrates (kills
    catalyst-only rules). Both negative kinds are required; omitting the
    first makes catalyzed vs catalyst-free rules indistinguishable."""
    distractors = [t for t in types
                   if t not in substrates and t not in catalysts]
    pairs = []
    for _ in range(n):
        D = Counter({d: 1 for d in distractors if rng.random() < 0.5})
        r = rng.random()
        if r < 0.45:
            before = Counter({s: 1 for s in substrates})
            for c in catalysts:
                before[c] = 1
            before.update(D)
            after = Counter({s: factor for s in substrates})
            for c in catalysts:
                after[c] = 1
            after.update(D)
        elif r < 0.75:
            s = rng.choice(substrates)
            before = Counter({s: 1})
            for c in catalysts:
                before[c] = 1
            before.update(D)
            after = Counter({s: factor})
            for c in catalysts:
                after[c] = 1
            after.update(D)
        else:
            if rng.random() < 0.5:
                # substrates present, catalyst ABSENT
                before = Counter({s: 1 for s in substrates})
            else:
                # catalyst present, substrates absent
                before = Counter()
                for c in catalysts:
                    before[c] = 1
            before.update(D)
            after = Counter(before)
        pairs.append((norm(before), norm(after)))
    return pairs


# ---------------- discovery ----------------
def discover(pairs_train, pairs_test, candidates):
    """2b discovery: behavioral filter on train, pre-declared tie-break
    (fewest atoms, then lexicographic), single held-out evaluation."""
    surv = [r for r in candidates
            if all(matches(r, b, a) for b, a in pairs_train)]
    surv.sort(key=lambda r: (len(r[0]) + len(r[1]), r[0], r[1]))
    if not surv:
        return None, 0, len(candidates)
    best = surv[0]
    if not all(matches(best, b, a) for b, a in pairs_test):
        return None, len(surv), len(candidates)
    return best, len(surv), len(candidates)


def discover_set(pairs_train, pairs_test, singles, k, cap_s=240):
    """Joint search over k-rule systems (single-pass semantics)."""
    t0 = time.time()
    surv = []
    total = 0
    capped = False
    for combo in itertools.combinations(singles, k):
        total += 1
        if all(matches_system(combo, b, a) for b, a in pairs_train):
            surv.append(combo)
        if total % 2000 == 0 and time.time() - t0 > cap_s:
            capped = True
            break
    dt = time.time() - t0
    surv.sort(key=lambda c: (sum(len(R) + len(P) for R, P in c),
                             tuple(sorted(c))))
    best = tuple(sorted(surv[0])) if surv else None
    ok_held = (best is not None
               and all(matches_system(best, b, a) for b, a in pairs_test))
    return best, len(surv), total, dt, capped, ok_held


def mem_available_gb():
    try:
        with open('/proc/meminfo') as f:
            for line in f:
                if line.startswith('MemAvailable'):
                    return int(line.split()[1]) / 1e6
    except Exception:
        pass
    return 99.0


# ---------------- experiments ----------------
def exp1_scale():
    """Alphabet scale on the 2b T1 task (c-catalyzed doubling)."""
    rows = []
    for n in (6, 10, 14, 20):
        t0 = time.time()
        types, charge = std_alphabet(n)
        nraw_th = raw_count(n)
        npr_th = pruned_count_1neutral(n)

        # S3 always (fast, low memory); asserts the closed form.
        t = time.time()
        pruned = enum_S3(types, charge)
        tS3 = time.time() - t
        assert len(pruned) == npr_th, (n, len(pruned), npr_th)
        if n == 6:
            assert nraw_th == 12936 and npr_th == 128, '2b replication'

        # S1 (naive, faithful to 2b) where memory allows.
        tS1 = None
        if n <= 10 or (n == 14 and mem_available_gb() > 2.0):
            t = time.time()
            nraw1, pr1 = enum_S1(types, charge)
            tS1 = time.time() - t
            assert nraw1 == nraw_th and len(pr1) == npr_th

        # S2 (streaming) at 10+ for the time-scaling comparison.
        tS2 = None
        if n >= 10:
            t = time.time()
            nraw2, pr2 = enum_S2(types, charge)
            tS2 = time.time() - t
            assert nraw2 == nraw_th and len(pr2) == npr_th

        # Discovery (T1) on the S3 pruned set.
        rng = random.Random(SEED + n)
        tr = gen_pairs(rng, types, ('a',), ('c',), 2, N_TRAIN)
        te = gen_pairs(rng, types, ('a',), ('c',), 2, N_TEST)
        t = time.time()
        r1, nsurv, nev = discover(tr, te, pruned)
        tdisc = time.time() - t
        exp = (('a', 'c'), ('a', 'a', 'c'))
        ok = (r1 == exp and nsurv == 1)

        row = dict(n=n, raw=nraw_th, pruned=npr_th, factor=nraw_th / npr_th,
                   tS1=tS1, tS2=tS2, tS3=tS3, tdisc=tdisc,
                   rule=rule_str(r1) if r1 else None, surv=nsurv, ok=ok,
                   wall=time.time() - t0)
        rows.append(row)
        print('n=%2d raw=%10d pruned=%4d factor=%9.1fx | '
              'tS1=%7s tS2=%7s tS3=%6.2fs tdisc=%5.2fs | surv=%d rule=%s %s '
              '[%.1fs]' % (
                  n, nraw_th, npr_th, nraw_th / npr_th,
                  ('%.2fs' % tS1) if tS1 is not None else 'skipped',
                  ('%.2fs' % tS2) if tS2 is not None else 'n/a',
                  tS3, tdisc, nsurv, row['rule'],
                  'OK' if ok else 'FAIL', row['wall']),
              flush=True)
    return rows


def exp2a_tworule():
    """Two independent doublings sharing a catalyst; JOINT pair search."""
    types, charge = make_alphabet(('a', 'b'), ['c', 'd1', 'd2', 'd3'])
    pruned = enum_S3(types, charge)
    rng = random.Random(SEED + 100)
    tr = gen_pairs_multi(rng, types, ('a', 'b'), ('c',), 2, 96)
    te = gen_pairs_multi(rng, types, ('a', 'b'), ('c',), 2, 48)
    best, nsurv, total, dt, capped, okh = discover_set(tr, te, pruned, 2)
    exp = tuple(sorted([(('a', 'c'), ('a', 'a', 'c')),
                        (('b', 'c'), ('b', 'b', 'c'))]))
    ok = (best == exp and nsurv == 1 and okh and not capped)
    print('2-rule: pruned=%d pairs=%d time=%.1fs capped=%s surv=%d rule=%s %s'
          % (len(pruned), total, dt, capped, nsurv,
             '+'.join(rule_str(r) for r in best) if best else None,
             'OK' if ok else 'FAIL'),
          flush=True)
    return dict(name='2-rule', pruned=len(pruned), combos=total, dt=dt,
                capped=capped, surv=nsurv,
                rule='+'.join(rule_str(r) for r in best) if best else None,
                ok=ok)


def exp2b_twocat():
    """Two-catalyst rule a+c1+c2 -> 2a+c1+c2 under |R|<=3 grammar."""
    types, charge = make_alphabet(('a',), ['c1', 'c2', 'd1', 'd2', 'd3'])
    nraw_th = raw_count(6, 3, 5)
    pruned = enum_S3(types, charge, max_r=3, max_p=5)
    rng = random.Random(SEED + 200)
    tr = gen_pairs(rng, types, ('a',), ('c1', 'c2'), 2, N_TRAIN)
    te = gen_pairs(rng, types, ('a',), ('c1', 'c2'), 2, N_TEST)
    r, nsurv, nev = discover(tr, te, pruned)
    exp = (('a', 'c1', 'c2'), ('a', 'a', 'c1', 'c2'))
    ok = (r == exp and nsurv == 1)
    print('2-catalyst: raw=%d pruned=%d (%.1fx) surv=%d rule=%s %s'
          % (nraw_th, len(pruned), nraw_th / len(pruned), nsurv,
             rule_str(r) if r else None, 'OK' if ok else 'FAIL'),
          flush=True)
    return dict(name='2-catalyst', raw=nraw_th, pruned=len(pruned),
                factor=nraw_th / len(pruned), surv=nsurv,
                rule=rule_str(r) if r else None, ok=ok)


def exp2c_threerule():
    """Three independent doublings. Flat joint search over triples is
    pruned^3 — measured for feasibility, then compositional discovery:
    unsupervised change-set clustering -> per-substrate single-rule
    discovery (train only) -> joint verification -> held-out."""
    types, charge = make_alphabet(('a', 'b', 's'), ['c', 'd1', 'd2'])
    pruned = enum_S3(types, charge)
    rng = random.Random(SEED + 300)
    tr = gen_pairs_multi(rng, types, ('a', 'b', 's'), ('c',), 2, 96)
    te = gen_pairs_multi(rng, types, ('a', 'b', 's'), ('c',), 2, 48)

    # (i) joint feasibility: exact triple count + timed sample extrapolation
    ntriples = Cnk(len(pruned), 3)
    t = time.time()
    sample = itertools.islice(itertools.combinations(pruned, 3), 2000)
    for combo in sample:
        all(matches_system(combo, b, a) for b, a in tr)
    tsample = time.time() - t
    textrap = tsample / 2000 * ntriples

    # (ii) compositional discovery via unsupervised change-set clustering.
    # Per-substrate discovery data = singleton-cluster reactive pairs +
    # ALL empty-change pairs as negatives. Both negative kinds are needed:
    # substrates-without-catalyst kills a->2a and 0->a; catalyst-without-
    # substrate kills c->a+c (which is otherwise behaviorally identical to
    # a+c->2a+c on reactive-only data and wins the atom-count tie-break).
    clusters = {}
    for b, a in tr:
        clusters.setdefault(change_set(b, a), []).append((b, a))
    neg = clusters.get(frozenset(), [])
    assert neg, 'no non-reactive pairs?'
    rules = []
    per_ok = True
    for cs in sorted(clusters, key=lambda c: sorted(c)):
        if len(cs) != 1:
            continue
        s = next(iter(cs))
        assert clusters[cs], 'empty singleton cluster for %s' % s
        data_tr = clusters[cs] + neg  # all empty-change pairs as negatives
        r, nsurv, nev = discover(data_tr, data_tr, pruned)
        per_ok = per_ok and (r is not None and nsurv == 1)
        if r:
            rules.append(r)
    triple = tuple(sorted(rules)) if len(rules) == 3 else None
    joint_train = (triple is not None
                   and all(matches_system(triple, b, a) for b, a in tr))
    joint_test = (triple is not None
                  and all(matches_system(triple, b, a) for b, a in te))
    exp = tuple(sorted([(('a', 'c'), ('a', 'a', 'c')),
                        (('b', 'c'), ('b', 'b', 'c')),
                        (('s', 'c'), ('s', 's', 'c'))]))
    ok = (triple == exp and per_ok and joint_train and joint_test)
    print('3-rule: pruned=%d triples=%d sample2000=%.1fs extrap=%.0fs '
          '-> joint INFEASIBLE' % (len(pruned), ntriples, tsample, textrap),
          flush=True)
    print('3-rule compositional: per-substrate rules=%d joint_train=%s '
          'joint_test=%s rule=%s %s'
          % (len(rules), joint_train, joint_test,
             '+'.join(rule_str(r) for r in triple) if triple else None,
             'OK' if ok else 'FAIL'),
          flush=True)
    return dict(name='3-rule', pruned=len(pruned), triples=ntriples,
                textrap=textrap, per_ok=per_ok, joint_train=joint_train,
                joint_test=joint_test,
                rule='+'.join(rule_str(r) for r in triple) if triple else None,
                ok=ok)


def main():
    t00 = time.time()
    print('=== PROBE 3: Conservation Reaction Substrate — SCALE kill gate ===',
          flush=True)
    print('kill if: raw > 1M at n<=14, or 10min wall without exact rule',
          flush=True)

    print('--- exp1: alphabet scale (T1 c-doubling) ---', flush=True)
    scale = exp1_scale()

    print('--- exp2a: two-rule system (joint) ---', flush=True)
    r2a = exp2a_tworule()

    print('--- exp2b: two-catalyst rule ---', flush=True)
    r2b = exp2b_twocat()

    print('--- exp2c: three-rule system (joint feasibility + compositional) ---',
          flush=True)
    r2c = exp2c_threerule()

    # ---- verdict: pre-declared criterion, honored exactly as written ----
    reasons = []
    for row in scale:
        if row['n'] <= 14 and row['raw'] > KILL_RAW:
            reasons.append('raw=%d exceeds 1M at n=%d'
                           % (row['raw'], row['n']))
        if not row['ok']:
            reasons.append('exact rule NOT recovered at n=%d' % row['n'])
        if row['wall'] > KILL_WALL_S and not row['ok']:
            reasons.append('wall %.0fs > 10min without rule at n=%d'
                           % (row['wall'], row['n']))
    killed = bool(reasons)
    print('--- verdict ---', flush=True)
    print('VERDICT: %s' % ('KILL' if killed else 'PASS'), flush=True)
    for r in reasons:
        print('  kill reason: %s' % r, flush=True)
    print('total wall: %.1fs' % (time.time() - t00), flush=True)
    return 1 if killed else 0


if __name__ == '__main__':
    raise SystemExit(main())
