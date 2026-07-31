"""_sint_batch_exact must be BIT-IDENTICAL to the scalar sint().

coldh's batched inner loop replaced ~85M scalar sint() calls with
_sint_batch_exact; the expected ortho/para-H tapes are exact-0.0 matches to
NJOY, so ANY drift (1-ulp transcendental differences, reassociated
multiplies, lost exact-hit early exits) is a regression. The scalar sint()
is kept as the semantic reference and this test pins the equivalence over
every branch: SCT beyond beta_max (both signs), interior exact grid hits
(raw sex return), grid-endpoint hits (log-interp fall-through), ordinary
interpolation, zero-S table points (slim sentinel), and underflow.
"""
import numpy as np
from math import exp, log

from irma.core.kernels import bfill, exts, sint, _sint_batch_exact


def _coldh_like_tables(seed=12345):
    """Build (bex, rdbex, sex, betan) the way coldh does, with a few
    zero-S points to exercise the slim sentinel."""
    rng = np.random.RandomState(seed)
    nbeta = 25
    betan = np.cumsum(0.05 + rng.rand(nbeta) * 0.4)
    betan[0] = 0.0  # coldh grids start at beta=0
    bex, rdbex, nbx = bfill(betan, nbeta)

    sexpb = np.exp(-0.5 * betan) * (0.1 + rng.rand(nbeta))
    sexpb[7] = 0.0                       # zero-S point -> slim branch
    sexpb[nbeta - 1] = 1.0e-200          # near-underflow tail
    exb = np.exp(-betan / 2.0)
    sex = exts(sexpb, exb, betan, nbeta)
    return bex, rdbex, sex, nbx, betan, nbeta


def test_batch_matches_scalar_bit_for_bit_over_every_branch():
    bex, rdbex, sex, nbx, betan, nbeta = _coldh_like_tables()
    beta_max = betan[nbeta - 1]
    al, wt, tbart = 3.7, 1.25, 1.9

    rng = np.random.RandomState(99)
    xs = np.concatenate([
        rng.uniform(-beta_max, beta_max, 400),       # ordinary interp
        bex[1:nbx - 1],                              # every interior hit
        [bex[0], bex[nbx - 1]],                      # endpoint fall-through
        [-beta_max - 0.5, beta_max + 0.5,            # SCT both signs
         -3.0 * beta_max, 3.0 * beta_max],
        [0.0],                                       # beta=0 node
    ])
    log_sex = np.array([log(s) if s > 0.0 else -225.0 for s in sex])

    got = _sint_batch_exact(xs, bex, rdbex, sex, log_sex, nbx,
                            al, wt, tbart, beta_max)
    want = np.array([sint(x, bex, rdbex, sex, nbx, al, wt, tbart,
                          betan, nbeta) for x in xs])
    # bit identity, not approx: the expected tapes are exact-0.0 matches
    assert got.tobytes() == want.tobytes()


def test_interior_exact_hit_returns_raw_sex_not_exp_log():
    """The bisection's early exit returns sex[k] UNTRANSFORMED; exp(log(s))
    can differ by 1 ulp. coldh hits this systematically at betap=0."""
    bex, rdbex, sex, nbx, betan, nbeta = _coldh_like_tables()
    beta_max = betan[nbeta - 1]
    log_sex = np.array([log(s) if s > 0.0 else -225.0 for s in sex])
    k = nbx // 2 + 1  # interior node with sex > 0
    assert sex[k] > 0.0
    got = _sint_batch_exact(np.array([bex[k]]), bex, rdbex, sex, log_sex,
                            nbx, 2.0, 1.0, 1.5, beta_max)
    assert got[0] == sex[k]                      # raw table value
    roundtrip = exp(log(sex[k]))
    if roundtrip != sex[k]:                      # 1-ulp case actually occurs
        assert got[0] != roundtrip


def test_sct_branch_zero_alpha_returns_zero():
    bex, rdbex, sex, nbx, betan, nbeta = _coldh_like_tables()
    beta_max = betan[nbeta - 1]
    log_sex = np.array([log(s) if s > 0.0 else -225.0 for s in sex])
    x = np.array([beta_max + 1.0, -(beta_max + 1.0)])
    got = _sint_batch_exact(x, bex, rdbex, sex, log_sex, nbx,
                            0.0, 1.0, 1.5, beta_max)
    want = [sint(v, bex, rdbex, sex, nbx, 0.0, 1.0, 1.5, betan, nbeta)
            for v in x]
    assert list(got) == want == [0.0, 0.0]
