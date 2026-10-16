import numpy as np

from modlab.core import montecarlo, ode, pca


def test_rk4_matches_exponential_to_fourth_order():
    errs = []
    for h in (0.1, 0.05):
        t, Y = ode.rk4(lambda _, y: -y, [1.0], 2.0, h)
        errs.append(abs(Y[-1, 0] - np.exp(-2.0)))
    assert errs[0] < 1e-6
    assert 12 < errs[0] / errs[1] < 20          # halving h cuts the error ~16x


def test_rk4_events_apply_jumps():
    t, Y = ode.rk4(lambda _, y: 0 * y, [0.0], 1.0, 0.25, events=lambda tt, y: y + 1)
    assert Y[-1, 0] == 4


def test_pca_recovers_a_single_direction():
    g = np.random.default_rng(0)
    d = np.array([3.0, 4.0]) / 5
    X = g.standard_normal((500, 1)) * 2 * d + 0.01 * g.standard_normal((500, 2))
    m = pca.fit(X)
    assert abs(abs(m.modes[0] @ d) - 1) < 1e-3 and m.var_ratio[0] > 0.99
    np.testing.assert_allclose(m.rebuild(X, 2), X, atol=1e-9)


def test_max_drawdown():
    assert np.isclose(montecarlo.max_drawdown(np.log(np.array([[1, 2, 1, 3]]))), 0.5)
