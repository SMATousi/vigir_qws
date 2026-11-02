import numpy as np
import pytest

from qws.eval import synthetic_dataset
from qws.learn import CalibratedSampler, TrainConfig, Trainer, estimate_beta_pl, infer_map, infer_marginals
from qws.model import LabelModelParams
from qws.solvers import ExactSolver, GibbsSampler, GraphCutSolver, Solver

from .conftest import random_problem

TRUE = LabelModelParams(alpha=[0.5, 0.0, -0.5, 0.2], theta=[1.0, 0.6, 0.3, 0.8], alphaP=[0.0], thetaP=[0.9])


@pytest.mark.parametrize("beta0", [0.4, 1.0, 2.5])
def test_pl_estimator_recovers_temperature(beta0, rng):
    p = random_problem(10, rng, density=0.4, scale=0.5)
    ss = ExactSolver(seed=1).sample([p], 20000, beta=beta0)[0]
    assert estimate_beta_pl(p, ss.samples, ss.counts) == pytest.approx(beta0, rel=0.08)


def test_pl_zero_temperature_detected(doc02_example):
    ground = np.array([[1, 1, -1]])
    assert estimate_beta_pl(doc02_example, ground) >= 1e3 - 1e-6


class _MisTempered(Solver):
    """Exact sampler that runs at the wrong inverse temperature (like an uncalibrated QPU)."""

    name = "mistempered"

    def __init__(self, beta_eff):
        self.beta_eff, self.inner = beta_eff, ExactSolver(seed=0)

    def sample(self, problems, num_samples, **kw):
        return self.inner.sample(problems, num_samples, beta=self.beta_eff)


def test_calibrated_sampler_converges(rng):
    p = random_problem(9, rng, density=0.4, scale=0.5)
    cal = CalibratedSampler(_MisTempered(2.0), damping=1.0)
    for _ in range(6):
        cal.sample([p], 5000)
    assert cal.scale == pytest.approx(0.5, rel=0.1)
    assert cal.history[-1]["beta_hat"] == pytest.approx(1.0, abs=0.1)


@pytest.fixture(scope="module")
def data():
    ds = synthetic_dataset(TRUE, n_images=12, shape=(12, 12), smoothness=2.5, seed=5)
    return ds, [(im.lfo, im.sites) for im in ds]


@pytest.mark.slow
def test_trainer_recovers_parameters_gradient(data):
    _, d = data
    res = Trainer(GibbsSampler(num_sweeps=60, seed=0), TrainConfig(epochs=25, num_samples=60, beta=0.6)).fit(d)
    np.testing.assert_allclose(res.params.theta, TRUE.theta, atol=0.15)
    np.testing.assert_allclose(res.params.thetaP, TRUE.thetaP, atol=0.2)
    assert len(res.history) >= 1 and "max_param_change" in res.history[-1]


@pytest.mark.slow
def test_trainer_em_mode(data):
    _, d = data
    res = Trainer(GibbsSampler(num_sweeps=60, seed=0),
                  TrainConfig(mode="em", epochs=8, num_samples=60, beta=0.6)).fit(d)
    np.testing.assert_allclose(res.params.theta, TRUE.theta, atol=0.15)


def test_learning_beta_and_b_runs(data):
    _, d = data
    cfg = TrainConfig(epochs=2, num_samples=20, beta=0.3, learn_beta=True, learn_b=True)
    res = Trainer(GibbsSampler(num_sweeps=10, persistent=True, seed=0), cfg).fit(d[:3])
    assert res.params.beta >= 0 and res.params.beta != 0.3


def test_inference_helpers(data):
    ds, d = data
    params = TRUE.copy()
    params.beta = 0.6
    unary_only = [(type(lfo)(lfo.L, lfo.P[:, :0]), s) for lfo, s in d[:2]]
    p0 = params.copy()
    p0.alphaP, p0.thetaP = np.zeros(0), np.zeros(0)
    masks = infer_map(p0, unary_only, GraphCutSolver())  # ferromagnetic without pairwise LFs
    assert masks[0].shape == (d[0][1].n,) and set(np.unique(masks[0])) <= {-1, 1}
    mu = infer_marginals(params, d[:2], GibbsSampler(num_sweeps=20, seed=0), num_samples=50, polish_sweeps=2, seed=0)
    assert np.all((mu[0] >= 0) & (mu[0] <= 1))
