import numpy as np
import pytest

from qws.ising import IsingProblem


def random_problem(n: int, rng: np.random.Generator, density: float = 0.5, ferro: bool = False,
                   scale: float = 1.0) -> IsingProblem:
    pairs = [(i, k) for i in range(n) for k in range(i + 1, n) if rng.random() < density]
    edges = np.array(pairs, dtype=np.int64).reshape(-1, 2)
    J = rng.uniform(0, 1, len(pairs)) if ferro else rng.normal(0, 1, len(pairs))
    return IsingProblem(rng.normal(0, 1, n) * scale, edges, J * scale, rng.normal())


@pytest.fixture
def rng():
    return np.random.default_rng(1234)


@pytest.fixture
def doc02_example() -> IsingProblem:
    """Worked example of spec doc 02 §8."""
    return IsingProblem(np.array([1.5, 0.5, -1.0]), np.array([[0, 1], [1, 2]]), np.array([0.5, 0.5]))
