import numpy as np
import pytest

from qws.ising import (
    IsingProblem, bits_to_spins, condition, ising_to_qubo, pauli_terms, qubo_to_ising, spins_to_bits,
    strong_field_fix,
)
from qws.solvers import ExactSolver, all_spins

from .conftest import random_problem


def test_doc02_worked_example(doc02_example):
    p = doc02_example
    Y = all_spins(3)
    E = p.energy(Y)
    best = Y[np.argmin(E)]
    assert best.tolist() == [1, 1, -1]
    assert E.min() == pytest.approx(-3.0)
    assert sorted(E)[1] == pytest.approx(-2.0) and sorted(E)[2] == pytest.approx(-2.0)
    q = ising_to_qubo(p)
    assert q.linear.tolist() == pytest.approx([-2.0, 1.0, 3.0])
    assert q.quadratic.tolist() == pytest.approx([-2.0, -2.0])
    assert q.offset == pytest.approx(0.0)
    assert q.energy(np.array([1, 1, 0]))[0] == pytest.approx(-3.0)


@pytest.mark.parametrize("n", [1, 2, 5, 8, 12])
def test_qubo_energy_identity(n, rng):
    for _ in range(3):
        p = random_problem(n, rng)
        Y = all_spins(n)
        q = ising_to_qubo(p)
        np.testing.assert_allclose(q.energy(spins_to_bits(Y)), p.energy(Y), atol=1e-9)
        back = qubo_to_ising(q)
        np.testing.assert_allclose(back.energy(Y), p.energy(Y), atol=1e-9)
    assert np.array_equal(bits_to_spins(spins_to_bits(Y)), Y)


def test_pauli_terms_match_energy(rng):
    p = random_problem(5, rng)
    Y = all_spins(5)  # y_i = Z_i eigenvalue
    diag = np.zeros(len(Y))
    for label, coeff in pauli_terms(p):
        term = np.ones(len(Y))
        for i, ch in enumerate(label):
            if ch == "Z":
                term = term * Y[:, i]
        diag += coeff * term
    np.testing.assert_allclose(diag, p.energy(Y), atol=1e-9)


def test_dimod_sign_convention(rng):
    dimod = pytest.importorskip("dimod")
    from qws.ising import to_dimod_bqm

    p = random_problem(6, rng)
    bqm = to_dimod_bqm(p)
    for y in all_spins(6)[::7]:
        assert bqm.energy({i: int(v) for i, v in enumerate(y)}) == pytest.approx(p.energy(y))


def test_duplicate_and_ordering_validation():
    with pytest.raises(ValueError):
        IsingProblem(np.zeros(3), np.array([[0, 1], [1, 0]]), np.array([1.0, 1.0]))
    with pytest.raises(ValueError):
        IsingProblem(np.zeros(3), np.array([[1, 1]]), np.array([1.0]))
    p = IsingProblem.from_dicts({0: 1.0}, {(1, 0): 0.5, (0, 1): 0.25, (2, 1): -1.0})
    assert p.n == 3 and p.m == 2
    assert dict(zip(map(tuple, p.edges.tolist()), p.J)) == {(0, 1): 0.75, (1, 2): -1.0}


def test_conditioning_is_exact(rng):
    p = random_problem(8, rng)
    fixed = np.array([1, 4, 6])
    vals = np.array([1, -1, 1], dtype=np.int8)
    c = condition(p, fixed, vals)
    Yr = all_spins(c.problem.n)
    full = c.lift(Yr, p.n)
    # energies agree up to nothing (offset included)
    np.testing.assert_allclose(c.problem.energy(Yr), p.energy(full), atol=1e-9)
    # conditional distribution equals the restriction of the joint
    ex = ExactSolver()
    _, pr = ex.distribution(c.problem)
    joint = np.exp(-p.energy(full))
    np.testing.assert_allclose(pr, joint / joint.sum(), atol=1e-12)


def test_strong_field_fixing_preserves_minimizers(rng):
    ex = ExactSolver()
    for _ in range(20):
        p = random_problem(10, rng, density=0.3)
        p = IsingProblem(p.h * 3, p.edges, p.J, p.offset)  # make some fields strong
        c = strong_field_fix(p)
        ground = ex.minimize([p])[0]
        for y in ground.samples:
            assert np.all(y[c.fixed_idx] == c.fixed_val)
        if c.problem.n:
            sub = ex.minimize([c.problem])[0]
            assert sub.best()[1] == pytest.approx(ground.best()[1])


def test_scaled_and_dynamic_range(doc02_example):
    p = doc02_example.scaled(2.0)
    np.testing.assert_allclose(p.energy(all_spins(3)), 2 * doc02_example.energy(all_spins(3)))
    assert doc02_example.dynamic_range() == pytest.approx(1.5 / 0.5)
