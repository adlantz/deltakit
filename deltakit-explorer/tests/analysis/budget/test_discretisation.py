import itertools

import numpy as np
import numpy.typing as npt
import pytest

from deltakit_explorer.analysis.error_budget._discretisation import (
    DiscretisationStrategy,
    GradientFitDiscretisationGenerator,
    get_c_optimal_points,
    get_linear_points,
    get_logarithmic_points,
)
from deltakit_explorer.analysis.error_budget._gradient import (
    _get_variance_of_gradient_estimation_at_point,
)


def _assert_is_linear(arr: npt.NDArray[np.floating]) -> None:
    diff = np.abs(arr[1:] - arr[:-1])
    np.testing.assert_allclose(diff - diff[0], 0, atol=1e-7)


@pytest.mark.parametrize(
    ("a", "b", "c", "num_points", "degree"),
    itertools.product([-1, 0, 0.1], [1, 2], [0.5], [5, 10, 1000], [1, 2, 3]),
)
def test_linear_points(
    a: float, b: float, c: float, num_points: int, degree: int
) -> None:
    ret = get_linear_points(a, b, c, num_points, degree)
    assert len(ret) == num_points
    assert np.all(np.logical_and(a <= ret, ret <= b))
    _assert_is_linear(ret)


@pytest.mark.parametrize(
    ("a", "b", "c", "num_points", "degree"),
    itertools.product([0.1, 0.5, 1.0], [1.1, 2.0, 5.0], [1.05], [5, 10], [1, 2, 3]),
)
def test_logarithmic_points(
    a: float, b: float, c: float, num_points: int, degree: int
) -> None:
    ret = get_logarithmic_points(a, b, c, num_points, degree)
    assert len(ret) == num_points
    eps = 1e-7
    assert np.all(np.logical_and(a <= ret + eps, ret <= b + eps))
    _assert_is_linear(np.log10(ret))


@pytest.mark.parametrize(
    ("func", "abc"),
    itertools.product(
        [get_linear_points, get_logarithmic_points],
        [
            (1, 2, 3),  # a < b < c
            (2, 1, 3),  # b < a < c
            (3, 1, 2),  # b < c < a
            (2, 3, 1),  # c < a < b
            (3, 2, 1),  # c < b < a
        ],
    ),
)
def test_raises_on_invalid_inputs(
    func: GradientFitDiscretisationGenerator, abc: tuple[float, float, float]
) -> None:
    a, b, c = abc
    with pytest.raises(ValueError, match=f"Expected {a=} < {c=} < {b=}"):
        func(a, b, c, 5, 3)


@pytest.mark.parametrize(
    "abc",
    [
        (1.0, 2.0, 3.0),  # a < b < c
        (2.0, 1.0, 3.0),  # b < a < c
        (3.0, 1.0, 2.0),  # b < c < a
        (2.0, 3.0, 1.0),  # c < a < b
        (3.0, 2.0, 1.0),  # c < b < a
    ],
)
def test_c_optimal_raises_on_invalid_inputs(
    abc: tuple[float, float, float],
) -> None:
    """C-optimal coerces ``c`` to ``float`` internally (to handle the length-1
    array case from ``generate_sweep_parameters``), so the error message
    formats ``c`` as a float. Use float inputs throughout for a stable match.
    """
    a, b, c = abc
    with pytest.raises(ValueError, match=f"Expected {a=} < {c=} < {b=}"):
        get_c_optimal_points(a, b, c, 5, 3)


def test_raise_on_negative_inputs_log() -> None:
    with pytest.raises(
        ValueError,
        match="Cannot get logarithmically-spaced points for negative values.*",
    ):
        get_logarithmic_points(-1, 1, 0, 5, 3)


# ---------------------------------------------------------------------------
# C-optimal tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("a", "b", "num_points", "degree"),
    itertools.product([2e-3, 1e-2], [5e-2, 1e-1], [5, 10, 15], [1, 2, 3]),
)
def test_c_optimal_points_shape_and_bounds(
    a: float, b: float, num_points: int, degree: int
) -> None:
    c = (a + b) / 2
    ret = get_c_optimal_points(a, b, c, num_points, degree)
    assert len(ret) == num_points
    assert np.all(np.logical_and(a <= ret, ret <= b))
    # Must be sorted (part of the protocol contract)
    assert np.all(ret[:-1] <= ret[1:])


def test_c_optimal_raises_below_minimum_num_points() -> None:
    with pytest.raises(ValueError, match="must sample at least"):
        get_c_optimal_points(2e-3, 1e-2, 7e-3, 3, 3)


@pytest.mark.parametrize("bad_c", [np.array([7e-3]), np.array([[7e-3]]), np.array([6e-3, 8e-3])])
def test_c_optimal_rejects_non_scalar_c(bad_c: npt.NDArray[np.floating]) -> None:
    """``c`` must be a scalar float. A non-scalar (e.g. a length-1 array)
    silently corrupts the slope-variance objective via broadcasting in
    ``_get_variance_of_gradient_estimation_at_point``, so it is rejected
    loudly rather than coerced. Callers must pass a scalar (see
    ``generate_sweep_parameters``, which flattens ``central_point`` first)."""
    with pytest.raises(ValueError, match="c must be a scalar"):
        get_c_optimal_points(2e-3, 1e-2, bad_c, 10, 3)


def test_c_optimal_is_deterministic() -> None:
    """Fixed ``seed=0`` in ``differential_evolution`` -> identical results
    across calls."""
    pts1 = get_c_optimal_points(2e-3, 1e-2, 7e-3, 10, 3)
    pts2 = get_c_optimal_points(2e-3, 1e-2, 7e-3, 10, 3)
    np.testing.assert_array_equal(pts1, pts2)


@pytest.mark.parametrize(
    ("a", "b"),
    [
        (1e-5, 1e-3),  # very small noise regime
        (1e-3, 1e-2),  # default
        (1e-2, 5e-2),  # higher noise regime
    ],
)
def test_c_optimal_produces_wellconditioned_designs(a: float, b: float) -> None:
    """The cond threshold should prevent the optimizer from picking rank-
    deficient designs, even across orders of magnitude in x-scale."""
    c = (a + b) / 2
    pts = get_c_optimal_points(a, b, c, 10, 3)
    # Rescale to [-1, 1] (same as the objective does internally) and check
    # the conditioning is comfortably below the threshold.
    u = 2 * (pts - pts.min()) / (pts.max() - pts.min()) - 1
    X = np.vander(u, 4, increasing=True)
    assert np.linalg.cond(X.T @ X) < 1e10


@pytest.mark.parametrize("c", [6e-3, 7e-3])
def test_c_optimal_beats_linear_on_slope_variance_at_c(c: float) -> None:
    """The whole point of c-optimal: it should minimise slope-variance at
    ``c`` versus the linear baseline. Validated under unit-weight
    homoscedastic noise (no W weighting needed here — we compare the
    underlying objective ``g(c)^T (X^T X)^{-1} g(c)`` directly).

    Tested at c-values reasonably centered in the interval. ``c`` values
    further from the centre can land ``differential_evolution`` in a
    sub-optimal local minimum at ``seed=0`` (a known limitation of the
    heuristic search — the c-optimal criterion itself is well defined).
    Production users typically have ``c = P/2`` in the interior, where the
    optimizer is reliable.
    """
    a, b, degree, num_points = 2e-3, 1e-2, 3, 10

    c_pts = get_c_optimal_points(a, b, c, num_points, degree)
    linear_pts = get_linear_points(a, b, c, num_points, degree)

    def slope_var(pts: npt.NDArray[np.floating]) -> float:
        # Same rescaling as inside the c-optimal objective.
        u = 2 * (pts - a) / (b - a) - 1
        uc = 2 * (c - a) / (b - a) - 1
        X = np.vander(u, degree + 1, increasing=True)
        cov = np.linalg.inv(X.T @ X)
        return _get_variance_of_gradient_estimation_at_point(cov, uc)

    # Allow 1% tolerance for differential_evolution stochasticity.
    assert slope_var(c_pts) <= slope_var(linear_pts) * 1.01


def test_discretisation_strategy_exposes_c_optimal() -> None:
    assert hasattr(DiscretisationStrategy, "C_OPTIMAL")


@pytest.mark.parametrize(
    ("num_points", "degree"), itertools.product([5, 10], [1, 2, 3])
)
def test_discretisation_strategy_dispatches_c_optimal(
    num_points: int, degree: int
) -> None:
    pts = DiscretisationStrategy.C_OPTIMAL(2e-3, 1e-2, 7e-3, num_points, degree)
    assert len(pts) == num_points
    assert np.all((pts >= 2e-3) & (pts <= 1e-2))
