# (c) Copyright Riverlane 2020-2025.
"""Description of ``deltakit.explorer.analysis`` namespace here."""

from deltakit_explorer.analysis._analysis import (
    get_exp_fit,
    get_lambda_fit,
)
from deltakit_explorer.analysis._binomial_fit import (
    DEFAULT_MAX_LIKELIHOOD_FACTOR,
    ConfidenceInterval,
    fit_binomial,
    fit_binomial_batch,
    fit_leppr_and_spam,
)
from deltakit_explorer.analysis._lambda import (
    LambdaData,
    calculate_lambda_and_lambda_stddev,
    calculate_lambda_asymmetric,
)
from deltakit_explorer.analysis._leppr import (
    LogicalErrorProbabilityPerRoundData,
    calculate_lep_and_lep_stddev,
    calculate_lep_asymmetric,
    compute_logical_error_per_round,
    fit_logical_error_per_round_asymmetric,
    simulate_different_round_numbers_for_lep_per_round_estimation,
)
from deltakit_explorer.analysis._quops import (
    predict_distance_for_quops,
    predict_quops_at_distance,
    predict_quops_interval,
)

from . import error_budget

# List only public members in `__all__`.
__all__ = [s for s in dir() if not s.startswith("_")]
