"""K-class extension: Potts label model, one-hot QUBO encoding (paper Proposition 11), samplers."""

from .model import (
    MulticlassParams, MulticlassVotes, fit_multiclass, gradient_multiclass, model_constants, multiclass_dataset,
    posterior_potts, sample_moments, sample_multiclass_votes,
)
from .potts import (
    PottsGibbs, PottsProblem, all_labelings, decode_onehot, exact_potts, onehot_index, to_onehot_ising,
    to_onehot_qubo,
)

__all__ = [
    "MulticlassParams", "MulticlassVotes", "fit_multiclass", "gradient_multiclass", "model_constants",
    "multiclass_dataset", "posterior_potts", "sample_moments", "sample_multiclass_votes", "PottsGibbs",
    "PottsProblem", "all_labelings", "decode_onehot", "exact_potts", "onehot_index", "to_onehot_ising",
    "to_onehot_qubo",
]
