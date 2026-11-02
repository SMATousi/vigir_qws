"""qws: quantum weak supervision.

Structured data programming for image segmentation, formulated as Ising/QUBO problems
that run on classical solvers, quantum annealers and gate-based QAOA. See ``spec/``.
"""

from .graph import Sites, grid_sites, superpixel_sites
from .ising import IsingProblem
from .lf import LFOutputs, apply_lfs
from .model import LabelModelParams, posterior_ising, prior_ising

__version__ = "0.1.0"
__all__ = [
    "IsingProblem", "Sites", "grid_sites", "superpixel_sites", "LFOutputs", "apply_lfs", "LabelModelParams",
    "posterior_ising", "prior_ising",
]
