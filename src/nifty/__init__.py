"""Near-Infrared Fitting for T and Y Dwarfs."""

from . import cli, load, model, plot, prob, sample
from .model import IndexMap, Model, ModelGrid, linear_teff
from .prob import BayesianProbability
from .sample import PosteriorSamples

try:
    from ._version import __version__
except ImportError:
    __version__ = "0.0.0"

__all__ = [
    "BayesianProbability",
    "IndexMap",
    "Model",
    "ModelGrid",
    "PosteriorSamples",
    "build_model",
    "cli",
    "linear_teff",
    "load",
    "model",
    "plot",
    "prob",
    "sample",
    "sample_mcmc",
    "sample_nautilus",
]
