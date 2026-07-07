"""Sampling for parameter estimation."""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import emcee
import nautilus
import numpy as np
import numpy.typing as npt

from .model import IndexMap, Model
from .prob import BayesianProbability

__all__ = ["PosteriorSamples", "linear_teff", "sample_mcmc", "sample_nautilus"]


@dataclass(frozen=True)
class PosteriorSamples:
    """Weighted or unweighted samples estimating the posterior.

    Parameters
    ----------
    samples : ndarray of shape (..., P)
        Samples of the parameters from the posterior. The last axis corresponds
        to the parameters.
    weights : ndarray of shape (...,), optional
        Optional weights for each sample. If no weights are provided, samples
        have equal weight.

    Attributes
    ----------
    is_weighted : bool
        Whether the samples are weighted.
    """

    samples: npt.NDArray[np.float32]
    weights: npt.NDArray[np.float32] | None = None

    @property
    def is_weighted(self) -> bool:
        """Whether these samples are weighted.

        Returns
        -------
        weighted : bool
            True if the samples are weighted. False if they are equally
            weighted.
        """
        return self.weights is not None

    def quantile(
        self, q: float | Sequence[float] | npt.NDArray[np.float32]
    ) -> npt.NDArray[np.float32]:
        """Quantiles of sampled parameter values.

        q : array_like of float
            Quantiles to compute. Values must be between 0 and 1 inclusive.

        Returns
        -------
        quantiles : ndarray
            If one quantile is given, the array has shape (P,) where P is the
            number of parameters. If multiple quantiles are given, the array has
            shape (Q, P) where Q is the number of quantiles.

        Notes
        -----
        Quantiles are comptued using "inverted_cdf" in `numpy.quantile`.
        This supports both equal-weight and weighted samples. For a large number
        of samples, the difference between "inverted_cdf" and the default
        "linear" method is negligible.
        """
        return np.quantile(
            self.samples,
            q,
            axis=0,
            weights=self.weights,
            method="inverted_cdf",
        ).astype(np.float32)

    def model_quantile(
        self,
        model: Model,
        q: float | Sequence[float] | npt.NDArray[np.float32],
    ) -> npt.NDArray[np.float32]:
        """Quantiles of model values at sampled parameters.

        q : array_like of float
            Quantiles to compute. Values must be between 0 and 1 inclusive.
        model : Model
            Model used for forward modeling parameters.

        Returns
        -------
        quantiles : ndarray
            If one quantile is given, the array has shape (M,) where M is the
            size of modeling one set of parameters. If multiple quantiles are
            given, the array has shape (Q, M) where Q is the number of
            quantiles.

        Notes
        -----
        Quantiles are comptued using "inverted_cdf" in `numpy.quantile`.
        This supports both equal-weight and weighted samples. For a large number
        of samples, the difference between "inverted_cdf" and the default
        "linear" method is negligible.
        """
        return np.quantile(
            model(self.samples),
            q,
            axis=0,
            weights=self.weights,
            method="inverted_cdf",
        ).astype(np.float32)


def linear_teff(
    a: npt.NDArray[np.float32], axes_map: IndexMap, inplace=False
) -> npt.NDArray[np.float32]:
    """Convert effective temperature from logarithmic to linear space.

    Parameters
    ----------
    a : ndarray of shape (..., P)
        Array with logarithmic effective temperature values to convert.
        Parameters are along the last axis.
    axes_map : IndexMap
        Index mapping followed by the last axis of the input array.
    inplace : bool, default=False
        If True, the array will be modified in-place.

    Returns
    -------
    out : ndarray of shape (..., P)
        Array with logarithmic effective temperature valeus converted to linear.
    """
    if inplace:
        out = a
    else:
        out = a.copy()
    i = axes_map["teff"]
    out[..., i] = 10 ** out[..., i]
    return out


def sample_mcmc(
    prob: BayesianProbability,
    obs_flux: npt.NDArray[np.float32],
    obs_error: npt.NDArray[np.float32],
    *,
    n_walkers: int = 100,
    n_steps: int = 100000,
    backend_file: str | Path | None = None,
    progress_bar: bool = True,
) -> PosteriorSamples:
    """Sample the posterior using Markov Chain Monte Carlo.

    Parameters
    ----------
    prob : BayesianProbability
        Probability model used for sampling.
    obs_flux : ndarray of shape (N,)
        Object flux to fit. Must have the same last axis length as the model
        used in prob.
    obs_error : ndarray of shape (N,)
        Object error.
    n_walkers : int, default=100
        Number of walkers sampling simultaneously.
    n_steps : int, default=100000
        Maximum number of steps to take. Fewer steps may be taken if convergence
        is detected early.
    backend_file : str or Path, optional
        Path to a backend file that stores chains. If not given, no backend file
        is used and chains are only stored in memory.
    progress_bar : bool, default=True
        Display a progress bar while sampling.

    Returns
    -------
    samples : PosteriorSamples
        Unweighted samples of the posterior after discarding burn-in and
        thinning.
    """
    rng = np.random.default_rng()
    n_dim = prob.model.n_params
    walker_init = prob.prior_transform(
        rng.uniform(0, 1, (n_walkers, n_dim)).astype(np.float32)
    )

    if backend_file is not None:
        backend = emcee.backends.HDFBackend(backend_file)
        backend.reset(n_walkers, n_dim)
    else:
        backend = None

    sampler = emcee.EnsembleSampler(
        n_walkers,
        n_dim,
        prob.log_posterior,
        moves=[
            (emcee.moves.DEMove(), 0.8),
            (emcee.moves.DESnookerMove(), 0.2),
        ],
        args=(obs_flux, obs_error),
        backend=backend,
        vectorize=True,
    )

    # Run for up to n_steps, checking autocorrelation every 1000 steps.
    # Stop early if the chain is longer than 100x the autocorrelation time and that
    # estimate has changed by less than 1% from the previous iteration.
    prev_tau = np.inf
    for sample in sampler.sample(
        walker_init,
        iterations=n_steps,
        progress=progress_bar,
        progress_kwargs={"ncols": 100},
    ):
        if sampler.iteration % 1000 != 0:
            # Only check every 1000 iterations for efficiency.
            continue
        tau = sampler.get_autocorr_time(tol=0)
        if np.all(100 * tau < sampler.iteration) and np.all(
            np.abs(prev_tau - tau) / tau < 0.01
        ):
            # We have converged early.
            break
        prev_tau = tau

    tau = sampler.get_autocorr_time(quiet=True)
    # Number of initial steps that are skipped.
    burn_in = int(2 * np.max(tau))
    # Only include every thin steps to save memory.
    thin = int(np.min(tau) / 2)
    samples = sampler.get_chain(discard=burn_in, thin=thin, flat=True)
    return PosteriorSamples(samples)


def sample_nautilus(
    prob: BayesianProbability,
    obs_flux: npt.NDArray[np.float32],
    obs_error: npt.NDArray[np.float32],
    *,
    n_live: int = 2000,
    backend_file: Path | str | None = None,
    progress_bar: bool = True,
) -> PosteriorSamples:
    """Sample the posterior using Nautilus Nested Sampling.

    Parameters
    ----------
    prob : BayesianProbability
        Probability model used for sampling.
    obs_flux : ndarray of shape (N,)
        Object flux to fit. Must have the same last axis length as the model
        used in prob.
    obs_error : ndarray of shape (N,)
        Object error.
    n_live : int, default=2000
        Number of live points.
    backend_file : str or Path, optional
        Path to a backend file that stores results. If not given, no backend
        file is used.
    progress_bar : bool, default=True
        Display progress while sampling.

    Returns
    -------
    samples : PosteriorSamples
        Weighted samples of the posterior.
    """
    n_dim = prob.model.n_params
    sampler = nautilus.Sampler(
        prob.prior_transform,
        prob.log_likelihood,
        likelihood_kwargs={"obs_flux": obs_flux, "obs_error": obs_error},
        n_dim=n_dim,
        n_live=n_live,
        vectorized=True,
        filepath=backend_file,
    )
    sampler.run(verbose=progress_bar)
    samples, log_w, log_l = sampler.posterior()
    weights = np.exp(log_w)
    return PosteriorSamples(samples, weights)
