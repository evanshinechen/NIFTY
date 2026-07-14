"""Bayesian probabilities for sampling."""

import numpy as np
import numpy.typing as npt

from .model import Model

__all__ = ["BayesianProbability"]


class BayesianProbability:
    """
    Bayesian probabilities of parameters using a model and observed data.

    Parameters
    ----------
    model : Model
        Model used for forward modeling parameters.
    frac_model_floor: float, default=0
        Fractional model flux floor added in quadrature to the reported errors
        to account for model systematics at the roughly 3% level.
    d_min : float, default=1
        Minimum distance in pc. Must be positive.
    d_max : float, default=2e4
        Maximum distance in pc.

    Attributes
    ----------
    bounds : ndarray of shape (2, N)
        Prior boundaries for parameters. bounds[0] is the minimum and bounds[1]
        is the maximum.
    """

    def __init__(
        self,
        model: Model,
        *,
        frac_model_floor: float = 0.0,
        d_min: float = 1e0,
        d_max: float = 2e4,
    ):
        self.model = model
        self.frac_model_floor = frac_model_floor
        self.axes_map = self.model.axes_map
        self._d_index = self.axes_map["d"]
        self._teff_index = self.axes_map["teff"]

        self.bounds = np.empty((2, model.n_params), dtype=np.float32)
        # Default no bounds for unfilled parameters.
        self.bounds[0] = np.full(model.n_params, -np.inf)
        self.bounds[1] = np.full(model.n_params, np.inf)
        # Get the upper and lower bounds from interp grid.
        interp_bounds = np.stack(
            tuple(axis[[0, -1]] for axis in model.interp.grid), axis=1
        )
        self.bounds[:, : model.n_interp_params] = interp_bounds
        if d_min <= 0:
            raise ValueError("d_min must be positive.")
        self.bounds[:, self._d_index] = np.array(
            [d_min, d_max], dtype=np.float32
        )

    def prior_transform(
        self, cube: npt.NDArray[np.float32]
    ) -> npt.NDArray[np.float32]:
        """
        Transform a unit hypercube into the parameter space following the prior.

        The prior is the same as the prior in log_prior.

        Parameters
        ----------
        cube : ndarray of shape (..., N)
            Each element of cube is a float between 0 to 1 for an axis.

        Returns
        -------
        theta : ndarray of shape (..., N)
            Parameters corresponding to cube.
        """
        # Basic linear flat prior.
        theta = (self.bounds[1] - self.bounds[0]) * cube + self.bounds[0]

        # Flat prior on teff, which is parameterized in log10(T_eff).
        teff_min = 10 ** self.bounds[0, self._teff_index]
        teff_max = 10 ** self.bounds[1, self._teff_index]
        theta[..., self._teff_index] = np.log10(
            (teff_max - teff_min) * cube[..., self._teff_index] + teff_min
        )

        # 1/d log-flat prior on distance.
        log_d_min = np.log(self.bounds[0, self._d_index])
        log_d_max = np.log(self.bounds[1, self._d_index])
        theta[..., self._d_index] = np.exp(
            (log_d_max - log_d_min) * cube[..., self._d_index] + log_d_min
        )

        return theta

    def log_prior(
        self, theta: npt.NDArray[np.float32]
    ) -> npt.NDArray[np.float32]:
        """
        Log prior for a set of parameters.

        The prior is flat, spanning the range of axes in the model with a weak 1/d
        log-flat prior on distance.

        Parameters
        ----------
        theta : ndarray of shape (..., N)
            Parameters following the order of axes_map.

        Returns
        -------
        log_prior : ndarray of shape (...,)
            Log prior for the given parameters.
        """
        in_bounds = np.all(
            (theta >= self.bounds[0]) & (theta <= self.bounds[1]),
            axis=-1,
        )
        out = np.full(theta.shape[:-1], -np.inf, dtype=np.float32)
        # The bounds for distance requires d > 0, so we can safely take the log.
        out[in_bounds] = -np.log(theta[..., self._d_index][in_bounds])
        # Correct for sampling in log(teff) by making p(theta) prop to teff.
        out[in_bounds] += np.log(theta[..., self._teff_index][in_bounds])
        return out

    def log_likelihood(
        self,
        theta: npt.NDArray[np.float32],
        obs_flux: npt.NDArray[np.float32],
        obs_error: npt.NDArray[np.float32],
    ) -> npt.NDArray[np.float32]:
        """
        Log likelihood for a set of parameters and observed values.

        Uses the log of the chi-squared value between the model and object values.

        Parameters
        ----------
        theta : ndarray of shape (..., N)
            Parameters following the order of axes_map.
        obs_flux : ndarray of shape (F,)
            Observed object flux in nJy.
        obs_error : ndarray of shape (F,)
            Observed object error in nJy.

        Returns
        -------
        log_likelihood : ndarray of shape (...,)
            Log likelihood for the given parameters and observation.
        """
        valid = (~np.isnan(obs_flux)) & (obs_error > 0)
        model_flux = self.model(theta)
        variance = np.square(obs_error) + np.square(
            self.frac_model_floor * model_flux
        )
        chi_squared = np.sum(
            np.where(valid, np.square(obs_flux - model_flux) / variance, 0.0),
            axis=-1,
        )
        return np.where(np.isfinite(chi_squared), -0.5 * chi_squared, -np.inf)

    def log_posterior(
        self,
        theta: npt.NDArray[np.float32],
        obs_flux: npt.NDArray[np.float32],
        obs_error: npt.NDArray[np.float32],
    ) -> npt.NDArray[np.float32]:
        """
        Log posterior for a set of parameters and observed values.

        Uses log_prior and log_likelihood.

        Parameters
        ----------
        theta : ndarray of shape (..., N)
            Parameters following the order of axes_map.
        obs_flux : ndarray of shape (F,)
            Observed object flux in nJy.
        obs_error : ndarray of shape (F,)
            Observed object error in nJy.

        Returns
        -------
        log_posterior : ndarray of shape (...,)
            Log posterior for the given parameters and observation.
        """
        lp = self.log_prior(theta)
        ll = self.log_likelihood(theta, obs_flux, obs_error)
        return lp + ll
