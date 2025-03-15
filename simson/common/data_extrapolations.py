from abc import abstractmethod
from typing import ClassVar, Optional, Tuple
import numpy as np
import sys
from simson.common.base_model import SimsonBaseModel
from pydantic import model_validator
from scipy.optimize import least_squares


class Extrapolation(SimsonBaseModel):

    data_to_extrapolate: np.ndarray
    """historical data"""
    target_range: np.ndarray
    """predictor variable(s) covering range of data_to_extrapolate and beyond"""
    weights: Optional[np.ndarray] = None
    saturation_level: Optional[np.ndarray] = None
    independent_dims: Optional[Tuple[int, ...]] = ()
    """Indizes for dimensions across which to regress independently. Other dimensions are regressed commonly.
    If None, all dimensions are regressed individually. If empty (), all dimensions are regressed aggregately."""
    fit_prms: np.ndarray = None
    n_prms: ClassVar[int]

    @model_validator(mode="after")
    def validate_data(self):
        assert (
            self.data_to_extrapolate.shape[0] < self.target_range.shape[0]
        ), "data_to_extrapolate must be smaller then target_range"
        assert (
            self.data_to_extrapolate.shape[1:] == self.target_range.shape[1:]
        ), "Data to extrapolate and target range must have the same shape except for the first dimension."
        if self.weights is None:
            self.weights = np.ones_like(self.data_to_extrapolate)
        else:
            assert (
                self.weights.shape == self.data_to_extrapolate.shape
            ), "Weights must have the same shape as data_to_extrapolate."
        return self

    @property
    def n_historic(self):
        return self.data_to_extrapolate.shape[0]

    def extrapolate(self, historic_from_regression: bool = False):
        regression = self.regress()
        if not historic_from_regression:
            regression[: self.n_historic, ...] = self.data_to_extrapolate
        return regression

    @abstractmethod
    def func(x: np.ndarray, prms: np.ndarray, **kwargs) -> np.ndarray:
        pass

    @abstractmethod
    def initial_guess(
        self, target_range: np.ndarray, data_to_extrapolate: np.ndarray
    ) -> np.ndarray:
        """gets either one-dimensional or multi-dimensional data, but always returns one scalar value per prm"""
        pass

    def get_fitting_function(
        self,
        target_range: np.ndarray,
        data_to_extrapolate: np.ndarray,
        weights: np.ndarray,
        saturation_level: np.ndarray,
    ) -> callable:
        kwargs = {"saturation_level": saturation_level} if saturation_level is not None else {}

        def fitting_function(prms: np.ndarray) -> np.ndarray:
            f = self.func(target_range, prms, **kwargs)
            loss = weights * (f - data_to_extrapolate)
            return loss.flatten()

        return fitting_function

    @staticmethod
    def remove_shape_dimensions(shape, retain_idx):
        """Removes dimensions from shape, except indices in retain_idx."""
        result_shape = [dim_size for i, dim_size in enumerate(shape) if i in retain_idx]
        return tuple(result_shape)

    def regress(self):
        # extract dimensions that are regressed independently
        if self.independent_dims is not None:
            target_shape = self.remove_shape_dimensions(
                self.target_range.shape, self.independent_dims
            )
        else:
            target_shape = self.target_range.shape[1:]
        regression = np.zeros_like(self.target_range)
        self.fit_prms = np.zeros(self.target_range.shape[1:] + (self.n_prms,))

        # loop over dimensions that are regressed independently
        for idx in np.ndindex(target_shape):
            index = (slice(None),) + idx
            self.fit_prms[idx], regression[index] = self.regress_common(
                self.target_range[index],
                self.data_to_extrapolate[index],
                self.weights[index],
                self.saturation_level[idx] if self.saturation_level is not None else None,
            )

        return regression

    def regress_common(self, target, data, weights, saturation_level):
        """Finds optimal fit of data and extrapolates to target."""
        fitting_function = self.get_fitting_function(
            target[: self.n_historic, ...],
            data,
            weights,
            saturation_level,
        )
        initial_guess = self.initial_guess(target, data)
        fit_prms = least_squares(fitting_function, x0=initial_guess, gtol=1.0e-12).x
        kwargs = {"saturation_level": saturation_level} if saturation_level is not None else {}
        regression = self.func(target, fit_prms, **kwargs)
        return fit_prms, regression


class ProportionalExtrapolation(Extrapolation):

    n_prms: ClassVar[int] = 1

    @staticmethod
    def func(x, prms):
        return prms[0] * x

    @staticmethod
    def initial_guess(target_range, data_to_extrapolate):
        return np.array([1.0])


class PehlExtrapolation(Extrapolation):

    n_prms: ClassVar[int] = 2

    @staticmethod
    def func(x, prms):
        return prms[0] / (1.0 + np.exp(prms[1] / x))

    def initial_guess(self, target_range, data_to_extrapolate):
        return np.array(
            [
                2.0 * np.max(target_range[self.n_historic - 1, ...]),
                np.max(data_to_extrapolate[-1, ...]),
            ]
        )


class ExponentialSaturationExtrapolation(Extrapolation):

    n_prms: ClassVar[int] = 2

    @staticmethod
    def func(x, prms):
        return prms[0] * (1 - np.exp(-prms[1] * x))

    def initial_guess(self, target_range, data_to_extrapolate):
        current_level = np.max(data_to_extrapolate[-1, ...])
        current_extrapolator = np.max(target_range[self.n_historic - 1, ...])
        initial_saturation_level = 2.0 * current_level
        initial_stretch_factor = (
            -np.log(1 - current_level / initial_saturation_level) / current_extrapolator
        )
        return np.array([initial_saturation_level, initial_stretch_factor])


class VarySatLogSigmoidExtrapolation(Extrapolation):

    n_prms: ClassVar[int] = 3

    @staticmethod
    def func(x, prms):
        return prms[0] / (1 + np.exp(-prms[1] * (np.log(x) - prms[2])))

    @staticmethod
    def initial_guess(target_range, data_to_extrapolate):
        max_level = np.max(np.log(data_to_extrapolate))
        sat_level_guess = 2 * max_level

        mean_target = np.mean(np.log(target_range))

        target_max_level = np.max(np.log(target_range))
        stretch_factor = 2 / (target_max_level - mean_target)
        return np.array([sat_level_guess, stretch_factor, mean_target])


class FixedSatLogSigmoidExtrapolation(Extrapolation):

    n_prms: ClassVar[int] = 2
    saturation_level: np.ndarray = np.array([1.0])

    @staticmethod
    def func(x, prms, saturation_level):
        return saturation_level / (1 + np.exp(-prms[0] * (np.log(x) - prms[1])))

    @staticmethod
    def initial_guess(target_range, data_to_extrapolate):
        mean_target = np.mean(np.log(target_range))
        target_max_level = np.max(np.log(target_range))
        stretch_factor = 2 / (target_max_level - mean_target)
        return np.array([stretch_factor, mean_target])


class SigmoidExtrapolation(Extrapolation):

    n_prms: ClassVar[int] = 3

    @staticmethod
    def func(x, prms):
        return prms[0] / (1.0 + np.exp(-prms[1] * (x - prms[2])))

    def initial_guess(self):
        current_level = self.data_to_extrapolate[-1]
        current_extrapolator = self.target_range[self.n_historic - 1]
        initial_saturation_level = (
            2.0 * current_level if np.max(np.abs(current_level)) > sys.float_info.epsilon else 1.0
        )

        # Estimate slope based on historical data points
        if len(self.data_to_extrapolate) > 1:
            # Calculate average rate of change in recent history
            recent_y_change = self.data_to_extrapolate[-1] - self.data_to_extrapolate[-2]
            recent_x_change = (
                self.target_range[self.n_historic - 1] - self.target_range[self.n_historic - 2]
            )
            if abs(recent_x_change) > sys.float_info.epsilon:
                slope_estimate = recent_y_change / recent_x_change
                # Convert slope to stretch factor (sigmoid derivative at midpoint is prms[0]*prms[1]/4)
                initial_stretch_factor = 4.0 * slope_estimate / initial_saturation_level
            else:
                initial_stretch_factor = 0.1
        else:
            initial_stretch_factor = 0.1

        # If current level is approximately half the saturation level, set x-offset to current x value
        ratio = current_level / initial_saturation_level
        # Solve for x-offset using the sigmoid equation at current point
        if 0 < ratio < 1:
            logit = np.log(ratio / (1.0 - ratio))
            initial_x_offset = current_extrapolator - logit / initial_stretch_factor
        else:
            # Fallback if ratio is not in (0,1)
            initial_x_offset = current_extrapolator

        return np.array([initial_saturation_level, initial_stretch_factor, initial_x_offset])
