from abc import abstractmethod
from typing import ClassVar
import numpy as np
import sys
from simson.common.base_model import SimsonBaseModel
from pydantic import model_validator
from scipy.optimize import least_squares


class Extrapolation(SimsonBaseModel):

    data_to_extrapolate: np.ndarray  # historical data, 1 dimensional (time)
    target_range: np.ndarray  # predictor variable(s)
    fit_prms: np.ndarray = None

    @property
    def n_historic(self):
        return self.data_to_extrapolate.shape[0]

    def extrapolate(self, historic_from_regression: bool = False):
        regression = self.regress()
        if not historic_from_regression:
            regression[: self.n_historic] = self.data_to_extrapolate
        return regression


class OneDimensionalExtrapolation(Extrapolation):

    n_prms: ClassVar[int] = 2  # should be overwritten by subclasses if not 2

    @model_validator(mode="after")
    def validate_data(self):
        assert self.data_to_extrapolate.ndim == 1, "Data to extrapolate must be 1-dimensional."
        assert self.target_range.ndim == 1, "Target range must be 1-dimensional."
        assert (
            self.data_to_extrapolate.shape[0] < self.target_range.shape[0]
        ), "data_to_extrapolate must be smaller then target_range"
        return self

    @abstractmethod
    def func(x, prms):
        pass

    @abstractmethod
    def initial_guess(self):
        pass

    def fitting_function(self, prms):
        f = self.func(self.target_range[: self.n_historic], prms)
        return f - self.data_to_extrapolate

    def regress(self):
        self.fit_prms = least_squares(
            self.fitting_function, x0=self.initial_guess(), gtol=1.0e-12
        ).x
        regression = self.func(self.target_range, self.fit_prms)
        return regression


class WeightedProportionalExtrapolation(Extrapolation):
    """
    Regression of a function of the form y = a * x, i.e. a linear scaling without offset.
    For regression, the last n_last_points_to_match points are used. Their weights are linearly decreasing to zero.
    """

    n_last_points_to_match: int = 5

    @model_validator(mode="after")
    def validate_input(self):
        assert self.n_last_points_to_match > 0, "n_last_points_to_match must be greater than 0."
        assert (
            self.data_to_extrapolate.shape[0] >= self.n_last_points_to_match
        ), f"data_to_extrapolate must have at least n_last_points_to_match data points ({self.n_last_points_to_match})."
        return self

    def regress(self):
        """ "
        Formula a = sum_i (w_i x_i y_i) / sum_i (w_i x_i^2) is the result of the weighted least squares regression
        a = argmin sum_i (w_i (a * x_i - y_i)^2).
        """
        regression_x = self.target_range[
            self.n_historic - self.n_last_points_to_match : self.n_historic
        ]
        regression_y = self.data_to_extrapolate[-self.n_last_points_to_match :]

        # move last points axis to back for multiplication
        regression_x = np.moveaxis(regression_x, 0, -1)
        regression_y = np.moveaxis(regression_y, 0, -1)

        # calculate weights
        regression_weights = np.arange(1, self.n_last_points_to_match + 1)
        regression_weights = regression_weights / regression_weights.sum()

        # calculate slope
        slope_dividend = np.sum(regression_x * regression_y * regression_weights, axis=-1)
        slope_divisor = np.sum(regression_x**2 * regression_weights, axis=-1)
        slope_divisor[slope_divisor == 0] = (
            sys.float_info.epsilon
        )  # avoid division by zero, slope will be zero anyways
        slope = slope_dividend / slope_divisor

        regression = self.target_range * slope
        return regression


class PehlExtrapolation(OneDimensionalExtrapolation):

    @staticmethod
    def func(x, prms):
        return prms[0] / (1.0 + np.exp(prms[1] / x))

    def initial_guess(self):
        return np.array(
            [2.0 * self.target_range[self.n_historic - 1], self.data_to_extrapolate[-1]]
        )


class ExponentialSaturationExtrapolation(OneDimensionalExtrapolation):

    @staticmethod
    def func(x, prms):
        return prms[0] * (1 - np.exp(-prms[1] * x))

    def initial_guess(self):
        current_level = self.data_to_extrapolate[-1]
        current_extrapolator = self.target_range[self.n_historic - 1]
        initial_saturation_level = (
            2.0 * current_level if np.max(np.abs(current_level)) > sys.float_info.epsilon else 1.0
        )
        initial_stretch_factor = (
            -np.log(1 - current_level / initial_saturation_level) / current_extrapolator
        )

        return np.array([initial_saturation_level, initial_stretch_factor])


class SigmoidExtrapolation(OneDimensionalExtrapolation):

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
