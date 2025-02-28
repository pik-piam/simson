from abc import abstractmethod
import numpy as np
import sys
from simson.common.base_model import SimsonBaseModel
from pydantic import model_validator
from scipy.optimize import least_squares


class Extrapolation(SimsonBaseModel):

    data_to_extrapolate: np.ndarray  # historical data, 1 dimensional (time)
    target_range: np.ndarray  # predictor variable(s)

    @property
    def n_historic(self):
        return self.data_to_extrapolate.shape[0]

    def extrapolate(self, historic_from_regression: bool = False):
        regression = self.regress()
        if not historic_from_regression:
            regression[: self.n_historic] = self.data_to_extrapolate
        return regression

    @abstractmethod
    def regress(self):
        pass


class OneDimensionalExtrapolation(Extrapolation):

    @model_validator(mode="after")
    def validate_data(self):
        assert self.data_to_extrapolate.ndim == 1, "Data to extrapolate must be 1-dimensional."
        assert self.target_range.ndim == 1, "Target range must be 1-dimensional."
        assert (
            self.data_to_extrapolate.shape[0] < self.target_range.shape[0]
        ), "data_to_extrapolate must be smaller then target_range"
        return self


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


class SigmoidalExtrapolation(OneDimensionalExtrapolation):

    def initial_guess(self):
        return np.array(
            [2.0 * self.target_range[self.n_historic - 1], self.data_to_extrapolate[-1]]
        )

    def fitting_function(self, prms):
        return (
            prms[0] / (1.0 + np.exp(prms[1] / self.target_range[: self.n_historic]))
        ) - self.data_to_extrapolate

    def regress(self):
        prms_out = least_squares(self.fitting_function, x0=self.initial_guess(), gtol=1.0e-12)
        regression = prms_out.x[0] / (1.0 + np.exp(prms_out.x[1] / self.target_range))
        return regression


class ExponentialExtrapolation(OneDimensionalExtrapolation):

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

    def fitting_function(self, prms):
        return (
            prms[0] * (1 - np.exp(-prms[1] * self.target_range[: self.n_historic]))
        ) - self.data_to_extrapolate

    def regress(self):
        prms_out = least_squares(self.fitting_function, x0=self.initial_guess(), gtol=1.0e-12)
        regression = prms_out.x[0] * (1 - np.exp(-prms_out.x[1] * self.target_range))

        return regression


class MultiDimLogSigmoidalExtrapolation(Extrapolation):
    saturation_level: float = None
    guess_sat_level_over_max_by_pct: float = 0.2  # TODO store default value in new config

    def __init__(self, **data):
        super().__init__(**data)
        self.target_range = np.log(self.target_range)

    def initial_guess(self):
        max_level = np.max(self.data_to_extrapolate)
        sat_level_guess = (1.0 + self.guess_sat_level_over_max_by_pct) * max_level

        mean_target = np.mean(
            self.target_range
        )  # TODO: decide maybe only use mean of historic values

        index_max_level = np.where(self.data_to_extrapolate == max_level)  # todo check
        target_max_level = self.target_range[index_max_level][0]
        stretch_factor = -np.log(self.guess_sat_level_over_max_by_pct) / (
            target_max_level - mean_target
        )

        return np.array([sat_level_guess, stretch_factor, mean_target])

    def fitting_function_with_saturation_level(self, prms):
        return (
            self.saturation_level
            / (
                1.0
                + np.exp(-(prms[0] * (self.target_range[: self.n_historic].flatten() - prms[1])))
            )
        ) - self.data_to_extrapolate.flatten()

    def fitting_function_without_saturation_level(self, prms):
        return (
            prms[0]
            / (
                1.0
                + np.exp(-(prms[1] * (self.target_range[: self.n_historic].flatten() - prms[2])))
            )
        ) - self.data_to_extrapolate.flatten()

    def get_params(self):
        initial_guess = self.initial_guess()

        if self.saturation_level is None:
            fitting_function = self.fitting_function_without_saturation_level
        else:
            fitting_function = self.fitting_function_with_saturation_level
            initial_guess = initial_guess[1:]

        return least_squares(fitting_function, x0=initial_guess, gtol=1.0e-12).x

    def regress(self):
        prms_out = self.get_params()
        if len(prms_out) == 2:
            prms_out = np.concatenate([[self.saturation_level], prms_out])
        regression = prms_out[0] / (
            1.0 + np.exp(-(prms_out[1] * (self.target_range - prms_out[2])))
        )

        return regression


class LogSigmoidalExtrapolation(MultiDimLogSigmoidalExtrapolation, OneDimensionalExtrapolation):
    # TODO decide class structure -> is OneDimensionalExtrapolation really necessary?
    pass
