from abc import abstractmethod
from typing import ClassVar
import numpy as np
import sys
from simson.common.base_model import SimsonBaseModel
from pydantic import model_validator
from scipy.optimize import least_squares


class Extrapolation(SimsonBaseModel):

    data_to_extrapolate: np.ndarray
    """historical data, 1 dimensional (time)"""
    target_range: np.ndarray
    """predictor variable(s)"""
    weights: np.ndarray = None
    independent: bool = False
    """Whether to regress each entry (apart along time dim) independently or do a common regression for all entries"""
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
    def func(x: np.ndarray, prms: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def initial_guess(
        self, target_range: np.ndarray, data_to_extrapolate: np.ndarray
    ) -> np.ndarray:
        """gets either one-dimensional or multi-dimensional data, but always returns one scalar value per prm"""
        pass

    def get_fitting_function(
        self, target_range: np.ndarray, data_to_extrapolate: np.ndarray, weights: np.ndarray
    ) -> callable:
        def fitting_function(prms: np.ndarray) -> np.ndarray:
            f = self.func(target_range, prms)
            loss = weights * (f - data_to_extrapolate)
            return loss.flatten()

        return fitting_function

    def regress(self):
        if self.independent:
            return self.regress_independently()
        else:
            return self.regress_common()

    def regress_common(self):
        fitting_function = self.get_fitting_function(
            self.target_range[: self.n_historic, ...], self.data_to_extrapolate, self.weights
        )
        initial_guess = self.initial_guess(self.target_range, self.data_to_extrapolate)

        self.fit_prms = least_squares(fitting_function, x0=initial_guess, gtol=1.0e-12).x
        regression = self.func(self.target_range, self.fit_prms)
        return regression

    def regress_independently(self):
        regression = np.zeros_like(self.target_range)
        self.fit_prms = np.zeros(self.target_range.shape[1:] + (self.n_prms,))
        for idx in np.ndindex(self.target_range.shape[1:]):
            index = (slice(None),) + idx
            fitting_function = self.get_fitting_function(
                self.target_range[: self.n_historic, ...][index],
                self.data_to_extrapolate[index],
                self.weights[index],
            )
            initial_guess = self.initial_guess(
                self.target_range[index], self.data_to_extrapolate[index]
            )
            self.fit_prms[idx] = least_squares(fitting_function, x0=initial_guess, gtol=1.0e-12).x
            regression[index] = self.func(self.target_range[index], self.fit_prms[idx])
        return regression


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
    saturation_level: float = 1.0

    def func(self, x, prms):
        return self.saturation_level / (1 + np.exp(-prms[0] * (np.log(x) - prms[1])))

    @staticmethod
    def initial_guess(target_range, data_to_extrapolate):
        mean_target = np.mean(np.log(target_range))
        target_max_level = np.max(np.log(target_range))
        stretch_factor = 2 / (target_max_level - mean_target)
        return np.array([stretch_factor, mean_target])
