from abc import abstractmethod
import numpy as np
from pydantic import BaseModel, ConfigDict
from typing import Union
from scipy.optimize import least_squares


class Extrapolation(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    data_to_extrapolate: np.ndarray  # historical data, 1 dimensional (time)
    extrapolate_from: Union[np.ndarray, list[np.ndarray]]  # predictor variable(s)

    @property
    def n_historic(self):
        return self.data_to_extrapolate.shape[0]

    @abstractmethod
    def predict(self):
        pass


class LinearExtrapolation(Extrapolation):



    def predict(self):
        divisor = np.maximum(self.extrapolate_from, 1e-10)[:self.n_historic]
        share = self.data_to_extrapolate / divisor
        scale_factor = (0.3 * share[-1] +
                        0.25 * share[-2] +
                        0.2 * share[-3] +
                        0.15 * share[-4] +
                        0.1 * share[-5])
        prediction = self.extrapolate_from * scale_factor

        return prediction


class SigmoidalExtrapolation(Extrapolation):

    def initial_guess(self):
        return np.array([2.*self.extrapolate_from[self.n_historic-1], self.data_to_extrapolate[-1]])

    def fitting_function(self, prms):
        return (
            prms[0] / (1. + np.exp(prms[1]/self.extrapolate_from[:self.n_historic]))
        ) - self.data_to_extrapolate

    def predict(self):
        prms_out = least_squares(self.fitting_function, x0=self.initial_guess(), gtol=1.e-12)
        prediction = prms_out.x[0] / (1. + np.exp(prms_out.x[1] / self.extrapolate_from))
        return prediction


class ExponentialExtrapolation(Extrapolation):

    def initial_guess(self):
        current_level = self.data_to_extrapolate[-1]
        current_extrapolator = self.extrapolate_from[self.n_historic - 1]
        initial_saturation_level = 2. * current_level if current_level != 0.0 else 1.0
        initial_stretch_factor = - np.log(1 -  current_level / initial_saturation_level) / current_extrapolator

        return np.array([initial_saturation_level, initial_stretch_factor])

    def fitting_function(self, prms):
        return (
            prms[0] * (1 - np.exp(-prms[1]*self.extrapolate_from[:self.n_historic]))
        ) - self.data_to_extrapolate

    def predict(self):
        prms_out = least_squares(self.fitting_function, x0=self.initial_guess(), gtol=1.e-12)
        prediction = (prms_out.x[0] * (1 - np.exp(-prms_out.x[1] * self.extrapolate_from)))

        return prediction
