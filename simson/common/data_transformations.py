import numpy as np
import flodym as fd
from typing import Tuple, Optional

from .data_extrapolations import (
    Extrapolation,
    ProportionalExtrapolation,
)


class StockExtrapolation:

    def __init__(
        self,
        historic_stocks: fd.StockArray,
        dims: fd.DimensionSet,
        parameters: dict[str, fd.Parameter],
        stock_extrapolation_class: Extrapolation,
        target_dim_letters: Optional[Tuple[str, ...]] = None,
        fit_dim_letters: Optional[Tuple[str, ...]] = None,
        saturation_level: Optional[np.ndarray] = None,
        do_gdppc_accumulation: bool = True,
        stock_correction: str = "gaussian_first_order",
    ):
        """
        Initialize the StockExtrapolation class.

        Args:
            historic_stocks (fd.StockArray): Historical stock data.
            dims (fd.DimensionSet): Dimension set for the data.
            parameters (dict[str, fd.Parameter]): Parameters for the extrapolation.
            stock_extrapolation_class (Extrapolation): Class used for stock extrapolation.
            target_dim_letters (Optional[Tuple[str, ...]], optional): Sets the dimensions of the stock extrapolation output. Defaults to None.
            fit_dim_letters (Optional[Tuple[str, ...]], optional): Sets the dimensions across which an individual fit is performed, must be subset of target_dim_letters. Defaults to None.
            saturation_level (Optional[np.ndarray], optional): Saturation level for the extrapolation. Defaults to None.
            do_gdppc_accumulation (bool, optional): Flag to perform GDP per capita accumulation. Defaults to True.
            stock_correction (str, optional): Method for stock correction. Possible values are "gaussian_first_order", "shift_zeroth_order", "none". Defaults to "gaussian_first_order".
        """

        self.historic_stocks = historic_stocks
        self.dims = dims
        self.parameters = parameters
        self.stock_extrapolation_class = stock_extrapolation_class
        self.target_dim_letters = target_dim_letters
        self.set_dims(fit_dim_letters)
        self.saturation_level = saturation_level
        self.do_gdppc_accumulation = do_gdppc_accumulation
        self.stock_correction = stock_correction
        self.extrapolate()

    def set_dims(self, fit_dim_letters: Tuple[str, ...]):
        """
        Check target_dim_letters.
        Set fit_dim_letters and check:
        fit_dim_letters should be the same as target_dim_letters, but without the time dimension, except if otherwise defined.
        In this case, fit_dim_letters should be a subset of target_dim_letters.
        This check cannot be performed if self.target_dim_letters or self.fit_dim_letters is None.
        """
        if self.target_dim_letters is None:
            self.historic_dim_letters = self.historic_stocks.dims.letters
            self.target_dim_letters = ("t",) + self.historic_dim_letters[1:]
        else:
            self.historic_dim_letters = ("h",) + self.target_dim_letters[1:]

        if fit_dim_letters is None:
            # fit_dim_letters should be the same as target_dim_letters, but without the time dimension
            self.fit_dim_letters = tuple(x for x in self.target_dim_letters if x != "t")
        else:
            self.fit_dim_letters = fit_dim_letters
            if not set(self.fit_dim_letters).issubset(self.target_dim_letters):
                raise ValueError("fit_dim_letters must be subset of target_dim_letters.")
        self.get_fit_idx()

    def get_fit_idx(self):
        """Get the indices of the fit dimensions in the historic_stocks dimensions."""
        if self.fit_dim_letters is None:
            self.fit_dim_idx = ()
        else:
            self.fit_dim_idx = tuple(
                i
                for i, x in enumerate(self.historic_stocks.dims.letters)
                if x in self.fit_dim_letters
            )

    def extrapolate(self):
        self.per_capita_transformation()
        self.gdp_regression()

    def per_capita_transformation(self):
        self.pop = self.parameters["population"]
        self.gdppc = self.parameters["gdppc"]
        if self.do_gdppc_accumulation:
            self.gdppc_acc = np.maximum.accumulate(self.gdppc.values, axis=0)
        self.historic_pop = fd.FlodymArray(dims=self.dims[("h", "r")])
        self.historic_gdppc = fd.FlodymArray(dims=self.dims[("h", "r")])
        self.historic_stocks_pc = fd.FlodymArray(dims=self.dims[self.historic_dim_letters])
        self.stocks_pc = fd.FlodymArray(dims=self.dims[self.target_dim_letters])
        self.stocks = fd.FlodymArray(dims=self.dims[self.target_dim_letters])

        self.historic_pop[...] = self.pop[{"t": self.dims["h"]}]
        self.historic_gdppc[...] = self.gdppc[{"t": self.dims["h"]}]
        self.historic_stocks_pc[...] = self.historic_stocks / self.historic_pop

    def gaussian_correction(
        self, historic: np.ndarray, prediction: np.ndarray, approaching_time: float = 50
    ):
        """Gaussian smoothing of extrapolation around interface historic/future to remove discontinuities."""
        """Multiplies Gaussian with a Taylor expansion around the difference beteween historic and fit."""
        time = np.array(self.dims["t"].items)
        last_history_idx = len(historic) - 1
        last_history_year = time[last_history_idx]
        difference_0th = historic[last_history_idx, :] - prediction[last_history_idx, :]

        # standard approach: only take last 2 points
        # last_historic_1st = historic[last_history_idx, :] - historic[last_history_idx - 1, :]
        # last_prediction_1st = prediction[last_history_idx, :] - prediction[last_history_idx - 1, :]

        # do a proper linear regression to last n1 points
        def lin_fit(x, y, last_idx, n=5):
            x_cut = np.vstack([x[last_idx - n : last_idx], np.ones(n)]).T
            y_cut = y[last_idx - n : last_idx, :]
            y_reshaped = y_cut.reshape(n, -1).T
            slopes = [np.linalg.lstsq(x_cut, y_dim, rcond=None)[0][0] for y_dim in y_reshaped]
            slopes_reshaped = np.array(slopes).reshape(y.shape[1:])
            return slopes_reshaped

        last_historic_1st = lin_fit(time, historic, last_history_idx)
        last_prediction_1st = lin_fit(time, prediction, last_history_idx)

        difference_1st = (last_historic_1st - last_prediction_1st) / (
            last_history_year - time[last_history_idx - 1]
        )

        def gaussian(t, approaching_time):
            """After the approaching time, the amplitude of the gaussian has decreased to 5%."""
            a = np.sqrt(np.log(20))
            return np.exp(-((a * t / approaching_time) ** 2))

        time_extended = time.reshape(-1, *([1] * len(difference_0th.shape)))
        taylor = difference_0th + difference_1st * (time_extended - last_history_year)
        correction = taylor * gaussian(time_extended - last_history_year, approaching_time)

        return prediction[...] + correction

    def gdp_regression(self):
        """Updates per capita stock to future by extrapolation."""

        def match_dimensions(a, b):
            """Broadcasts b to the shape of a."""
            new_shape = b.shape + (1,) * (len(a.shape) - len(b.shape))
            b_reshaped = np.reshape(b, new_shape)
            b_broadcasted = np.broadcast_to(b_reshaped, a.shape)
            return b_broadcasted

        prediction_out = self.stocks_pc.values
        pure_prediction = np.zeros_like(prediction_out)
        historic_in = self.historic_stocks_pc.values
        gdppc = self.gdppc_acc if self.do_gdppc_accumulation else self.gdppc
        gdppc = match_dimensions(prediction_out, gdppc)
        n_historic = historic_in.shape[0]

        extrapolation = self.stock_extrapolation_class(
            data_to_extrapolate=historic_in,
            target_range=gdppc,
            independent_dims=self.fit_dim_idx,
            saturation_level=self.saturation_level,
        )
        pure_prediction = extrapolation.regress()

        if self.stock_correction == "gaussian_first_order":
            prediction_out[...] = self.gaussian_correction(historic_in, pure_prediction)
        elif self.stock_correction == "shift_zeroth_order":
            # match last point by adding the difference between the last historic point and the corresponding prediction
            prediction_out[...] = pure_prediction - (
                pure_prediction[n_historic - 1, :] - historic_in[n_historic - 1, :]
            )

        prediction_out[:n_historic, ...] = historic_in

        # transform back to total stocks
        self.stocks[...] = self.stocks_pc * self.pop


def extrapolate_to_future(
    historic_values: fd.FlodymArray, scale_by: fd.FlodymArray
) -> fd.FlodymArray:
    if not historic_values.dims.letters[0] == "h":
        raise ValueError("First dimension of historic_parameter must be historic time.")
    if not scale_by.dims.letters[0] == "t":
        raise ValueError("First dimension of scaler must be time.")
    if not set(scale_by.dims.letters[1:]).issubset(historic_values.dims.letters[1:]):
        raise ValueError("Scaler dimensions must be subset of historic_parameter dimensions.")

    all_dims = historic_values.dims.union_with(scale_by.dims)

    dim_letters_out = ("t",) + historic_values.dims.letters[1:]
    extrapolated_values = fd.FlodymArray.from_dims_superset(
        dims_superset=all_dims, dim_letters=dim_letters_out
    )

    scale_by = scale_by.cast_to(extrapolated_values.dims)

    # calculate weights
    n_hist_points = historic_values.dims.shape()[0]
    n_last_points = 5
    weights_1d = np.maximum(0.0, np.arange(-n_hist_points, 0) + n_last_points + 1)
    weights_1d = weights_1d / weights_1d.sum()
    weights = np.zeros_like(historic_values.values)
    weights[...] = weights_1d[(slice(None),) + (np.newaxis,) * (weights.ndim - 1)]

    extrapolation = ProportionalExtrapolation(
        data_to_extrapolate=historic_values.values,
        target_range=scale_by.values,
        weights=weights,
        independent_dims=(),
    )
    extrapolated_values.set_values(extrapolation.extrapolate())

    return extrapolated_values
