import flodym as fd
import numpy as np
from typing import Tuple, Optional, Union, Type

from simson.common.data_extrapolations import Extrapolation
from simson.common.data_transformations import broadcast_trailing_dimensions, BoundList


class StockExtrapolation:

    def __init__(
        self,
        historic_stocks: fd.StockArray,
        dims: fd.DimensionSet,
        parameters: dict[str, fd.Parameter],
        stock_extrapolation_class: Type[Extrapolation],
        target_dim_letters: Union[Tuple[str, ...], str] = "all",
        indep_fit_dim_letters: Union[Tuple[str, ...], str] = (),
        bound_list: BoundList = BoundList(),
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
            target_dim_letters (Union[Tuple[str, ...], str], optional): Sets the dimensions of the stock extrapolation output. If "all", the output will have the same shape as historic_stocks, except for the time dimension. Defaults to "all".
            indep_fit_dim_letters (Optional[Tuple[str, ...]], optional): Sets the dimensions across which an individual fit is performed, must be subset of target_dim_letters. If "all", all dimensions given in target_dim_letters are regressed individually. If empty (), all dimensions are regressed aggregately. Defaults to ().
            bounds (list[Bound], optional): List of bounds for the extrapolation. Defaults to [].
            do_gdppc_accumulation (bool, optional): Flag to perform GDP per capita accumulation. Defaults to True.
            stock_correction (str, optional): Method for stock correction. Possible values are "gaussian_first_order", "shift_zeroth_order", "none". Defaults to "gaussian_first_order".
        """
        self.historic_stocks = historic_stocks
        self.dims = dims
        self.parameters = parameters
        self.stock_extrapolation_class = stock_extrapolation_class
        self.target_dim_letters = target_dim_letters
        self.set_dims(indep_fit_dim_letters)
        self.bound_list = bound_list
        self.do_gdppc_accumulation = do_gdppc_accumulation
        self.stock_correction = stock_correction
        self.extrapolate()

    def set_dims(self, indep_fit_dim_letters: Tuple[str, ...]):
        """
        Check target_dim_letters.
        Set fit_dim_letters and check:
        fit_dim_letters should be the same as target_dim_letters, but without the time dimension, except if otherwise defined.
        In this case, fit_dim_letters should be a subset of target_dim_letters.
        This check cannot be performed if self.target_dim_letters or self.fit_dim_letters is None.
        """
        if self.target_dim_letters == "all":
            self.historic_dim_letters = self.historic_stocks.dims.letters
            self.target_dim_letters = ("t",) + self.historic_dim_letters[1:]
        else:
            self.historic_dim_letters = ("h",) + self.target_dim_letters[1:]

        if indep_fit_dim_letters == "all":
            # fit_dim_letters should be the same as target_dim_letters, but without the time dimension
            self.indep_fit_dim_letters = tuple(x for x in self.target_dim_letters if x != "t")
        else:
            self.indep_fit_dim_letters = indep_fit_dim_letters
            if not set(self.indep_fit_dim_letters).issubset(self.target_dim_letters):
                raise ValueError("fit_dim_letters must be subset of target_dim_letters.")
        self.get_fit_idx()

    def get_fit_idx(self):
        """Get the indices of the fit dimensions in the historic_stocks dimensions."""
        self.fit_dim_idx = tuple(
            i
            for i, x in enumerate(self.historic_stocks.dims.letters)
            if x in self.indep_fit_dim_letters
        )

    def extrapolate(self):
        self.per_capita_transformation()
        self.gdp_regression()

    def per_capita_transformation(self):
        self.pop = self.parameters["population"]
        self.gdppc = self.parameters["gdppc"]
        if self.do_gdppc_accumulation:
            self.gdppc_acc = np.maximum.accumulate(self.gdppc.values, axis=0)
        self.historic_pop = fd.Parameter(dims=self.dims[("h", "r")])
        self.historic_gdppc = fd.Parameter(dims=self.dims[("h", "r")])
        self.historic_stocks_pc = fd.StockArray(dims=self.dims[self.historic_dim_letters])
        self.stocks_pc = fd.StockArray(dims=self.dims[self.target_dim_letters])
        self.stocks = fd.StockArray(dims=self.dims[self.target_dim_letters])

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

        prediction_out = self.stocks_pc.values
        pure_prediction = np.zeros_like(prediction_out)
        historic_in = self.historic_stocks_pc.values
        gdppc = self.gdppc_acc if self.do_gdppc_accumulation else self.gdppc
        gdppc = broadcast_trailing_dimensions(gdppc, prediction_out)
        n_historic = historic_in.shape[0]

        extrapolation = self.stock_extrapolation_class(
            data_to_extrapolate=historic_in,
            target_range=gdppc,
            independent_dims=self.fit_dim_idx,
            bound_list=self.bound_list,
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
