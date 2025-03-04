import numpy as np
import flodym as fd

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
        target_dim_letters=None,
        saturation_level=None,
    ):
        self.historic_stocks = historic_stocks
        self.dims = dims
        self.parameters = parameters
        self.stock_extrapolation_class = stock_extrapolation_class
        self.target_dim_letters = target_dim_letters
        self.saturation_level = saturation_level
        self.extrapolate()

    def extrapolate(self):
        self.per_capita_transformation()
        self.gdp_regression()

    def per_capita_transformation(self):
        if self.target_dim_letters is None:
            self.historic_dim_letters = self.historic_stocks.dims.letters
            self.target_dim_letters = ("t",) + self.historic_dim_letters[1:]
        else:
            self.historic_dim_letters = ("h",) + self.target_dim_letters[1:]

        # transform to per capita
        self.pop = self.parameters["population"]
        self.gdppc = self.parameters["gdppc"]
        self.historic_pop = fd.FlodymArray(dims=self.dims[("h", "r")])
        self.historic_gdppc = fd.FlodymArray(dims=self.dims[("h", "r")])
        self.historic_stocks_pc = fd.FlodymArray(dims=self.dims[self.historic_dim_letters])
        self.stocks_pc = fd.FlodymArray(dims=self.dims[self.target_dim_letters])
        self.stocks = fd.FlodymArray(dims=self.dims[self.target_dim_letters])

        self.historic_pop[...] = self.pop[{"t": self.dims["h"]}]
        self.historic_gdppc[...] = self.gdppc[{"t": self.dims["h"]}]
        self.historic_stocks_pc[...] = self.historic_stocks / self.historic_pop

    def gdp_regression(self):
        """Updates per capita stock to future by extrapolation."""
        prediction_out = self.stocks_pc.values
        historic_in = self.historic_stocks_pc.values
        shape_out = prediction_out.shape
        pure_prediction = np.zeros_like(prediction_out)
        n_historic = historic_in.shape[0]

        for idx in np.ndindex(shape_out[1:]):
            # idx is a tuple of indices for all dimensions except the time dimension
            index = (slice(None),) + idx
            current_hist_stock_pc = historic_in[index]
            current_gdppc = self.gdppc.values[index[:2]]
            kwargs = {}
            if self.saturation_level is not None:
                kwargs["saturation_level"] = self.saturation_level[idx]
            extrapolation = self.stock_extrapolation_class(
                data_to_extrapolate=current_hist_stock_pc, target_range=current_gdppc, **kwargs
            )
            pure_prediction[index] = extrapolation.regress()

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
        independent=True,
    )
    extrapolated_values.set_values(extrapolation.extrapolate())

    return extrapolated_values
