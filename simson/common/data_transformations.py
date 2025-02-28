import numpy as np
import flodym as fd

from .data_extrapolations import (
    SigmoidalExtrapolation,
    ExponentialExtrapolation,
    WeightedProportionalExtrapolation,
    MultiDimLogSigmoidalExtrapolation,
    LogSigmoidalExtrapolation,
)


def extrapolate_stock(
    historic_stocks: fd.StockArray,
    dims: fd.DimensionSet,
    parameters: dict[str, fd.Parameter],
    curve_strategy: str,
    target_dim_letters=None,
    saturation_level=None,
):
    """Performs the per-capita transformation and the extrapolation."""

    if target_dim_letters is None:
        historic_dim_letters = historic_stocks.dims.letters
        target_dim_letters = ("t",) + historic_dim_letters[1:]
    else:
        historic_dim_letters = ("h",) + target_dim_letters[1:]

    # transform to per capita
    historic_stocks_pc = fd.FlodymArray(dims=dims[historic_dim_letters])
    stocks_pc = fd.FlodymArray(dims=dims[target_dim_letters])
    stocks = fd.FlodymArray(dims=dims[target_dim_letters])

    historic_stocks_pc[...] = historic_stocks / parameters["population"][{"t": dims["h"]}]

    extrapolation_class_dict = {
        "GDP_regression": SigmoidalExtrapolation,
        "Exponential_GDP_regression": ExponentialExtrapolation,
        "LogSigmoid_GDP_regression": LogSigmoidalExtrapolation,
    }

    assert (
        curve_strategy in extrapolation_class_dict.keys()
    ), f"Extrapolation strategy {curve_strategy} is not defined."

    gdp_regression(
        historic_stocks_pc.values,
        parameters["gdppc"].values,
        stocks_pc.values,
        extrapolation_class_dict[curve_strategy],
        saturation_level=saturation_level,
    )

    # transform back to total stocks
    stocks[...] = stocks_pc * parameters["population"]

    return fd.StockArray(**dict(stocks))


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

    extrapolation = WeightedProportionalExtrapolation(
        data_to_extrapolate=historic_values.values, target_range=scale_by.values
    )
    extrapolated_values.set_values(extrapolation.extrapolate())

    return extrapolated_values


def gdp_regression(
    historic_stocks_pc,
    gdppc,
    prediction_out,
    extrapolation_class=SigmoidalExtrapolation,
    saturation_level=None,
):
    shape_out = prediction_out.shape
    pure_prediction = np.zeros_like(prediction_out)
    n_historic = historic_stocks_pc.shape[0]

    # TODO decide whether to delete this line
    # gdppc = np.maximum.accumulate(gdppc, axis=0) TODO doesn't let GDP drop ever

    for idx in np.ndindex(shape_out[1:]):
        # idx is a tuple of indices for all dimensions except the time dimension
        idx_with_time_dim = (slice(None),) + idx
        current_hist_stock_pc = historic_stocks_pc[idx_with_time_dim]
        current_gdppc = gdppc[idx_with_time_dim[:2]]
        kwargs = {}
        if saturation_level is not None:
            kwargs["saturation_level"] = saturation_level[idx]
        extrapolation = extrapolation_class(
            data_to_extrapolate=current_hist_stock_pc, target_range=current_gdppc, **kwargs
        )
        pure_prediction[idx_with_time_dim] = extrapolation.regress()

    # TODO: Discuss this - how should we deal with continuation at current point (currently changes sat level
    do_fit_current_levels = False
    if do_fit_current_levels:
        prediction_out[...] = pure_prediction - (
            pure_prediction[n_historic - 1, :] - historic_stocks_pc[n_historic - 1, :]
        )
    else:
        prediction_out[...] = pure_prediction

    prediction_out[:n_historic, ...] = historic_stocks_pc
