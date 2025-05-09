import numpy as np
import flodym as fd

from simson.common.data_extrapolations import ProportionalExtrapolation
from simson.common.data_transformations import broadcast_trailing_dimensions
from simson.common.trade import Trade


def predict_by_extrapolation(
    trade: Trade,
    scaler: fd.FlodymArray,
    scale_first: str,
    adopt_scaler_dims: bool = False,
    balance_to: str = None,
):
    """
    Predict future trade values by extrapolating the trade data using a given scaler.
    :param trade: Trade object with historic trade data.
    :param scaler: NamedDimArray object with the scaler values.
    :param scale_first: str, either 'imports' or 'exports', indicating which trade values to scale first and
                        use as a scaler for the other trade values (scale_second).
    :param adopt_scaler_dims: bool, whether to adopt the dimensions of the scaler or use the ones of the trade data.
    :param balance_to: str, which method to use for balancing the future trade data. If None, no balancing is done.
    """

    # prepare prediction
    assert scale_first in ["imports", "exports"], "Scale by must be either 'imports' or 'exports'."
    assert (
        "h" in trade.imports.dims.letters and "h" in trade.exports.dims.letters
    ), "Trade data must have a historic time dimension."

    scale_second = "exports" if scale_first == "imports" else "imports"

    # predict via extrapolation

    ## The scaler needs to be summed across dimensions that the historic trade doesn't have for the extrapolation
    historic_dims_with_t_dimension = trade.imports.dims.replace("h", scaler.dims["t"])
    total_scaler = scaler.sum_to(historic_dims_with_t_dimension.intersect_with(scaler.dims).letters)

    ## The extrapolate_to_future function uses the WeightedProportionalExtrapolation, basically a linear regression
    ## so that the share of the historic trade in the scaler is kept constant
    future_scale_first = extrapolate_to_future(
        historic_values=getattr(trade, scale_first), scale_by=total_scaler
    )

    global_scale_first = future_scale_first.sum_over(sum_over_dims=("r",))

    future_scale_second = extrapolate_to_future(
        historic_values=getattr(trade, scale_second), scale_by=global_scale_first
    )
    if adopt_scaler_dims:
        ## If the scaler has more dimensions than the historic trade, the historic trade data is split into the missing
        ## dimensions of the scaler, adapting the same sector split as the scaler.
        missing_dims = scaler.dims.difference_with(future_scale_first.dims)
        with np.errstate(divide="ignore"):
            future_scale_first = future_scale_first * scaler.get_shares_over(missing_dims.letters)
            future_scale_first.set_values(np.nan_to_num(future_scale_first.values))
            global_scale_first = future_scale_first.sum_over(sum_over_dims="r")
            future_scale_second = future_scale_second * global_scale_first.get_shares_over(
                missing_dims.letters
            )
            future_scale_second.set_values(np.nan_to_num(future_scale_second.values))

    # create future trade object
    future_dims = scaler.dims if adopt_scaler_dims else historic_dims_with_t_dimension
    future_trade = Trade(
        imports=fd.Parameter(name=trade.imports.name, dims=future_dims),
        exports=fd.Parameter(name=trade.exports.name, dims=future_dims),
    )

    getattr(future_trade, scale_first)[...] = future_scale_first
    getattr(future_trade, scale_second)[...] = future_scale_second

    # balance
    if balance_to is not None:
        future_trade.balance(to=balance_to, inplace=True)

    return future_trade


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
    dims_out = all_dims[dim_letters_out]

    extrapolated_values = fd.FlodymArray(dims=dims_out)

    scale_by = scale_by.cast_to(extrapolated_values.dims)

    # calculate weights
    n_hist_points = historic_values.dims.shape[0]
    n_last_points = 5
    weights_1d = np.maximum(0.0, np.arange(-n_hist_points, 0) + n_last_points + 1)
    weights_1d = weights_1d / weights_1d.sum()
    weights = broadcast_trailing_dimensions(weights_1d, historic_values.values)

    extrapolation = ProportionalExtrapolation(
        data_to_extrapolate=historic_values.values,
        target_range=scale_by.values,
        weights=weights,
        independent_dims=tuple(range(1, dims_out.ndim)),
    )
    extrapolated_values.set_values(extrapolation.extrapolate())

    return extrapolated_values
