from simson.common.data_transformations import extrapolate_to_future
from simson.common.trade import Trade
from simson.common.trade_balancers import balance_by_scaling
from sodym import Parameter


def predict_by_extrapolation(trade, scaler, scale_first: str, adopt_scaler_dims: bool = False, do_balance: bool = True,
                             balancer=balance_by_scaling):
    """
    Predict future trade values by extrapolating the trade data using a given scaler.
    :param trade: Trade object with historic trade data.
    :param scaler: NamedDimArray object with the scaler values.
    :param scale_first: str, either 'Imports' or 'Exports', indicating which trade values to scale first and
                        use as a scaler for the other trade values (scale_second).
    :param adopt_scaler_dims: bool, whether to adopt the dimensions of the scaler or use the ones of the trade data.
    :param do_balance: bool, whether to balance the future trade data.
    :param balancer: callable, function to balance the future trade data, here we use balance_by_scaling as a default.
    """

    # prepare prediction

    assert scale_first in ['Imports', 'Exports'], "Scale by must be either 'Imports' or 'Exports'."
    assert 'h' in trade.imports.dims.letters and 'h' in trade.exports.dims.letters, \
        "Trade data must have a historic time dimension."

    scale_second = 'Exports' if scale_first == 'Imports' else 'Imports'

    # predict via extrapolation

    ## The scaler needs to be summed across dimensions that the historic trade doesn't have for the extrapolation
    historic_dims_with_t_dimension = trade.imports.dims.replace('h', scaler.dims['t'])
    total_scaler = scaler.sum_nda_to(historic_dims_with_t_dimension.intersect_with(scaler.dims).letters)

    ## The extrapolate_to_future function uses the WeightedProportionalExtrapolation, basically a linear regression
    ## so that the share of the historic trade in the scaler is kept constant
    future_scale_first = (
        extrapolate_to_future(historic_values=trade[scale_first],
                              scale_by=total_scaler))

    global_scale_first = future_scale_first.sum_nda_over(sum_over_dims=('r',))

    future_scale_second = extrapolate_to_future(historic_values=trade[scale_second],
                                                scale_by=global_scale_first)
    if adopt_scaler_dims:
        ## If the scaler has more dimensions than the historic trade, the historic trade data is split into the missing
        ## dimensions of the scaler, adapting the same sector split as the scaler.
        missing_dims = scaler.dims.difference_with(future_scale_first.dims)
        future_scale_first = future_scale_first * scaler.get_shares_over(missing_dims.letters)
        global_scale_first = future_scale_first.sum_nda_over(sum_over_dims='r')
        future_scale_second = future_scale_second * global_scale_first.get_shares_over(missing_dims.letters)

    # create future trade object

    future_dims = scaler.dims if adopt_scaler_dims else historic_dims_with_t_dimension
    future_trade = Trade(dims=future_dims,
                         imports=Parameter(name=trade.imports.name, dims=future_dims),
                         exports=Parameter(name=trade.exports.name, dims=future_dims),
                         balancer=balancer)

    future_trade[scale_first][...] = future_scale_first
    future_trade[scale_second][...] = future_scale_second

    # balance

    if do_balance:
        future_trade.balance()

    return future_trade
