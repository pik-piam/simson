import sys
from simson.common.trade import Trade
from sodym import Parameter


def balance_by_extrenum(trade, by: str, inplace=False):
    global_imports = trade.imports.sum_nda_over('r')
    global_exports = trade.exports.sum_nda_over('r')

    reference_trade_lookup = {
        'maximum': global_imports.maximum(global_exports),
        'minimum': global_imports.minimum(global_exports),
        'imports': global_imports,
        'exports': global_exports,
    }
    if by not in reference_trade_lookup:
        raise ValueError(f"Extrenum {by} not recognized. Must be one of {list(reference_trade_lookup.keys())}")
    reference_trade = reference_trade_lookup[by]

    import_factor = reference_trade / global_imports.maximum(sys.float_info.epsilon)
    export_factor = reference_trade / global_exports.maximum(sys.float_info.epsilon)

    new_imports = trade.imports * import_factor
    new_exports = trade.exports * export_factor

    if not inplace:
        return Trade(imports=new_imports, exports=new_exports, balancer=trade.balancer)

    trade.imports = new_imports
    trade.exports = new_exports

    return trade


def balance_by_scaling(trade, inplace=False):
    net_trade = trade.imports - trade.exports
    global_net_trade = net_trade.sum_nda_over('r')
    global_absolute_net_trade = net_trade.abs().sum_nda_over('r')

    # avoid division by zero, net_trade will be zero when global absolute net trade is zero anyways
    global_absolute_net_trade = global_absolute_net_trade.maximum(sys.float_info.epsilon)

    new_net_trade = net_trade * (1 - net_trade.sign() * global_net_trade / global_absolute_net_trade)
    new_imports = new_net_trade.maximum(0)
    new_exports = new_net_trade.minimum(0).abs()

    if not inplace:
        return Trade(imports=new_imports, exports=new_exports, balancer=trade.balancer)

    trade.imports = new_imports
    trade.exports = new_exports

    return trade
