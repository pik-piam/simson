from copy import copy
import numpy as np

from sodym import (
    StockArray, DynamicStockModel, FlowDrivenStock,
    DimensionSet, NamedDimArray, Process, Parameter
)

from .data_extrapolations import SigmoidalExtrapolation, ExponentialExtrapolation, LinearExtrapolation


def get_new_array(dims: DimensionSet, **kwargs) -> NamedDimArray:
    dims = copy(dims).get_subset(kwargs["dim_letters"]) if "dim_letters" in kwargs else dims
    return NamedDimArray(dims=dims, **kwargs)


def get_subset_transformer(dims: DimensionSet, dim_letters: tuple):
    """Get a Parameter/NamedDimArray which transforms between two dimensions, one of which is a subset of the
    other."""
    assert len(dim_letters) == 2, "Only two dimensions are allowed"
    dims = copy(dims).get_subset(dim_letters)
    assert set(dims[0].items).issubset(set(dims[1].items)) or set(dims[1].items).issubset(
        set(dims[0].items)
    ), f"Dimensions '{dims[0].name}' and '{dims[1].name}' are not subset and superset or vice versa."
    out = NamedDimArray(name=f"transform_{dims[0].letter}_<->_{dims[1].letter}", dims=dims)
    # set all values to 1 if first axis item equals second axis item
    for i, item in enumerate(dims[0].items):
        if item in dims[1].items:
            out.values[i, dims[1].index(item)] = 1
    return out    


def extrapolate_stock(
        historic_stocks: StockArray, dims: DimensionSet,
        parameters: dict[str, Parameter], curve_strategy: str
        ):
    """Performs the per-capita transformation and the extrapolation."""

    # transform to per capita
    pop = parameters['population']
    transform_t_thist  = get_subset_transformer(dims=dims, dim_letters=('t', 'h'))
    historic_pop       = get_new_array(dims=dims, dim_letters=('h','r'))
    historic_gdppc     = get_new_array(dims=dims, dim_letters=('h','r'))
    historic_stocks_pc = get_new_array(dims=dims, dim_letters=('h','r','g'))
    stocks_pc          = get_new_array(dims=dims, dim_letters=('t','r','g'))
    stocks             = get_new_array(dims=dims, dim_letters=('t','r','g'))

    historic_pop[...] = pop * transform_t_thist
    historic_gdppc[...] = parameters['gdppc'] * transform_t_thist
    historic_stocks_pc[...] = historic_stocks / historic_pop

    if curve_strategy == "GDP_regression":
        gdp_regression(historic_stocks_pc.values, parameters['gdppc'].values, stocks_pc.values)
    elif curve_strategy == 'Exponential_GDP_regression':
        gdp_regression(historic_stocks_pc.values, parameters['gdppc'].values, stocks_pc.values,
                       fitting_function_type='exponential')
    else:
        raise RuntimeError(f"Prediction strategy {curve_strategy} is not defined. "
                           f"It needs to be 'GDP_regression'.")

    # transform back to total stocks
    stocks[...] = stocks_pc * pop

    #visualize_stock(self, self.parameters['gdppc'], historic_gdppc, stocks, historic_stocks, stocks_pc, historic_stocks_pc)
    return StockArray(**dict(stocks))


def scale_parameter_to_future(historic_parameter : NamedDimArray, scaler : NamedDimArray) -> NamedDimArray:
    """Scale parameter to future."""
    n_historic = historic_parameter.shape[0]
    dims = historic_parameter.dims.union_with(scaler.dims)

    future_parameter = get_new_array(dims, dim_letters=('t',) + historic_parameter.dims.letters[1:])
    future_parameter.values[:n_historic] = historic_parameter.values

    # add missing dimensions to scaler values
    scaler_values = scaler.values
    for i, letter in enumerate(historic_parameter.dims.letters[1:],start=1):  # first dim is h which should not be ammended
        if letter not in scaler.dims.letters:
            scaler_values = np.expand_dims(scaler_values, axis=i)

    extrapolation = LinearExtrapolation(data_to_extrapolate=historic_parameter.values,
                                        extrapolate_from=scaler_values)
    future_parameter.values = extrapolation.predict()
    future_parameter.values[:n_historic] = historic_parameter.values

    return Parameter(dims=future_parameter.dims, values=future_parameter.values)


def split_parameter_to_sectors(to_split : NamedDimArray, split_by : NamedDimArray) -> NamedDimArray:
    """
    Split to_split parameter data into the same sector shares as the split_by data, mainly used for in-use goods.
    """
    result_dims = to_split.dims.union_with(split_by.dims)
    intersect_dims = to_split.dims.intersect_with(split_by.dims)
    split_data = get_new_array(dims=result_dims, dim_letters=result_dims.letters)

    sum_split_by = split_by.sum_nda_to(result_dims=intersect_dims.letters)
    sector_splits = split_by / sum_split_by

    split_data[...] = to_split * sector_splits

    return Parameter(dims=split_data.dims, values=split_data.values)


def gdp_regression(historic_stocks_pc, gdppc, prediction_out, fitting_function_type='sigmoid'):
    shape_out = prediction_out.shape
    assert len(shape_out) == 3, "Prediction array must have 3 dimensions: Time, Region, Good"
    pure_prediction = np.zeros_like(prediction_out)
    n_historic = historic_stocks_pc.shape[0]

    for i_region in range(shape_out[1]):
        for i_good in range(shape_out[2]):
            region_category_historic_stock = historic_stocks_pc[:, i_region, i_good]
            regional_gdppc = gdppc[:, i_region]
            if fitting_function_type == 'sigmoid':
                extrapolation = SigmoidalExtrapolation(
                    data_to_extrapolate=region_category_historic_stock,
                    extrapolate_from=regional_gdppc
                )
            elif fitting_function_type == 'exponential':
                extrapolation = ExponentialExtrapolation(
                    data_to_extrapolate=region_category_historic_stock,
                    extrapolate_from=regional_gdppc
                )
            else:
                raise ValueError('fitting_function_type must be either "sigmoid" or "exponential".')
            pure_prediction[:, i_region, i_good] = extrapolation.predict()

    prediction_out[...] = pure_prediction - (
        pure_prediction[n_historic - 1, :, :] - historic_stocks_pc[n_historic - 1, :, :]
        )
    prediction_out[:n_historic,:,:] = historic_stocks_pc


def prepare_stock_for_mfa(
        dims: DimensionSet, dsm: DynamicStockModel, prm: dict[str, Parameter], use: Process
    ):
    # We use an auxiliary stock for the prediction step to save dimensions and computation time
    # Therefore, we have to transfer the result to the higher-dimensional stock in the MFA system
    stock_extd = dsm.stock * prm['material_shares_in_goods'] * prm['carbon_content_materials']
    inflow = dsm.inflow * prm['material_shares_in_goods'] * prm['carbon_content_materials']
    outflow = dsm.outflow * prm['material_shares_in_goods'] * prm['carbon_content_materials']
    stock_dims = dims.get_subset(('t','r','g','m','e'))
    stock_extd = StockArray(values=stock_extd.values, name='in_use_stock', dims=stock_dims)
    inflow = StockArray(values=inflow.values, name='in_use_inflow', dims=stock_dims)
    outflow = StockArray(values=outflow.values, name='in_use_outflow', dims=stock_dims)
    stock = FlowDrivenStock(
        stock=stock_extd, inflow=inflow, outflow=outflow, name='in_use', process_name='use',
        process=use,
    )
    return stock
