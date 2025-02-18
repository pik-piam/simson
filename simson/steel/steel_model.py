import numpy as np
import flodym as fd
import flodym.export as fde

from simson.common.data_blending import blend, blend_over_time
from simson.common.common_cfg import GeneralCfg
from simson.common.data_transformations import extrapolate_stock, extrapolate_to_future
from simson.common.custom_data_reader import CustomDataReader
from simson.common.trade import TradeSet
from simson.steel.steel_export import SteelDataExporter
from simson.steel.steel_mfa_system_future import StockDrivenSteelMFASystem
from simson.steel.steel_mfa_system_historic import InflowDrivenHistoricSteelMFASystem
from simson.steel.steel_definition import get_definition


class SteelModel:

    def __init__(self, cfg: GeneralCfg):
        self.cfg = cfg
        self.definition = get_definition(self.cfg)
        self.data_reader = CustomDataReader(
            input_data_path=self.cfg.input_data_path, definition=self.definition
        )
        self.data_writer = SteelDataExporter(
            **dict(self.cfg.visualization),
            output_path=self.cfg.output_path,
        )
        self.dims = self.data_reader.read_dimensions(self.definition.dimensions)
        self.parameters = self.data_reader.read_parameters(
            self.definition.parameters, dims=self.dims
        )
        self.processes = fd.make_processes(self.definition.processes)

    def run(self):
        self.historic_mfa = self.make_historic_mfa()
        self.historic_mfa.compute()

        self.future_mfa = self.make_future_mfa()
        future_demand = self.get_future_demand()
        self.future_mfa.compute(future_demand, self.historic_mfa.trade_set)

        self.data_writer.export_mfa(mfa=self.future_mfa)
        self.data_writer.visualize_results(mfa=self.future_mfa)

    def make_historic_mfa(self) -> InflowDrivenHistoricSteelMFASystem:
        """
        Splitting production and direct trade by IP sector splits, and indirect trade by category trade sector splits (s. step 3)
        subtracting Losses in steel forming from production by IP data
        adding direct trade by IP to production by IP
        transforming that to production by category via some distribution assumptions
        subtracting losses in steel fabrication (transformation of IP to end use products)
        adding indirect trade by category
        This equals the inflow into the in use stock
        via lifetime assumptions I can calculate in use stock from inflow into in use stock and lifetime
        """

        historic_dim_letters = tuple([d for d in self.dims.letters if d != "t"])
        historic_dims = self.dims[historic_dim_letters]
        historic_processes = [
            "sysenv",
            "forming",
            "ip_market",
            # 'ip_trade', # todo decide whether to incorporate, depending on trade balancing
            "fabrication",
            # 'indirect_trade', # todo decide whether to incorporate, depending on trade balancing
            "use",
        ]
        processes = fd.make_processes(historic_processes)
        flows = fd.make_empty_flows(
            processes=processes,
            flow_definitions=[f for f in self.definition.flows if "h" in f.dim_letters],
            dims=historic_dims,
        )
        stocks = fd.make_empty_stocks(
            processes=processes,
            stock_definitions=[s for s in self.definition.stocks if "h" in s.dim_letters],
            dims=historic_dims,
        )
        trade_set = TradeSet.from_definitions(
            definitions=[td for td in self.definition.trades if "h" in td.dim_letters],
            dims=historic_dims,
        )

        return InflowDrivenHistoricSteelMFASystem(
            cfg=self.cfg,
            parameters=self.parameters,
            processes=processes,
            dims=historic_dims,
            flows=flows,
            stocks=stocks,
            trade_set=trade_set,
        )

    def get_future_demand(self):
        long_term_stock = self.get_long_term_stock()
        long_term_demand = self.get_demand_from_stock(long_term_stock)
        short_term_demand = self.get_short_term_demand_trend(
            historic_demand=self.historic_mfa.stocks["historic_in_use"].inflow,
        )
        demand = blend_over_time(
            target_dims=long_term_demand.dims,
            y_lower=short_term_demand,
            y_upper=long_term_demand,
            t_lower=self.historic_mfa.dims["h"].items[-1],
            t_upper=self.historic_mfa.dims["h"].items[-1] + 20,
        )
        return demand

    def get_long_term_stock(self):
        # extrapolate in use stock to future
        total_in_use_stock = extrapolate_stock(
            self.historic_mfa.stocks["historic_in_use"].stock,
            dims=self.dims,
            parameters=self.parameters,
            curve_strategy=self.cfg.customization.curve_strategy,
            target_dim_letters=("t", "r"),
        )

        # calculate and apply sector splits for in use stock
        sector_splits = self.calc_stock_sector_splits(
            self.historic_mfa.stocks["historic_in_use"].stock.get_shares_over("g"),
        )
        long_term_stock = total_in_use_stock * sector_splits
        return long_term_stock

    def calc_stock_sector_splits(self, historical_sector_splits: fd.FlodymArray):
        prm = self.parameters
        sector_split_high = (prm["lifetime_mean"] * prm["sector_split_high"]).get_shares_over("g")
        sector_split_theory = blend(
            target_dims=self.dims["t", "r", "g"],
            y_lower=prm["sector_split_low"],
            y_upper=sector_split_high,
            x=prm["gdppc"].apply(np.log),
            x_lower=float(np.log(1000)),
            x_upper=float(np.log(100000)),
        )
        last_historical = historical_sector_splits[{"h": self.dims["h"].items[-1]}]
        historical_extrapolated = last_historical.cast_to(self.dims["t", "r", "g"])
        historical_extrapolated[{"t": self.dims["h"]}] = historical_sector_splits
        sector_splits = blend_over_time(
            target_dims=self.dims["t", "r", "g"],
            y_lower=historical_extrapolated,
            y_upper=sector_split_theory,
            t_lower=self.dims["h"].items[-1],
            t_upper=self.dims["t"].items[-1],
            type="converge_quadratic",
        )

        # # DEBUG
        # array_dict = {
        #     "Theory": sector_split_theory,
        #     "Historical": historical_extrapolated,
        #     "Blended": sector_splits,
        # }
        # for name, array in array_dict.items():
        #     plotter = fde.PlotlyArrayPlotter(
        #         array=array,
        #         intra_line_dim="Time",
        #         subplot_dim="Region",
        #         linecolor_dim="Good",
        #         title=name,
        #     )
        #     plotter.plot(do_show=True)
        return sector_splits

    def get_demand_from_stock(self, long_term_stock):
        # create dynamic stock model for in use stock
        in_use_dsm_long_term = fd.StockDrivenDSM(
            dims=self.dims["t", "r", "g"],
            lifetime_model=self.cfg.customization.lifetime_model,
        )
        in_use_dsm_long_term.lifetime_model.set_prms(
            mean=self.parameters["lifetime_mean"], std=self.parameters["lifetime_std"]
        )
        in_use_dsm_long_term.stock[...] = long_term_stock
        in_use_dsm_long_term.compute()
        return in_use_dsm_long_term.inflow

    def get_short_term_demand_trend(self, historic_demand: fd.FlodymArray):
        demand_via_gdp = extrapolate_to_future(historic_demand, scale_by=self.parameters["gdppc"])
        return demand_via_gdp

    def make_future_mfa(self) -> StockDrivenSteelMFASystem:
        flows = fd.make_empty_flows(
            processes=self.processes,
            flow_definitions=[f for f in self.definition.flows if "t" in f.dim_letters],
            dims=self.dims,
        )
        stocks = fd.make_empty_stocks(
            processes=self.processes,
            stock_definitions=[s for s in self.definition.stocks if "t" in s.dim_letters],
            dims=self.dims,
        )

        trade_set = TradeSet.from_definitions(
            definitions=[td for td in self.definition.trades if "t" in td.dim_letters],
            dims=self.dims,
        )

        return StockDrivenSteelMFASystem(
            dims=self.dims,
            parameters=self.parameters,
            processes=self.processes,
            flows=flows,
            stocks=stocks,
            trade_set=trade_set,
        )
