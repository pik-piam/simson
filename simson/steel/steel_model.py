import numpy as np
import flodym as fd

from simson.common.data_blending import blend, blend_over_time
from simson.common.common_cfg import GeneralCfg
from simson.common.data_extrapolations import LogSigmoidExtrapolation
from simson.common.data_transformations import Bound, BoundList
from simson.common.stock_extrapolation import StockExtrapolation
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
            cfg=self.cfg.visualization,
            do_export=self.cfg.do_export,
            output_path=self.cfg.output_path,
        )
        self.dims = self.data_reader.read_dimensions(self.definition.dimensions)
        self.parameters = self.data_reader.read_parameters(
            self.definition.parameters, dims=self.dims
        )
        self.modify_parameters()
        self.processes = fd.make_processes(self.definition.processes)

    def modify_parameters(self):
        """Manual changes to parameters in order to match historical scrap consumption."""
        lifetime_factor = fd.Parameter(dims=self.dims["t", "r"])
        lifetime_factor.values[...] = 1.3
        self.parameters["lifetime_factor"] = lifetime_factor

        self.parameters["lifetime_mean"] = fd.Parameter(
            dims=self.dims["t", "r", "g"],
            values=(self.parameters["lifetime_factor"] * self.parameters["lifetime_mean"]).values,
        )
        self.parameters["lifetime_std"] = fd.Parameter(
            dims=self.dims["t", "r", "g"],
            values=(self.parameters["lifetime_factor"] * self.parameters["lifetime_std"]).values,
        )
        self.parameters["lifetime_mean"]["Construction"] = (
            self.parameters["lifetime_mean"]["Construction"] * 1.5
        )
        self.parameters["lifetime_std"]["Construction"] = (
            self.parameters["lifetime_std"]["Construction"] * 1.5
        )

        scrap_rate_factor = fd.Parameter(dims=self.dims["t",])
        scrap_rate_factor.values[:80] = 1.4
        scrap_rate_factor.values[80:110] = np.linspace(1.4, 0.8, 30)
        scrap_rate_factor.values[110:] = 0.8
        self.parameters["forming_yield"] = fd.Parameter(
            dims=self.dims["t", "i"],
            values=(1 - scrap_rate_factor * (1 - self.parameters["forming_yield"])).values,
        )
        self.parameters["fabrication_yield"] = fd.Parameter(
            dims=self.dims["t", "g"],
            values=(1 - scrap_rate_factor * (1 - self.parameters["fabrication_yield"])).values,
        )

    def run(self):
        self.historic_mfa = self.make_historic_mfa()
        self.historic_mfa.compute()

        self.future_mfa = self.make_future_mfa()
        future_demand = self.get_future_demand()
        self.future_mfa.compute(future_demand, self.historic_mfa.trade_set)

        self.data_writer.export_mfa(mfa=self.future_mfa)
        self.data_writer.visualize_results(model=self)

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
            "good_market",
            "fabrication",
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
        demand = self.get_demand_from_stock(long_term_stock)
        return demand

    def get_long_term_stock(self):
        indep_fit_dim_letters = (
            ("g",) if self.cfg.customization.do_stock_extrapolation_by_category else ()
        )
        historic_stocks = self.historic_mfa.stocks["historic_in_use"].stock
        sat_level = self.get_saturation_level(historic_stocks)
        sat_bound = Bound(
            var_name="saturation_level",
            lower_bound=sat_level,
            upper_bound=sat_level,
            dims=self.dims[indep_fit_dim_letters],
        )
        bound_list = BoundList(
            bound_list=[
                sat_bound,
            ],
            target_dims=self.dims[indep_fit_dim_letters],
        )
        # extrapolate in use stock to future
        stock_handler = StockExtrapolation(
            historic_stocks,
            dims=self.dims,
            parameters=self.parameters,
            stock_extrapolation_class=self.cfg.customization.stock_extrapolation_class,
            target_dim_letters=(
                "all" if self.cfg.customization.do_stock_extrapolation_by_category else ("t", "r")
            ),
            indep_fit_dim_letters=indep_fit_dim_letters,
            bound_list=bound_list,
        )
        total_in_use_stock = stock_handler.stocks

        if not self.cfg.customization.do_stock_extrapolation_by_category:
            # calculate and apply sector splits for in use stock
            sector_splits = self.calc_stock_sector_splits()
            total_in_use_stock = total_in_use_stock * sector_splits
        return total_in_use_stock

    def get_saturation_level(self, historic_stocks: fd.StockArray):
        pop = self.parameters["population"]
        gdppc = self.parameters["gdppc"]
        historic_pop = pop[{"t": self.dims["h"]}]
        historic_stocks_pc = historic_stocks.sum_over("g") / historic_pop

        multi_dim_extrapolation = LogSigmoidExtrapolation(
            data_to_extrapolate=historic_stocks_pc.values,
            target_range=gdppc.values,
            independent_dims=(),
        )
        multi_dim_extrapolation.regress()
        saturation_level = multi_dim_extrapolation.fit_prms[0, 0]

        if self.cfg.customization.do_stock_extrapolation_by_category:
            high_stock_sector_split = self.get_high_stock_sector_split()
            saturation_level = saturation_level * high_stock_sector_split.values

        return saturation_level

    def get_high_stock_sector_split(self):
        prm = self.parameters
        last_lifetime = prm["lifetime_mean"][{"t": self.dims["t"].items[-1]}]
        last_gdppc = prm["gdppc"][{"t": self.dims["t"].items[-1]}]
        av_lifetime = (last_lifetime * last_gdppc).sum_over("r") / last_gdppc.sum_over("r")
        high_stock_sector_split = (av_lifetime * prm["sector_split_high"]).get_shares_over("g")
        return high_stock_sector_split

    def calc_stock_sector_splits(self):
        historical_sector_splits = self.historic_mfa.stocks[
            "historic_in_use"
        ].stock.get_shares_over("g")
        prm = self.parameters
        sector_split_high = self.get_high_stock_sector_split()
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
