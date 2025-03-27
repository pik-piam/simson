import flodym as fd

from simson.common.common_cfg import GeneralCfg
from simson.common.data_transformations import Bound, BoundList
from simson.cement.cement_definition import get_definition
from simson.cement.cement_mfa_system_historic import (
    InflowDrivenHistoricCementMFASystem,
)
from simson.cement.cement_mfa_system_historic import InflowDrivenHistoricCementMFASystem
from simson.cement.cement_mfa_system_future import StockDrivenCementMFASystem
from simson.cement.cement_data_reader import CementDataReader
from simson.cement.cement_export import CementDataExporter
from simson.common.stock_extrapolation import StockExtrapolation


class CementModel:

    def __init__(self, cfg: GeneralCfg):
        self.cfg = cfg
        self.definition = get_definition(self.cfg)
        self.data_reader = CementDataReader(
            input_data_path=self.cfg.input_data_path, definition=self.definition
        )
        self.data_writer = CementDataExporter(
            cfg=self.cfg.visualization,
            do_export=self.cfg.do_export,
            output_path=self.cfg.output_path,
        )
        self.dims = self.data_reader.read_dimensions(self.definition.dimensions)
        self.parameters = self.data_reader.read_parameters(
            self.definition.parameters, dims=self.dims
        )
        self.processes = fd.make_processes(self.definition.processes)

    def run(self):
        # historic mfa
        self.historic_mfa = self.make_historic_mfa()
        self.historic_mfa.compute()

        # future mfa
        self.future_mfa = self.make_future_mfa()
        future_stock = self.get_long_term_stock()
        self.future_mfa.compute(future_stock)

        # visualization and export
        self.data_writer.export_mfa(mfa=self.future_mfa)
        self.data_writer.visualize_results(model=self)

        # visualize extrapolation
        # self.data_writer.visualize_extrapolation(mfa=self.historic_mfa, future_demand=future_demand)

    def make_historic_mfa(self) -> InflowDrivenHistoricCementMFASystem:
        historic_dim_letters = tuple([d for d in self.dims.letters if d != "t"])
        historic_dims = self.dims[historic_dim_letters]
        historic_processes = [
            "sysenv",
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
        return InflowDrivenHistoricCementMFASystem(
            cfg=self.cfg,
            parameters=self.parameters,
            processes=processes,
            dims=historic_dims,
            flows=flows,
            stocks=stocks,
        )

    def get_long_term_stock(self) -> fd.FlodymArray:
        # extrapolate in use stock to future
        indep_fit_dim_letters = ("r",)
        sat_bound = Bound(var_name="saturation_level", lower_bound=100, upper_bound=300)
        bound_list = BoundList(
            bound_list=[
                sat_bound,
            ],
            target_dims=self.dims[indep_fit_dim_letters],
        )
        self.stock_handler = StockExtrapolation(
            self.historic_mfa.stocks["historic_in_use"].stock,
            dims=self.dims,
            parameters=self.parameters,
            stock_extrapolation_class=self.cfg.customization.stock_extrapolation_class,
            target_dim_letters=("t", "r"),
            indep_fit_dim_letters=indep_fit_dim_letters,
            bound_list=bound_list,
        )

        total_in_use_stock = self.stock_handler.stocks

        total_in_use_stock = total_in_use_stock * self.parameters["use_split"]
        return total_in_use_stock

    def make_future_mfa(self) -> StockDrivenCementMFASystem:
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

        return StockDrivenCementMFASystem(
            dims=self.dims,
            parameters=self.parameters,
            processes=self.processes,
            flows=flows,
            stocks=stocks,
        )
