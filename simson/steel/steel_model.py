from sodym import (
    MFADefinition, DimensionDefinition, FlowDefinition, ParameterDefinition, StockDefinition,
    Process, FlowDrivenStock
)
from sodym.stock_helper import create_dynamic_stock, make_empty_stocks
from sodym.flow_helper import make_empty_flows

from simson.common.common_cfg import CommonCfg
from simson.common.data_transformations import extrapolate_stock
from simson.common.custom_data_reader import CustomDataReader
from simson.common.custom_export import CustomDataExporter
from simson.steel.stock_driven_steel import StockDrivenSteelMFASystem
from simson.steel.inflow_driven_steel_historic import InflowDrivenHistoricSteelMFASystem


class SteelModel:

    def __init__(self, cfg: CommonCfg):
        self.cfg = cfg
        self.definition = self.set_up_definition()
        self.data_reader = CustomDataReader(input_data_path=self.cfg.input_data_path)
        self.data_writer = CustomDataExporter(
            **dict(self.cfg.visualization), output_path=self.cfg.output_path,
            display_names=self.display_names
        )

        self.dims = self.data_reader.read_dimensions(self.definition.dimensions)
        # TODO: confirm all required data is being loaded
        # loading the steel production data
        # loading steel direct and indirect trade data (
        # direct is for intermediate steel products, indirect for finished products like cars)
        # loading steel sector splits for intermediate products, and indirect trade
        self.parameters = self.data_reader.read_parameters(self.definition.parameters, dims=self.dims)
        self.scalar_parameters = self.data_reader.read_scalar_data(self.definition.scalar_parameters)

        self.processes = {
            name: Process(name=name, id=id) for id, name in enumerate(self.definition.processes)
        }

    def run(self):
        historic_mfa = self.make_historic_mfa()
        historic_mfa.compute()
        historic_in_use_stock = historic_mfa.stocks['in_use'].stock
        future_in_use_stock = self.create_future_stock_from_historic(historic_in_use_stock)
        mfa = self.make_future_mfa(future_in_use_stock)
        mfa.compute()
        self.data_writer.export_mfa(mfa=mfa)
        self.data_writer.visualize_results(mfa=mfa)

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


        historic_dims = self.dims.get_subset(('h', 'r', 'i', 'g'))
        flows = make_empty_flows(
            processes=self.processes,
            flow_definitions=[f for f in self.definition.flows if 'h' in f.dim_letters],
            dims=historic_dims
        )
        stocks = make_empty_stocks(
            processes=self.processes,
            stock_definitions=[s for s in self.definition.stocks if 'h' in s.dim_letters],
            dims=historic_dims
        )

        historic_processes = [
            'sysenv',
            'forming',
            'ip_market',
            'ip_trade',
            'fabrication',
            'indirect_trade',
            'use'
        ]

        processes = {p : self.processes[p] for p in historic_processes}
        return InflowDrivenHistoricSteelMFASystem(
            cfg=self.cfg,
            parameters=self.parameters,
            scalar_parameters=self.scalar_parameters,
            processes=processes,
            dims=historic_dims,
            flows=flows,
            stocks=stocks,
        )

    def create_future_stock_from_historic(self, historic_in_use_stock):
        # TODO: predict the In-Use stock according to saturation assumptions similar to Pauliuks 'Steel Scrap Age'. i.e. we need another different extrapolate_stock function
        in_use_stock = extrapolate_stock(
            historic_in_use_stock, dims=self.dims, parameters=self.parameters,
            curve_strategy=self.cfg.model_customization.curve_strategy,
        )
        dsm = create_dynamic_stock(
            name='in_use', process=self.processes['use'],
            ldf_type=self.cfg.model_customization.ldf_type,
            stock=in_use_stock, lifetime_mean=self.parameters['lifetime_mean'],
            lifetime_std=self.parameters['lifetime_std'],
        )
        dsm.compute()  # gives inflows and outflows corresponding to in-use stock
        return FlowDrivenStock(
        stock=dsm.stock, inflow=dsm.inflow, outflow=dsm.outflow, name='in_use', process_name='use',
        process=self.processes['use'],
    )
        """return prepare_stock_for_mfa(
            dsm=dsm, dims=self.dims, prm=self.parameters, use=self.processes['use']
        )"""

    def make_future_mfa(self, future_in_use_stock):
        future_dims = self.dims.drop('h', inplace=False)
        flows = make_empty_flows(
            processes=self.processes,
            flow_definitions=[f for f in self.definition.flows if 't' in f.dim_letters],
            dims=future_dims
        )
        stocks = make_empty_stocks(
            processes=self.processes,
            stock_definitions=[s for s in self.definition.stocks if 't' in s.dim_letters],
            dims=future_dims
        )
        stocks['in_use'] = future_in_use_stock
        return StockDrivenSteelMFASystem(
            dims=future_dims, parameters=self.parameters, scalar_parameters=self.scalar_parameters,
            processes=self.processes, flows=flows, stocks=stocks,
        )


    # Dictionary of variable names vs names displayed in figures. Used by visualization routines.
    display_names = {
        'sysenv': 'System environment',
        'bof_production': 'Production (BF/BOF)',
        'eaf_production': 'Production (EAF)',
        'forming': 'Forming',
        'fabrication_buffer': 'Fabrication Buffer',
        'ip_market': 'Intermediate product market',
        'ip_trade': 'Intermediate product trade',
        'fabrication': 'Fabrication',
        'indirect_trade': 'Indirect trade',
        'in_use': 'Use phase',
        'outflow_buffer': 'Outflow buffer',
        'obsolete': 'Obsolete stocks',
        'eol_market': 'End of life product market',
        'eol_trade': 'End of life trade',
        'recycling': 'Recycling',
        'scrap_market': 'Scrap market',
        'excess_scrap': 'Excess scrap'
    }

    def set_up_definition(self):
        dimensions = [
            DimensionDefinition(name='Time', dim_letter='t', dtype=int),
            DimensionDefinition(name='Element', dim_letter='e', dtype=str),
            DimensionDefinition(name='Region', dim_letter='r', dtype=str),
            DimensionDefinition(name='Intermediate', dim_letter='i', dtype=str),
            DimensionDefinition(name='Good', dim_letter='g', dtype=str),
            DimensionDefinition(name='Scenario', dim_letter='s', dtype=str),
            DimensionDefinition(name='Historic Time', dim_letter='h', dtype=int),
        ]

        processes = [
            'sysenv',
            'bof_production',
            'eaf_production',
            'forming',
            'fabrication_buffer',
            'ip_market',
            'ip_trade',
            'fabrication',
            'indirect_trade',
            'use',
            'outflow_buffer',
            'obsolete',
            'eol_market',
            'eol_trade',
            'recycling',
            'scrap_market',
            'excess_scrap'
        ]

        # names are auto-generated, see Flow class documetation
        flows = [
            # Historic Flows

            FlowDefinition(from_process='sysenv', to_process='forming', dim_letters=('h', 'r', 'i')),
            FlowDefinition(from_process='forming', to_process='ip_market', dim_letters=('h', 'r', 'i')),
            FlowDefinition(from_process='forming', to_process='sysenv', dim_letters=('h', 'r')),
            FlowDefinition(from_process='ip_market', to_process='fabrication', dim_letters=('h', 'r', 'i')),
            FlowDefinition(from_process='ip_market', to_process='ip_trade', dim_letters=('h', 'r', 'i')),
            FlowDefinition(from_process='ip_trade', to_process='ip_market', dim_letters=('h', 'r', 'i')),
            FlowDefinition(from_process='fabrication', to_process='use', dim_letters=(('h', 'r', 'g'))),
            FlowDefinition(from_process='fabrication', to_process='sysenv', dim_letters=('h', 'r')),
            FlowDefinition(from_process='use', to_process='indirect_trade', dim_letters=('h', 'r', 'g')),
            FlowDefinition(from_process='indirect_trade', to_process='use', dim_letters=('h', 'r', 'g')),

            # Future Flows


            FlowDefinition(from_process='sysenv', to_process='bof_production', dim_letters=('t', 'r')),
            FlowDefinition(from_process='scrap_market', to_process='bof_production', dim_letters=('t', 'r')),
            FlowDefinition(from_process='bof_production', to_process='forming', dim_letters=('t','r')),
            FlowDefinition(from_process='bof_production', to_process='sysenv', dim_letters=('t','r',)),
            FlowDefinition(from_process='scrap_market', to_process='eaf_production', dim_letters=('t', 'r')),
            FlowDefinition(from_process='eaf_production', to_process='forming', dim_letters=('t', 'r')),
            FlowDefinition(from_process='eaf_production', to_process='sysenv', dim_letters=('t', 'r')),
            FlowDefinition(from_process='forming', to_process='ip_market', dim_letters=('t', 'r', 'i')),
            FlowDefinition(from_process='forming', to_process='fabrication_buffer', dim_letters=('t', 'r')),
            FlowDefinition(from_process='forming', to_process='sysenv', dim_letters=('t', 'r')),
            FlowDefinition(from_process='ip_market', to_process='fabrication', dim_letters=('t', 'r', 'i')),
            FlowDefinition(from_process='ip_market', to_process='ip_trade', dim_letters=('t', 'r', 'i')),
            FlowDefinition(from_process='ip_trade', to_process='ip_market', dim_letters=('t', 'r', 'i')),
            FlowDefinition(from_process='fabrication', to_process='use', dim_letters=('t', 'r', 'g')),
            FlowDefinition(from_process='fabrication', to_process='fabrication_buffer', dim_letters=('t', 'r')),
            FlowDefinition(from_process='fabrication_buffer', to_process='scrap_market', dim_letters=('t', 'r')),
            FlowDefinition(from_process='use', to_process='outflow_buffer', dim_letters=('t', 'r', 'g')),
            FlowDefinition(from_process='use', to_process='indirect_trade', dim_letters=('t', 'r', 'g')),
            FlowDefinition(from_process='indirect_trade', to_process='use', dim_letters=('t', 'r', 'g')),
            FlowDefinition(from_process='outflow_buffer', to_process='obsolete', dim_letters=('t', 'r', 'g')),
            FlowDefinition(from_process='outflow_buffer', to_process='eol_market', dim_letters=('t', 'r', 'g')),
            FlowDefinition(from_process='eol_market', to_process='recycling', dim_letters=('t', 'r', 'g')),
            FlowDefinition(from_process='eol_market', to_process='eol_trade', dim_letters=('t', 'r', 'g')),
            FlowDefinition(from_process='eol_trade', to_process='eol_market', dim_letters=('t', 'r', 'g')),
            FlowDefinition(from_process='sysenv', to_process='recycling', dim_letters=('t', 'r', 'g')),
            FlowDefinition(from_process='recycling', to_process='scrap_market', dim_letters=('t', 'r', 'g')),
            FlowDefinition(from_process='scrap_market', to_process='excess_scrap', dim_letters=('t', 'r'))
        ]

        stocks = [
            StockDefinition(name='in_use', process='use', dim_letters=('h', 'r', 'g')),
            StockDefinition(name='use', process='use', dim_letters=('t', 'r', 'g')),
            StockDefinition(name='outflow_buffer', process='outflow_buffer', dim_letters=('t', 'r', 'g')),
            StockDefinition(name='obsolete', process='obsolete', dim_letters=('t', 'r', 'g')),
            StockDefinition(name='fabrication_buffer', process='fabrication_buffer', dim_letters=('t', 'r')),
            StockDefinition(name='excess_scrap', process='excess_scrap', dim_letters=('t', 'r'))
        ]

        parameters = [
            ParameterDefinition(name='forming_yield', dim_letters=('i',)),
            ParameterDefinition(name='fabrication_yield', dim_letters=('g',)),
            ParameterDefinition(name='recovery_rate', dim_letters=('g',)),
            ParameterDefinition(name='external_copper_rate', dim_letters=('g',)),
            ParameterDefinition(name='cu_tolerances', dim_letters=('i',)),
            ParameterDefinition(name='good_to_intermediate_distribution', dim_letters=('g', 'i')),

            ParameterDefinition(name='production', dim_letters=('h', 'r')),
            ParameterDefinition(name='production_by_intermediate', dim_letters=('h', 'r', 'i')),
            ParameterDefinition(name='population', dim_letters=('t', 'r')),
            ParameterDefinition(name='gdppc', dim_letters=('t', 'r')),
            ParameterDefinition(name='lifetime_mean', dim_letters=('r', 'g')),
            ParameterDefinition(name='lifetime_std', dim_letters=('r', 'g')),
        ]

        scalar_parameters = ['max_scrap_share_base_model','scrap_in_bof_rate','forming_losses','production_yield']

        return MFADefinition(
            dimensions=dimensions,
            processes=processes,
            flows=flows,
            stocks=stocks,
            parameters=parameters,
            scalar_parameters=scalar_parameters,
        )
