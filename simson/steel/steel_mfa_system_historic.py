import numpy as np
import flodym as fd

from simson.common.trade import TradeSet
from simson.common.data_blending import blend


class InflowDrivenHistoricSteelMFASystem(fd.MFASystem):
    trade_set: TradeSet

    def compute(self):
        """
        Perform all computations for the MFA system.
        """
        self.compute_trade()
        self.calc_sector_split()
        self.compute_flows()
        self.compute_in_use_stock()
        self.check_mass_balance()

    def compute_trade(self):
        """
        Create a trade module that stores and calculates the trade flows between regions and sectors.
        """
        for name, trade in self.trade_set.markets.items():
            trade.imports[...] = self.parameters[f"{name}_imports"]
            trade.exports[...] = self.parameters[f"{name}_exports"]
        self.trade_set.balance(to="maximum")

    def compute_flows(self):
        prm = self.parameters
        flw = self.flows
        trd = self.trade_set

        aux = {
            "fabrication_inflow_by_sector": self.get_new_array(dim_letters=("h", "r", "g")),
            "fabrication_loss": self.get_new_array(dim_letters=("h", "r", "g")),
            "fabrication_error": self.get_new_array(dim_letters=("h", "r")),
        }

        # fmt: off
        flw["sysenv => forming"][...] = prm["production_by_intermediate"]
        flw["forming => ip_market"][...] = prm["production_by_intermediate"] * prm["forming_yield"][{'t': self.dims['h']}]
        flw["forming => sysenv"][...] = flw["sysenv => forming"] - flw["forming => ip_market"]

        flw["ip_market => sysenv"][...] = trd["intermediate"].exports
        flw["sysenv => ip_market"][...] = trd["intermediate"].imports

        flw["ip_market => fabrication"][...] = flw["forming => ip_market"] + trd["intermediate"].net_imports

        aux["fabrication_inflow_by_sector"][...] = flw["ip_market => fabrication"] * prm["sector_split"]

        aux["fabrication_error"] = flw["ip_market => fabrication"] - aux["fabrication_inflow_by_sector"]

        flw["fabrication => use"][...] = aux["fabrication_inflow_by_sector"] * prm["fabrication_yield"][{'t': self.dims['h']}]
        aux["fabrication_loss"][...] = aux["fabrication_inflow_by_sector"] - flw["fabrication => use"]
        flw["fabrication => sysenv"][...] = aux["fabrication_error"] + aux["fabrication_loss"]

        # Recalculate indirect trade according to available inflow from fabrication
        trd["indirect"].exports[...] = trd["indirect"].exports.minimum(flw["fabrication => use"])
        trd["indirect"].balance(to="minimum")

        flw["sysenv => use"][...] = trd["indirect"].imports
        flw["use => sysenv"][...] = trd["indirect"].exports
        # fmt: on

    def calc_sector_split(self) -> fd.FlodymArray:
        """Blend over GDP per capita between typical sector splits for low and high GDP per capita regions."""
        target_dims = self.dims["h", "r", "g"]
        self.parameters["sector_split"] = fd.Parameter(dims=target_dims, name="sector_split"
        )
        sector_split_1 = fd.Parameter(dims=target_dims)
        sector_split_2 = fd.Parameter(dims=target_dims)
        log_gdppc = self.parameters["gdppc"][{'t': self.dims['h']}].apply(np.log)
        log_gdppc_low = self.parameters["secsplit_gdppc_low"].apply(np.log)
        log_gdppc_high = self.parameters["secsplit_gdppc_high"].apply(np.log)
        log_gddpc_medium = (log_gdppc_low + log_gdppc_high) / 2

        sector_split_1[...] = blend(
            target_dims=target_dims,
            y_lower=self.parameters["sector_split_low"],
            y_upper=self.parameters["sector_split_medium"],
            x=log_gdppc,
            x_lower=log_gdppc_low,
            x_upper=log_gddpc_medium,
            type="poly_mix",
        )

        sector_split_2[...] = blend(
            target_dims=target_dims,
            y_lower=self.parameters["sector_split_medium"],
            y_upper=self.parameters["sector_split_high"],
            x=log_gdppc,
            x_lower=log_gddpc_medium,
            x_upper=log_gdppc_high,
            type="poly_mix",
        )

        mask = log_gdppc.cast_values_to(target_dims) < log_gddpc_medium.cast_values_to(target_dims)
        self.parameters["sector_split"].values = np.where(
            mask, sector_split_1.values, sector_split_2.values
        )
        return

    def compute_in_use_stock(self):
        flw = self.flows
        stk = self.stocks
        prm = self.parameters
        flw = self.flows

        stk["historic_in_use"].inflow[...] = (
            flw["fabrication => use"] + flw["sysenv => use"] - flw["use => sysenv"]
        )

        stk["historic_in_use"].lifetime_model.set_prms(
            mean=prm["lifetime_mean"][{'t': self.dims['h']}], std=prm["lifetime_std"][{'t': self.dims['h']}]
        )

        stk["historic_in_use"].compute()  # gives stocks and outflows corresponding to inflow

        flw["use => sysenv"][...] += stk["historic_in_use"].outflow
