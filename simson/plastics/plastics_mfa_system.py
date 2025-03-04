from typing import Optional
import flodym as fd

from simson.common.data_transformations import StockExtrapolation
from simson.common.common_cfg import PlasticsCfg


class PlasticsMFASystem(fd.MFASystem):

    cfg: Optional[PlasticsCfg] = None

    def compute(self):
        """
        Perform all computations for the MFA system.
        """
        self.compute_historic_stock()
        self.compute_in_use_dsm()
        self.transfer_to_simple_stock()
        self.compute_flows()
        self.compute_other_stocks()
        self.check_mass_balance()

    def compute_historic_stock(self):
        self.stocks["in_use_historic"].inflow[...] = self.parameters["production"]
        self.stocks["in_use_historic"].lifetime_model.set_prms(
            mean=self.parameters["lifetime_mean"], std=self.parameters["lifetime_std"]
        )
        self.stocks["in_use_historic"].compute()

    def compute_in_use_dsm(self):
        stock_handler = StockExtrapolation(
            self.stocks["in_use_historic"].stock,
            dims=self.dims,
            parameters=self.parameters,
            stock_extrapolation_class=self.cfg.customization.stock_extrapolation_class,
        )
        in_use_stock = stock_handler.stocks
        self.stocks["in_use_dsm"].stock[...] = in_use_stock
        self.stocks["in_use_dsm"].lifetime_model.set_prms(
            mean=self.parameters["lifetime_mean"], std=self.parameters["lifetime_std"]
        )
        self.stocks["in_use_dsm"].compute()

    def transfer_to_simple_stock(self):
        # We use an auxiliary stock for the prediction step to save dimensions and computation time
        # Therefore, we have to transfer the result to the higher-dimensional stock in the MFA system
        split = (
            self.parameters["material_shares_in_goods"]
            * self.parameters["carbon_content_materials"]
        )
        self.stocks["in_use"].stock[...] = self.stocks["in_use_dsm"].stock * split
        self.stocks["in_use"].inflow[...] = self.stocks["in_use_dsm"].inflow * split
        self.stocks["in_use"].outflow[...] = self.stocks["in_use_dsm"].outflow * split

    def compute_flows(self):

        # abbreviations for better readability
        prm = self.parameters
        flw = self.flows
        stk = self.stocks

        aux = {
            "reclmech_loss": self.get_new_array(dim_letters=("t", "e", "r", "m")),
            "virgin_2_fabr_all_mat": self.get_new_array(dim_letters=("t", "e", "r")),
            "virgin_material_shares": self.get_new_array(dim_letters=("t", "e", "r", "m")),
            "captured_2_virginccu_by_mat": self.get_new_array(dim_letters=("t", "e", "r", "m")),
            "ratio_nonc_to_c": self.get_new_array(dim_letters=("m",)),
        }

        # non-C atmosphere & captured has no meaning & is equivalent to sysenv

        # fmt: off
        flw["fabrication => use"][...] = stk["in_use"].inflow
        flw["use => eol"][...] = stk["in_use"].outflow

        flw["eol => reclmech"][...] = flw["use => eol"] * prm["mechanical_recycling_rate"]
        flw["reclmech => recl"][...] = flw["eol => reclmech"] * prm["mechanical_recycling_yield"]
        aux["reclmech_loss"][...] = flw["eol => reclmech"] - flw["reclmech => recl"]
        flw["reclmech => uncontrolled"][...] = aux["reclmech_loss"] * prm["reclmech_loss_uncontrolled_rate"]
        flw["reclmech => incineration"][...] = aux["reclmech_loss"] - flw["reclmech => uncontrolled"]

        flw["eol => reclchem"][...] = flw["use => eol"] * prm["chemical_recycling_rate"]
        flw["reclchem => recl"][...] = flw["eol => reclchem"]

        flw["eol => reclsolv"][...] = flw["use => eol"] * prm["solvent_recycling_rate"]
        flw["reclsolv => recl"][...] = flw["eol => reclsolv"]

        flw["eol => incineration"][...] = flw["use => eol"] * prm["incineration_rate"]
        flw["eol => uncontrolled"][...] = flw["use => eol"] * prm["uncontrolled_losses_rate"]

        flw["eol => landfill"][...] = (
            flw["use => eol"]
            - flw["eol => reclmech"]
            - flw["eol => reclchem"]
            - flw["eol => reclsolv"]
            - flw["eol => incineration"]
            - flw["eol => uncontrolled"]
        )

        flw["incineration => emission"][...] = flw["eol => incineration"] + flw["reclmech => incineration"]

        flw["emission => captured"][...] = flw["incineration => emission"] * prm["emission_capture_rate"]
        flw["emission => atmosphere"][...] = flw["incineration => emission"] - flw["emission => captured"]
        flw["captured => virginccu"][...] = flw["emission => captured"]

        flw["recl => fabrication"][...] = flw["reclmech => recl"] + flw["reclchem => recl"] + flw["reclsolv => recl"]
        flw["virgin => fabrication"][...] = flw["fabrication => use"] - flw["recl => fabrication"]

        flw["virgindaccu => virgin"][...] = flw["virgin => fabrication"] * prm["daccu_production_rate"]
        flw["virginbio => virgin"][...] = flw["virgin => fabrication"] * prm["bio_production_rate"]

        aux["virgin_2_fabr_all_mat"][...] = flw["virgin => fabrication"]
        aux["virgin_material_shares"][...] = flw["virgin => fabrication"] / aux["virgin_2_fabr_all_mat"]
        aux["captured_2_virginccu_by_mat"][...] = flw["captured => virginccu"] * aux["virgin_material_shares"]

        flw["virginccu => virgin"]["C"] = aux["captured_2_virginccu_by_mat"]["C"]
        aux["ratio_nonc_to_c"][...] = prm["carbon_content_materials"]["Other Elements"] / prm["carbon_content_materials"]["C"]
        flw["virginccu => virgin"]["Other Elements"] = flw["virginccu => virgin"]["C"] * aux["ratio_nonc_to_c"]

        flw["virginfoss => virgin"][...] = (
            flw["virgin => fabrication"]
            - flw["virgindaccu => virgin"]
            - flw["virginbio => virgin"]
            - flw["virginccu => virgin"]
        )

        flw["sysenv => virginfoss"][...] = flw["virginfoss => virgin"]
        flw["atmosphere => virginbio"][...] = flw["virginbio => virgin"]
        flw["atmosphere => virgindaccu"][...] = flw["virgindaccu => virgin"]
        flw["sysenv => virginccu"][...] = flw["virginccu => virgin"] - aux["captured_2_virginccu_by_mat"]
        # fmt: on

    def compute_other_stocks(self):

        stk = self.stocks
        flw = self.flows

        # in-use stock is already computed in compute_in_use_stock

        stk["landfill"].inflow[...] = flw["eol => landfill"]
        stk["landfill"].compute()

        stk["uncontrolled"].inflow[...] = (
            flw["eol => uncontrolled"] + flw["reclmech => uncontrolled"]
        )
        stk["uncontrolled"].compute()

        stk["atmospheric"].inflow[...] = flw["emission => atmosphere"]
        stk["atmospheric"].outflow[...] = (
            flw["atmosphere => virgindaccu"] + flw["atmosphere => virginbio"]
        )
        stk["atmospheric"].compute()
