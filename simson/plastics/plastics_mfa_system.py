from typing import Optional
import flodym as fd
import numpy as np

from simson.common.stock_extrapolation import StockExtrapolation
from simson.common.common_cfg import PlasticsCfg
from simson.common.data_transformations import Bound, BoundList


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
        self.check_flows(no_error=True)

    def compute_historic_stock(self):
        self.stocks["in_use_historic"].inflow[...] = self.parameters["production"]
        self.stocks["in_use_historic"].lifetime_model.set_prms(
            mean=self.parameters["lifetime_mean"], std=self.parameters["lifetime_std"]
        )
        self.stocks["in_use_historic"].compute()

    def compute_in_use_dsm(self):
        saturation_level = 0.2 / 1e6  # t to Mt
        sat_bound = Bound(
            var_name="saturation_level",
            lower_bound=saturation_level,
            upper_bound=saturation_level * 3,
            dims=self.dims[()],
        )
        bound_list = BoundList(
            bound_list=[
                sat_bound,
            ],
            target_dims=self.dims[()],
        )
        stock_handler = StockExtrapolation(
            self.stocks["in_use_historic"].stock,
            dims=self.dims,
            parameters=self.parameters,
            stock_extrapolation_class=self.cfg.customization.stock_extrapolation_class,
            bound_list=bound_list,
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
            "final_2_fabrication": self.get_new_array(dim_letters=("t", "e", "m")),
        }

        # non-C atmosphere & captured has no meaning & is equivalent to sysenv
        split = prm["material_shares_in_goods"] * prm["carbon_content_materials"]

        # fmt: off
        flw["fabrication => use"][...] = stk["in_use"].inflow
        flw["use => eol"][...] = stk["in_use"].outflow

        flw["wastetrade => wasteimport"][...] = prm["wasteimport_rate"] * prm["wasteimporttotal"]  * split
        flw["wasteexport => wastetrade"][...] = prm["wasteexport_rate"] * prm["wasteimporttotal"]  * split
        flw["wasteimport => collected"][...] = flw["wastetrade => wasteimport"]
        flw["collected => wasteexport"][...] = flw["wasteexport => wastetrade"]

        # aux["final_2_fabrication"][...] = (
        #     flw["fabrication => use"]
        #     .sum_over(["r","g"]).get_shares_over(["e","m"])
        # )

        #flw["finaltrade => finalimport"][...] = prm["finalimport_rate"] * prm["finalimporttotal"] * aux["final_2_fabrication"]
        #flw["finalexport => finaltrade"][...] = prm["finalexport_rate"] * prm["finalimporttotal"] * aux["final_2_fabrication"]
        #flw["finalimport => fabrication"][...] = flw["finaltrade => finalimport"]
        #flw["fabrication => finalexport"][...] = flw["finalexport => finaltrade"]

        flw["eol => collected"][...] = flw["use => eol"] * prm["collection_rate"]
        flw["collected => reclmech"][...] = (flw["eol => collected"] + flw["wasteimport => collected"] - flw["collected => wasteexport"]) * prm["mechanical_recycling_rate"]
        flw["reclmech => recl"][...] = flw["collected => reclmech"] * prm["mechanical_recycling_yield"]
        aux["reclmech_loss"][...] = flw["collected => reclmech"] - flw["reclmech => recl"]
        flw["reclmech => uncontrolled"][...] = aux["reclmech_loss"] * prm["reclmech_loss_uncontrolled_rate"]
        flw["reclmech => incineration"][...] = aux["reclmech_loss"] - flw["reclmech => uncontrolled"]

        flw["collected => reclchem"][...] = (flw["eol => collected"] + flw["wasteimport => collected"] - flw["collected => wasteexport"]) * prm["chemical_recycling_rate"]
        flw["reclchem => recl"][...] = flw["collected => reclchem"]

        flw["collected => incineration"][...] = (flw["eol => collected"] + flw["wasteimport => collected"] - flw["collected => wasteexport"]) * prm["incineration_rate"]

        flw["collected => landfill"][...] = (
            flw["eol => collected"]
            + flw["wasteimport => collected"]
            - flw["collected => wasteexport"]
            - flw["collected => reclmech"]
            - flw["collected => reclchem"]
            - flw["collected => incineration"]
        )

        flw["eol => mismanaged"][...] = (
            flw["use => eol"]
            - flw["eol => collected"]
        )

        flw["mismanaged => uncontrolled"][...] = (
            flw["eol => mismanaged"]
        )

        flw["incineration => emission"][...] = flw["collected => incineration"] + flw["reclmech => incineration"]

        flw["emission => captured"][...] = flw["incineration => emission"] * prm["emission_capture_rate"]
        flw["emission => atmosphere"][...] = flw["incineration => emission"] - flw["emission => captured"]
        flw["captured => virginccu"][...] = flw["emission => captured"]

        flw["recl => fabrication"][...] = flw["reclmech => recl"] + flw["reclchem => recl"]
        #flw["virgin => fabrication"][...] = flw["fabrication => use"] - flw["recl => fabrication"] - flw["finalimport => fabrication"] + flw["fabrication => finalexport"]
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

        stk["landfill"].inflow[...] = flw["collected => landfill"]
        stk["landfill"].compute()

        stk["uncontrolled"].inflow[...] = flw["eol => mismanaged"] + flw["reclmech => uncontrolled"]
        stk["uncontrolled"].compute()

        stk["wastetrade"].inflow[...] = flw["wasteexport => wastetrade"]
        stk["wastetrade"].outflow[...] = flw["wastetrade => wasteimport"]
        stk["wastetrade"].compute()

        stk["atmospheric"].inflow[...] = flw["emission => atmosphere"]
        stk["atmospheric"].outflow[...] = (
            flw["atmosphere => virgindaccu"] + flw["atmosphere => virginbio"]
        )
        stk["atmospheric"].compute()
