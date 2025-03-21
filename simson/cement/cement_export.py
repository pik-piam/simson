from plotly import colors as plc
import flodym as fd
from typing import TYPE_CHECKING

from simson.common.common_export import CommonDataExporter
from simson.common.common_cfg import CementVisualizationCfg

if TYPE_CHECKING:
    from simson.cement.cement_model import CementModel


class CementDataExporter(CommonDataExporter):
    # They have to be defined here but are eventually overwritten by yml definitions
    cfg: CementVisualizationCfg

    _display_names: dict = {
        "sysenv": "System environment",
        "raw_meal_preparation": "Raw meal preparation",
        "clinker_production": "Clinker production",
        "cement_grinding": "Cement grinding",
        "concrete_production": "Concrete production",
        "use": "Use phase",
        "eol": "End of life",
    }

    def visualize_results(self, model: "CementModel"):
        if self.cfg.clinker_production["do_visualize"]:
            self.visualize_clinker_production(mfa=model.future_mfa)
        if self.cfg.cement_production["do_visualize"]:
            self.visualize_cement_production(mfa=model.future_mfa, regional=False)
            self.visualize_cement_production(mfa=model.future_mfa, regional=True)
        if self.cfg.concrete_production["do_visualize"]:
            self.visualize_concrete_production(mfa=model.future_mfa)
        if self.cfg.use_stock["do_visualize"]:
            self.visualize_use_stock(mfa=model.future_mfa, subplots_by_stock_type=False)
            self.visualize_use_stock(mfa=model.future_mfa, subplots_by_stock_type=True)
        if self.cfg.eol_stock["do_visualize"]:
            self.visualize_eol_stock(mfa=model.future_mfa)
        if self.cfg.sankey["do_visualize"]:
            self.visualize_sankey(mfa=model.future_mfa)
        if self.cfg.extrapolation["do_visualize"]:
            self.visualize_extrapolation(model=model)
        self.stop_and_show()

    def visualize_production(
        self, mfa: fd.MFASystem, production: fd.Flow, name: str, regional: bool = False
    ):

        x_array = None
        # intra_line_dim = "Time"
        # line_label = f"{name} Production"
        x_label = "Year"
        y_label = "Production [t]"
        linecolor_dim = None

        if regional:
            subplot_dim = "Region"
            title = f"Regional {name} Production"
            regional_tag = "_regional"
        else:
            subplot_dim = None
            regional_tag = ""
            title = f"Global {name} Production"
            production = production.sum_over("r")

        fig, ap_production = self.plot_history_and_future(
            mfa=mfa,
            data_to_plot=production,
            subplot_dim=subplot_dim,
            x_array=x_array,
            linecolor_dim=linecolor_dim,
            x_label=x_label,
            y_label=y_label,
            title=title,
            line_label="Production",
        )

        self.plot_and_save_figure(
            ap_production, f"{name}_production{regional_tag}.png", do_plot=False
        )

    def visualize_clinker_production(self, mfa: fd.MFASystem):
        production = mfa.flows["clinker_production => cement_grinding"]
        self.visualize_production(production, "Clinker")

    def visualize_cement_production(self, mfa: fd.MFASystem, regional: bool = False):
        production = mfa.flows["cement_grinding => concrete_production"]
        self.visualize_production(mfa=mfa, production=production, name="Cement", regional=regional)

    def visualize_concrete_production(self, mfa: fd.MFASystem):
        production = mfa.flows["concrete_production => use"].sum_over("s")
        self.visualize_production(production, "Concrete")

    def visualize_eol_stock(self, mfa: fd.MFASystem):
        over_gdp = self.cfg.eol_stock["over_gdp"]
        per_capita = self.cfg.eol_stock["per_capita"]
        stock = mfa.stocks["eol"].stock

        self.visualize_stock(mfa, stock, over_gdp, per_capita, "EOL")

    def visualize_use_stock(self, mfa: fd.MFASystem, subplots_by_stock_type=False):
        subplot_dim = "Stock Type" if subplots_by_stock_type else None
        super().visualize_use_stock(mfa, subplot_dim=subplot_dim)

    def visualize_stock(self, mfa: fd.MFASystem, stock, over_gdp, per_capita, name):
        population = mfa.parameters["population"]
        x_array = None

        pc_str = " pC" if per_capita else ""
        x_label = "Year"
        y_label = f"{name} Stock{pc_str}[t]"
        title = f"{name} stocks{pc_str}"
        if over_gdp:
            title = title + f" over GDP{pc_str}"

        if over_gdp:
            x_array = mfa.parameters["gdppc"]
            x_label = f"GDP/PPP{pc_str}[2005 USD]"

        # self.visualize_regional_stock(
        #     stock, x_array, population, x_label, y_label, title, per_capita, over_gdp
        # )
        self.visualize_global_stock(
            stock, x_array, population, x_label, y_label, title, per_capita, over_gdp
        )

    def visualize_global_stock(
        self, stock, x_array, population, x_label, y_label, title, per_capita, over_gdp
    ):
        if over_gdp:
            x_array = x_array * population
            x_array = x_array.sum_over("r")
            if per_capita:
                # get global GDP per capita
                x_array = x_array / population.sum_over("r")

        self.visualize_global_stock_by_type(
            stock, x_array, population, x_label, y_label, title, per_capita
        )
        # self.visualize_global_stock_by_region(stock, x_array, x_label, y_label, title, per_capita)

    def visualize_global_stock_by_type(
        self, stock, x_array, population, x_label, y_label, title, per_capita
    ):
        if "r" in stock.dims.letters:
            stock = stock.sum_over("r")
        stock = stock / population.sum_over("r") if per_capita else stock

        ap_stock = self.plotter_class(
            array=stock,
            intra_line_dim="Time",
            linecolor_dim="Stock Type",
            display_names=self._display_names,
            x_array=x_array,
            xlabel=x_label,
            ylabel=y_label,
            title=f"{title} (global by stock type)",
            area=True,
        )

        self.plot_and_save_figure(ap_stock, "use_stocks_global_by_type.png")

    def visualize_extrapolation(self, model: "CementModel"):
        """This needs to be reworked"""
        historic_mfa = model.historic_mfa
        historic_stock = historic_mfa.stocks["historic_in_use"].stock.sum_over("s")
        historic_stock_pc = historic_stock
        historic_stock_pc.values = (
            historic_stock.values / model.parameters["population"].values[:124]
        )
        gdppc = model.parameters["gdppc"]
        historic_gdppc = fd.FlodymArray(
            dims=model.dims[
                "h",
                "r",
            ]
        )
        historic_gdppc.values = gdppc.values[:124]

        fit = model.stock_handler.extrapolation_class.func(
            historic_gdppc.values, model.stock_handler.fit_prms.T
        )
        fd_fit = fd.FlodymArray(dims=historic_gdppc.dims, values=fit)

        ap_fit = self.plotter_class(
            array=fd_fit,
            intra_line_dim="Historic Time",
            subplot_dim="Region",
            line_label=f"Fit",
            display_names=self._display_names,
            title=f"Regional Stock",
        )

        ap_fit.plot(do_show=False)
        fig = ap_fit.fig

        ap_stock = self.plotter_class(
            array=historic_stock_pc,
            intra_line_dim="Historic Time",
            subplot_dim="Region",
            line_label=f"DSM",
            display_names=self._display_names,
            xlabel="Time",
            ylabel="Stock pc [t]",
            title=f"Regional Stock: Extrapolation Fit",
            fig=fig,
        )

        self.plot_and_save_figure(ap_stock, f"Stock_regional.png")
