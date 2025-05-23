from plotly import colors as plc
import flodym as fd
from typing import TYPE_CHECKING

from simson.common.common_export import CommonDataExporter
from simson.common.common_cfg import CementVisualizationCfg

if TYPE_CHECKING:
    from simson.cement.cement_model import CementModel


class CementDataExporter(CommonDataExporter):
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
        super().visualize_use_stock(mfa, stock=mfa.stocks["in_use"].stock, subplot_dim=subplot_dim)

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
        mfa = model.future_mfa
        per_capita = True  # TODO see where this shold go
        subplot_dim = "Region"
        stock = mfa.stocks["in_use"].stock
        population = mfa.parameters["population"]
        x_array = None

        pc_str = "pC" if per_capita else ""
        x_label = "Year"
        y_label = f"Stock{pc_str} [t]"
        title = f"Stock Extrapolation: Historic and Projected vs Pure Prediction"
        if self.cfg.use_stock["over_gdp"]:
            title = title + f" over GDP{pc_str}"
            x_label = f"GDP/PPP{pc_str} [2005 USD]"
            x_array = mfa.parameters["gdppc"]
            if not per_capita:
                x_array = x_array * population

        if subplot_dim is None:
            dimlist = ["t"]
        else:
            subplot_dimletter = next(
                dimlist.letter for dimlist in mfa.dims.dim_list if dimlist.name == subplot_dim
            )
            dimlist = ["t", subplot_dimletter]

        other_dimletters = tuple(letter for letter in stock.dims.letters if letter not in dimlist)
        stock = stock.sum_over(other_dimletters)

        if per_capita:
            stock = stock / population

        fig, ap_final_stock = self.plot_history_and_future(
            mfa=mfa,
            data_to_plot=stock,
            subplot_dim=subplot_dim,
            x_array=x_array,
            x_label=x_label,
            y_label=y_label,
            title=title,
            line_label="Historic + Modelled Future",
        )

        # extrapolation
        color = ["red"]
        ap_pure_prediction = self.plotter_class(
            array=model.stock_handler.pure_prediction,
            intra_line_dim="Time",
            subplot_dim=subplot_dim,
            x_array=x_array,
            title=title,
            fig=fig,
            # color_map=color,
            line_type="dot",
            line_label="Pure Extrapolation",
        )
        fig = ap_pure_prediction.plot()

        self.plot_and_save_figure(
            ap_pure_prediction,
            f"stocks_extrapolation.png",
            do_plot=False,
        )
