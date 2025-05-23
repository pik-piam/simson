import numpy as np
from plotly import colors as plc
import plotly.graph_objects as go
import flodym as fd
from typing import TYPE_CHECKING
import flodym.export as fde

from simson.common.common_export import CommonDataExporter
from simson.common.common_cfg import SteelVisualizationCfg

if TYPE_CHECKING:
    from simson.steel.steel_model import SteelModel


class SteelDataExporter(CommonDataExporter):

    cfg: SteelVisualizationCfg

    # Dictionary of variable names vs names displayed in figures. Used by visualization routines.
    _display_names: dict = {
        "sysenv": "System environment",
        "losses": "Losses",
        "imports": "Imports",
        "exports": "Exports",
        "extraction": "Ore<br>Extraction",
        "bof_production": "Production<br>from ores",
        "eaf_production": "Production<br>(EAF)",
        "forming": "Forming",
        "ip_market": "Intermediate<br>products",
        "fabrication": "Fabrication",
        "good_market": "Good Market",
        "in_use": "Use phase",
        "use": "Use phase",
        "obsolete": "Obsolete<br>stocks",
        "eol_market": "End of life<br>products",
        "recycling": "Recycling",
        "scrap_market": "Scrap<br>market",
        "excess_scrap": "Excess<br>scrap",
    }

    def visualize_results(self, model: "SteelModel"):
        if self.cfg.production["do_visualize"]:
            self.visualize_production(mfa=model.future_mfa, regional=True)
            self.visualize_production(mfa=model.future_mfa, regional=False)
        if self.cfg.consumption["do_visualize"]:
            self.visualize_consumption(model.future_mfa)
        if self.cfg.gdppc["do_visualize"]:
            self.visualize_gdppc(
                model.future_mfa, change=False, per_capita=self.cfg.gdppc["per_capita"]
            )
        if self.cfg.trade["do_visualize"]:
            self.visualize_trade(model.future_mfa)
        if self.cfg.use_stock["do_visualize"]:
            self.visualize_use_stock(mfa=model.future_mfa, subplots_by_good=True)
            self.visualize_use_stock(mfa=model.future_mfa, subplots_by_good=False)
        if self.cfg.scrap_demand_supply["do_visualize"]:
            self.visualize_scrap_demand_supply(model.future_mfa, regional=True)
            self.visualize_scrap_demand_supply(model.future_mfa, regional=False)
        if self.cfg.sector_splits["do_visualize"]:
            self.visualize_sector_splits(model.future_mfa, regional=True)
            self.visualize_sector_splits(model.future_mfa, regional=False)
        if self.cfg.sankey["do_visualize"]:
            self.visualize_sankey(model.future_mfa)
        if self.cfg.extrapolation["do_visualize"]:
            self.visualize_extrapolation(model=model)
        self.stop_and_show()

    def visualize_trade(self, mfa: fd.MFASystem):
        linecolor_dims = {
            "intermediate": "Intermediate",
            "indirect": "Good",
            "scrap": "Good",
        }

        for name, trade in mfa.trade_set.markets.items():
            n_colors = mfa.dims[linecolor_dims[name]].len
            colors = plc.qualitative.Dark24[:n_colors] * 2
            ap_imports = self.plotter_class(
                array=trade.imports.sum_over(trade.imports.dims[linecolor_dims[name]].letter),
                intra_line_dim="Time",
                subplot_dim="Region",
                # linecolor_dim=linecolor_dims[name],
                display_names=self._display_names,
                color_map=colors,
            )
            fig = ap_imports.plot()
            ap_exports = self.plotter_class(
                array=-trade.exports.sum_over(trade.exports.dims[linecolor_dims[name]].letter),
                intra_line_dim="Time",
                subplot_dim="Region",
                # linecolor_dim=linecolor_dims[name],
                line_type="dash",
                display_names=self._display_names,
                title=f"{name} Trade",
                ylabel="Trade (Exports negative)",
                suppress_legend=True,
                fig=fig,
                color_map=colors,
            )
            fig = ap_exports.plot()
            self.plot_and_save_figure(ap_exports, f"trade_{name}.png", do_plot=False)

    def visualize_consumption(self, mfa: fd.MFASystem):
        consumption = mfa.stocks["in_use"].inflow
        good_dim = consumption.dims.index("g")
        consumption = consumption.apply(np.cumsum, kwargs={"axis": good_dim})
        ap = self.plotter_class(
            array=consumption,
            intra_line_dim="Time",
            subplot_dim="Region",
            linecolor_dim="Good",
            chart_type="area",
            display_names=self._display_names,
            title="Consumption",
        )
        fig = ap.plot()
        self.plot_and_save_figure(ap, "consumption.png", do_plot=False)

    def visualize_gdppc(self, mfa: fd.MFASystem, change=False, per_capita=False):
        gdppc = mfa.parameters["gdppc"]
        if not per_capita:
            gdppc = gdppc * mfa.parameters["population"]
        if change:
            gdppc = gdppc.apply(np.diff, kwargs={"axis": 0, "prepend": 0})
            gdppc[1900] = gdppc[1901]
        ap = self.plotter_class(
            array=gdppc,
            intra_line_dim="Time",
            linecolor_dim="Region",
            display_names=self._display_names,
            title=f"GDP{' per capita' if per_capita else ''}{' growth rate' if change else ''}",
        )
        fig = ap.plot()
        if change:
            self.plot_and_save_figure(ap, "gdppc_change.png", do_plot=False)
        else:
            fig.update_yaxes(type="log")
            self.plot_and_save_figure(ap, "gdppc.png", do_plot=False)

    def visualize_sankey(self, mfa: fd.MFASystem):
        good_colors = [f"hsl({190 + 10 *i},40,{77-5*i})" for i in range(4)]
        production_color = "hsl(50,40,70)"
        scrap_color = "hsl(120,40,70)"
        losses_color = "hsl(20,40,70)"
        trade_color = "hsl(260,20,80)"

        flow_color_dict = {"default": production_color}
        flow_color_dict.update(
            {fn: ("Good", good_colors) for fn, f in mfa.flows.items() if "Good" in f.dims}
        )
        flow_color_dict.update(
            {
                fn: scrap_color
                for fn, f in mfa.flows.items()
                if f.from_process.name == "scrap_market" or f.to_process.name == "scrap_market"
            }
        )
        flow_color_dict.update(
            {
                fn: losses_color
                for fn, f in mfa.flows.items()
                if f.to_process.name in ["losses", "excess_scrap", "obsolete"]
            }
        )
        flow_color_dict.update(
            {
                fn: trade_color
                for fn, f in mfa.flows.items()
                if f.from_process.name == "imports" or f.to_process.name == "exports"
            }
        )
        self.cfg.sankey["flow_color_dict"] = flow_color_dict

        self.cfg.sankey["node_color_dict"] = {"default": "gray", "use": "black"}

        sdn = {k: f"<b>{v}</b>" for k, v in self._display_names.items()}
        plotter = fde.PlotlySankeyPlotter(mfa=mfa, display_names=sdn, **self.cfg.sankey)
        fig = plotter.plot()

        legend_entries = [
            [production_color, "Production Phase"],
            [scrap_color, "Scrap Treatment"],
            [losses_color, "Losses and Waste"],
            ["white", ""],
            ["white", "Product Phase"],
        ]
        for good, color in zip(mfa.dims["Good"].items, good_colors):
            # legend_entries.append([color, f"Product Phase ({good})"])
            legend_entries.append([color, good])

        for entry in legend_entries:
            fig.add_trace(
                go.Scatter(
                    mode="markers",
                    x=[None],
                    y=[None],
                    marker=dict(size=10, color=entry[0], symbol="square"),
                    name=entry[1],
                )
            )

        fig.update_layout(
            # title_text=f"Steel Flows ({', '.join([str(v) for v in self.sankey['slice_dict'].values()])})",
            font_size=18,
            showlegend=True,
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="black",
        )
        fig.update_xaxes(visible=False)
        fig.update_yaxes(visible=False)

        self._show_and_save_plotly(fig, name="sankey")

    def visualize_production(self, mfa: fd.MFASystem, regional=True):
        flw = mfa.flows
        production = flw["bof_production => forming"] + flw["eaf_production => forming"]

        subplot_dim, summing_func, name_str = self._get_regional_vs_global_params(regional)

        # visualize regional production
        ap_production = self.plotter_class(
            array=summing_func(production),
            intra_line_dim="Time",
            **subplot_dim,
            line_label="Production",
            display_names=self._display_names,
            xlabel="Year",
            ylabel="Production [t]",
            title=f"Steel Production {name_str}",
        )

        self.plot_and_save_figure(ap_production, f"production_{name_str}.png")

    def visualize_use_stock(self, mfa: fd.MFASystem, subplots_by_good=False):
        subplot_dim = "Good" if subplots_by_good else None
        super().visualize_use_stock(mfa, stock=mfa.stocks["in_use"].stock, subplot_dim=subplot_dim)

    def visualize_scrap_demand_supply(self, mfa: fd.MFASystem, regional=True):

        subplot_dim, summing_func, name_str = self._get_regional_vs_global_params(regional)

        flw = mfa.flows
        prm = mfa.parameters

        total_production = (
            flw["forming => ip_market"] / (prm["forming_yield"] * prm["production_yield"])
        )[{"t": mfa.dims["h"]}]
        scrap_supply = (
            flw["recycling => scrap_market"]
            + flw["forming => scrap_market"]
            + flw["fabrication => scrap_market"]
        )
        scrap_supply = scrap_supply[{"t": mfa.dims["h"]}]

        ap = self.plotter_class(
            array=summing_func(scrap_supply),
            intra_line_dim="Historic Time",
            **subplot_dim,
            line_label="Model",
            # fig=fig,
            display_names=self._display_names,
        )
        fig = ap.plot()

        ap = self.plotter_class(
            array=summing_func(mfa.parameters["scrap_consumption"]),
            intra_line_dim="Historic Time",
            **subplot_dim,
            line_label="Real World",
            fig=fig,
            xlabel="Year",
            ylabel="Scrap [t]",
            display_names=self._display_names,
            title="Scrap Demand and Supply",
        )

        fig = ap.plot()

        ap = self.plotter_class(
            array=summing_func(total_production.sum_to(("h", "r"))),
            intra_line_dim="Historic Time",
            **subplot_dim,
            line_label="Total Production",
            line_type="dash",
            fig=fig,
        )

        for trade_name, trade in mfa.trade_set.markets.items():
            fig = ap.plot()

            ap = self.plotter_class(
                array=summing_func(trade.net_imports[{"t": mfa.dims["h"]}].sum_to(("h", "r"))),
                intra_line_dim="Historic Time",
                **subplot_dim,
                line_label=f"Net imports ({trade_name})",
                line_type="dot",
                fig=fig,
            )

        self.plot_and_save_figure(ap, f"scrap_demand_supply_{name_str}.png")

    def visualize_sector_splits(self, mfa: fd.MFASystem, regional: bool = True):

        subplot_dim, summing_func, name_str = self._get_regional_vs_global_params(regional)

        flw = mfa.flows

        fabrication = summing_func(flw["good_market => use"])
        sector_splits = fabrication.get_shares_over("g")
        sector_splits = sector_splits.cumsum(dim_letter="g")

        ap_sector_splits = self.plotter_class(
            array=sector_splits,
            intra_line_dim="Time",
            **subplot_dim,
            linecolor_dim="Good",
            xlabel="Year",
            ylabel="Sector Splits [%]",
            display_names=self._display_names,
            title=f"Product demand sector splits ({name_str})",
            chart_type="area",
        )

        self.plot_and_save_figure(ap_sector_splits, f"sector_splits_{name_str}.png")

    def _get_regional_vs_global_params(self, regional: bool):
        if regional:
            subplot_dim = {"subplot_dim": "Region"}
            summing_func = lambda l: l
            name_str = "regional"
        else:
            subplot_dim = {}
            summing_func = lambda l: l.sum_over("r")
            name_str = "global"
        return subplot_dim, summing_func, name_str

    def visualize_extrapolation(self, model: "SteelModel"):
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

        pure_stock = model.stock_handler.pure_prediction.sum_over(other_dimletters)

        # extrapolation
        color = ["red"]
        ap_pure_prediction = self.plotter_class(
            array=pure_stock,
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
