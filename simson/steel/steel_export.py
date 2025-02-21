from plotly import colors as plc
import plotly.graph_objects as go
import numpy as np
import flodym as fd
import flodym.export as fde

from simson.common.custom_export import CustomDataExporter


class SteelDataExporter(CustomDataExporter):
    scrap_demand_supply: dict = {"do_visualize": True}
    sector_splits: dict = {"do_visualize": True}

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
        # 'ip_trade': 'Intermediate product trade',  # todo decide whether to incorporate, depending on trade balancing
        "fabrication": "Fabrication",
        # 'indirect_trade': 'Indirect trade', # todo decide whether to incorporate, depending on trade balancing
        "in_use": "Use phase",
        "use": "Use phase",
        "obsolete": "Obsolete<br>stocks",
        "eol_market": "End of life<br>products",
        # 'eol_trade': 'End of life trade', # todo decide whether to incorporate, depending on trade balancing
        "recycling": "Recycling",
        "scrap_market": "Scrap<br>market",
        "excess_scrap": "Excess<br>scrap",
    }

    def visualize_results(self, historic_mfa: fd.MFASystem, future_mfa: fd.MFASystem):
        if self.production["do_visualize"]:
            self.visualize_production(mfa=future_mfa)
        if self.stock["do_visualize"]:
            self.visualize_stock(mfa=future_mfa)
        if self.scrap_demand_supply["do_visualize"]:
            self.visualize_scrap_demand_supply(future_mfa, regional=True)
            self.visualize_scrap_demand_supply(future_mfa, regional=False)
        if self.sector_splits["do_visualize"]:
            self.visualize_sector_splits(future_mfa)
        if self.sankey["do_visualize"]:
            self.visualize_sankey(future_mfa)
        self.stop_and_show()

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
            {fn: scrap_color for fn, f in mfa.flows.items() if f.from_process.name == "scrap_market" or f.to_process.name == "scrap_market"}
        )
        flow_color_dict.update(
            {fn: losses_color for fn, f in mfa.flows.items() if f.to_process.name in ["losses", "excess_scrap", "obsolete"]}
        )
        flow_color_dict.update(
            {fn: trade_color for fn, f in mfa.flows.items() if f.from_process.name == "imports" or f.to_process.name == "exports"}
        )
        self.sankey["flow_color_dict"] = flow_color_dict

        self.sankey["node_color_dict"] = {"default": "gray", "use": "black"}


        sdn = {k: f"<b>{v}</b>" for k, v in self._display_names.items()}
        plotter = fde.PlotlySankeyPlotter(mfa=mfa, display_names=sdn, **self.sankey)
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

    def visualize_production(self, mfa: fd.MFASystem):
        flw = mfa.flows
        production = flw["bof_production => forming"] + flw["eaf_production => forming"]
        production = production.sum_over("e")

        # visualize regional production
        ap_production = self.plotter_class(
            array=production,
            intra_line_dim="Time",
            subplot_dim="Region",
            line_label="Production",
            display_names=self._display_names,
            xlabel="Year",
            ylabel="Production [t]",
            title="Regional Steel Production",
        )

        self.plot_and_save_figure(ap_production, "production_by_region.png")

        # visualize global production
        production = production.sum_over("r")

        ap_production = self.plotter_class(
            array=production,
            intra_line_dim="Time",
            line_label="Production",
            display_names=self._display_names,
            xlabel="Year",
            ylabel="Production [t]",
            title="Global Steel Production",
        )

        self.plot_and_save_figure(ap_production, "production_global.png")

    def visualize_stock(self, mfa: fd.MFASystem):
        over_gdp = self.stock["over_gdp"]
        per_capita = self.stock["per_capita"]

        stock = mfa.stocks["in_use"].stock
        population = mfa.parameters["population"]
        x_array = None

        pc_str = " pC" if per_capita else ""
        x_label = "Year"
        y_label = f"Stock{pc_str}[t]"
        title = f"Stocks{pc_str}"
        if over_gdp:
            title = title + f"over GDP{pc_str}"

        if over_gdp:
            x_array = mfa.parameters["gdppc"]
            x_label = f"GDP/PPP{pc_str}[2005 USD]"

        self.visualize_regional_stock(
            stock, x_array, population, x_label, y_label, title, per_capita, over_gdp
        )
        self.visiualize_global_stock(
            mfa, stock, x_array, population, x_label, y_label, title, per_capita, over_gdp
        )

    def visualize_regional_stock(
            self, stock, x_array, population, x_label, y_label, title, per_capita, over_gdp
    ):
        if per_capita:
            stock = stock / population
        else:
            if over_gdp:
                x_array = x_array * population

        ap_stock = self.plotter_class(
            array=stock,
            intra_line_dim="Time",
            subplot_dim="Region",
            linecolor_dim="Good",
            display_names=self._display_names,
            xlabel=x_label,
            x_array=x_array,
            ylabel=y_label,
            title=f"{title} (regional)",
            area=True,
        )

        self.plot_and_save_figure(ap_stock, "stocks_by_region.png")

    def visiualize_global_stock(
            self, mfa, stock, x_array, population, x_label, y_label, title, per_capita, over_gdp
    ):
        if over_gdp:
            x_array = x_array * population
            if per_capita:
                # get global GDP per capita
                x_array = x_array / population

        self.visualize_global_stock_by_region(mfa, stock, x_array, population, x_label, y_label, title, per_capita, subplots_by_good=True)
        self.visualize_global_stock_by_region(mfa, stock, x_array, population, x_label, y_label, title, per_capita, subplots_by_good=False)

    def visualize_global_stock_by_region(self, mfa, stock, x_array, population, x_label, y_label, title, per_capita, subplots_by_good=False):

        if subplots_by_good:
            subplot_dim = {"subplot_dim": "Good"}
        else:
            subplot_dim = {}
            stock = stock.sum_over("g")

        if per_capita:
            stock = stock / population

        colors = plc.qualitative.Dark24
        colors = colors[: stock.dims["r"].len] + colors[: stock.dims["r"].len] + ["black" for _ in range(stock.dims["r"].len)]

        ap_stock = self.plotter_class(
            array=stock,
            intra_line_dim="Time",
            linecolor_dim="Region",
            **subplot_dim,
            display_names=self._display_names,
            x_array=x_array,
            xlabel=x_label,
            ylabel=y_label,
            title=f"{title} (global by region{' per capita' if per_capita else ''})",
            color_map=colors,
            line_type="dot",
            suppress_legend=True,
        )
        fig = ap_stock.plot()

        hist_stock = stock[{"t": mfa.dims["h"]}]
        hist_x_array = x_array[{"t": mfa.dims["h"]}]

        ap_hist_stock = self.plotter_class(
            array=hist_stock,
            intra_line_dim="Historic Time",
            linecolor_dim="Region",
            **subplot_dim,
            display_names=self._display_names,
            x_array=hist_x_array,
            fig=fig,
            color_map=colors,
        )
        fig = ap_hist_stock.plot()

        last_year_dim = fd.Dimension(name="Last Historic Year", letter="l", items=[mfa.dims["h"].items[-1]])
        scatter_stock = hist_stock[{"h": last_year_dim}]
        scatter_x_array = hist_x_array[{"h": last_year_dim}]
        ap_scatter_stock = self.plotter_class(
            array=scatter_stock,
            intra_line_dim="Last Historic Year",
            linecolor_dim="Region",
            **subplot_dim,
            display_names=self._display_names,
            x_array=scatter_x_array,
            fig=fig,
            chart_type="scatter",
            color_map=colors,
            suppress_legend=True,
        )
        fig = ap_scatter_stock.plot()

        # adjust lower bound of x-axis
        fig.update_xaxes(type="log", range=[3, 5])

        self.plot_and_save_figure(ap_scatter_stock, f"stocks_global_by_region{'_per_capita' if per_capita else ''}.png", do_plot=False)

    def visualize_scrap_demand_supply(self, mfa: fd.MFASystem, regional=True):

        if regional:
            subplot_dim = {"subplot_dim": "Region"}
            summing_func = lambda l: l
            name_str = "regional"
        else:
            subplot_dim = {}
            summing_func = lambda l: l.sum_over("r")
            name_str = "global"

        flw = mfa.flows
        prm = mfa.parameters

        total_production = (
                flw["forming => ip_market"] / prm["forming_yield"] / prm["production_yield"]
        )[{'t': mfa.dims['h']}]
        scrap_supply = (
                flw["recycling => scrap_market"]
                + flw["forming => scrap_market"]
                + flw["fabrication => scrap_market"]
        )
        scrap_supply = scrap_supply[{'t': mfa.dims['h']}].sum_over("e")

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
                array=summing_func(trade.net_imports[{'t': mfa.dims['h']}].sum_to(("h", "r"))),
                intra_line_dim="Historic Time",
                **subplot_dim,
                line_label=f"Net imports ({trade_name})",
                line_type="dot",
                fig=fig,
            )

        self.plot_and_save_figure(ap, f"scrap_demand_supply_{name_str}.png")

    def visualize_sector_splits(self, mfa: fd.MFASystem):
        flw = mfa.flows
        prm = mfa.parameters

        fabrication = flw["fabrication => use"] / prm["fabrication_yield"]
        fabrication = fabrication.sum_over(("e",))
        sector_splits = fabrication.get_shares_over("g")
        sector_splits = sector_splits.cumsum(dim_letter="g")

        ap_sector_splits = self.plotter_class(
            array=sector_splits,
            intra_line_dim="Time",
            subplot_dim="Region",
            linecolor_dim="Good",
            xlabel="Year",
            ylabel="Sector Splits [%]",
            display_names=self._display_names,
            title="Regional Fabrication Sector Splits",
            chart_type="area",
        )

        self.plot_and_save_figure(ap_sector_splits, "sector_splits_regional.png")

        # plot global sector splits
        fabrication = fabrication.sum_over("r")
        sector_splits = fabrication.get_shares_over("g")
        i_axis = sector_splits.dims.index("g")
        sector_splits = sector_splits.apply(lambda a: np.cumsum(a, axis=i_axis))

        ap_sector_splits = self.plotter_class(
            array=sector_splits,
            intra_line_dim="Time",
            linecolor_dim="Good",
            xlabel="Year",
            ylabel="Sector Splits [%]",
            display_names=self._display_names,
            title="Global Fabrication Sector Splits",
            chart_type="area",
        )

        self.plot_and_save_figure(ap_sector_splits, "sector_splits_global.png")
