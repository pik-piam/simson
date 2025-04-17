import os
from matplotlib import pyplot as plt
import plotly.graph_objects as go
import plotly.colors as plc
import plotly.io as pio
from typing import Optional
from pydantic import model_validator
import flodym as fd
import flodym.export as fde

from simson.common.base_model import SimsonBaseModel
from simson.common.common_cfg import VisualizationCfg, ExportCfg
from simson.common.assumptions_doc import assumptions_str


class CommonDataExporter(SimsonBaseModel):
    output_path: str
    do_export: ExportCfg
    cfg: VisualizationCfg
    _display_names: dict = {}

    @model_validator(mode="after")
    def set_plotly_renderer(self):
        if self.cfg.plotting_engine == "plotly":
            pio.renderers.default = self.cfg.plotly_renderer
        return self

    def export_mfa(self, mfa: fd.MFASystem):
        if self.do_export.pickle:
            fde.export_mfa_to_pickle(mfa=mfa, export_path=self.export_path("mfa.pickle"))
        if self.do_export.csv:
            dir_out = os.path.join(self.export_path(), "flows")
            fde.export_mfa_flows_to_csv(mfa=mfa, export_directory=dir_out)
            fde.export_mfa_stocks_to_csv(mfa=mfa, export_directory=dir_out)
        if self.do_export.assumptions:
            file_out = os.path.join(self.export_path("assumptions.txt"))
            with open(file_out, "w") as f:
                f.write(assumptions_str())

    def export_path(self, filename: str = None):
        path_tuple = (self.output_path, "export")
        if filename is not None:
            path_tuple += (filename,)
        return os.path.join(*path_tuple)

    def figure_path(self, filename: str):
        return os.path.join(self.output_path, "figures", filename)

    def _show_and_save_plotly(self, fig: go.Figure, name):
        if self.cfg.do_save_figs:
            fig.write_image(self.figure_path(f"{name}.png"))
        if self.cfg.do_show_figs:
            fig.show()

    def visualize_sankey(self, mfa: fd.MFASystem):
        plotter = fde.PlotlySankeyPlotter(
            mfa=mfa, display_names=self._display_names, **self.cfg.sankey
        )
        fig = plotter.plot()

        fig.update_layout(
            # title_text=f"Steel Flows ({', '.join([str(v) for v in self.sankey['slice_dict'].values()])})",
            font_size=20,
        )

        self._show_and_save_plotly(fig, name="sankey")

    def figure_path(self, filename: str) -> str:
        return os.path.join(self.output_path, "figures", filename)

    def plot_and_save_figure(self, plotter: fde.ArrayPlotter, filename: str, do_plot: bool = True):
        if do_plot:
            plotter.plot()
        if self.cfg.do_show_figs:
            plotter.show()
        if self.cfg.do_save_figs:
            plotter.save(self.figure_path(filename), width=2200, height=1300)

    def stop_and_show(self):
        if self.cfg.plotting_engine == "pyplot" and self.cfg.do_show_figs:
            plt.show()

    @property
    def plotter_class(self):
        if self.cfg.plotting_engine == "plotly":
            return fde.PlotlyArrayPlotter
        elif self.cfg.plotting_engine == "pyplot":
            return fde.PyplotArrayPlotter
        else:
            raise ValueError(f"Unknown plotting engine: {self.cfg.plotting_engine}")

    def visualize_use_stock(
        self, mfa: fd.MFASystem, stock: fd.FlodymArray, subplot_dim: str = None
    ):
        """Visualize the use stock. If subplot_dim is not None, a separate plot for each item in the given dimension is created. Otherwise, one accumulated plot is generated."""
        per_capita = self.cfg.use_stock["per_capita"]

        population = mfa.parameters["population"]
        x_array = None
        linecolor_dim = "Region"

        pc_str = " pC" if per_capita else ""
        x_label = "Year"
        y_label = f"Stock{pc_str} [t]"
        title = f"Stocks{pc_str}"
        if self.cfg.use_stock["over_gdp"]:
            title = title + f" over GDP{pc_str}"
            x_label = f"GDP/PPP{pc_str} [2005 USD]"
            x_array = mfa.parameters["gdppc"]
            if not per_capita:
                # get global GDP per capita
                x_array = x_array * population

        if subplot_dim is None:
            # sum over all dimensions except time and linecolor_dim
            other_dimletters = tuple(
                letter
                for letter in stock.dims.letters
                if letter
                not in [
                    "t",
                    "r",
                ]
            )
            for dimletter in other_dimletters:
                stock = stock.sum_over(dimletter)

        if per_capita:
            stock = stock / population

        fig, ap_scatter_stock = self.plot_history_and_future(
            mfa=mfa,
            data_to_plot=stock,
            subplot_dim=subplot_dim,
            x_array=x_array,
            linecolor_dim=linecolor_dim,
            x_label=x_label,
            y_label=y_label,
            title=title,
        )

        # Adjust x-axis
        if self.cfg.use_stock["over_gdp"]:
            if self.cfg.plotting_engine == "plotly":
                fig.update_xaxes(type="log", range=[3, 5])
            elif self.cfg.plotting_engine == "pyplot":
                for ax in fig.get_axes():
                    ax.set_xscale("log")
                    ax.set_xlim(1e3, 1e5)

        self.plot_and_save_figure(
            ap_scatter_stock,
            f"stocks_global_by_region{'_per_capita' if per_capita else ''}.png",
            do_plot=False,
        )

    def plot_history_and_future(
        self,
        mfa: fd.MFASystem,
        data_to_plot: fd.FlodymArray,
        subplot_dim: Optional[str] = None,
        x_array: Optional[fd.FlodymArray] = None,
        linecolor_dim: Optional[str] = None,
        x_label: Optional[str] = None,
        y_label: Optional[str] = None,
        title: Optional[str] = None,
        **kwargs,
    ):

        colors = plc.qualitative.Dark24
        if linecolor_dim:
            dimletter = next(
                dimlist.letter for dimlist in mfa.dims.dim_list if dimlist.name == linecolor_dim
            )
            colors = (
                colors[: data_to_plot.dims[dimletter].len]
                + colors[: data_to_plot.dims[dimletter].len]
                + ["black" for _ in range(data_to_plot.dims[dimletter].len)]
            )
        else:
            colors = colors[:1] + colors[:1] + ["black"]

        # data preparation
        hist = data_to_plot[{"t": mfa.dims["h"]}]
        last_year_dim = fd.Dimension(
            name="Last Historic Year", letter="l", items=[mfa.dims["h"].items[-1]]
        )
        scatter = hist[{"h": last_year_dim}]
        if x_array is None:
            hist_x_array = None
            scatter_x_array = None
        else:
            hist_x_array = x_array[{"t": mfa.dims["h"]}]
            scatter_x_array = hist_x_array[{"h": last_year_dim}]

        # Future stock (dotted)
        ap = self.plotter_class(
            array=data_to_plot,
            intra_line_dim="Time",
            linecolor_dim=linecolor_dim,
            subplot_dim=subplot_dim,
            x_array=x_array,
            title=title,
            color_map=colors,
            line_type="dot",
            suppress_legend=True,
            **kwargs,
        )
        fig = ap.plot()

        # Historic stock (solid)
        ap_hist = self.plotter_class(
            array=hist,
            intra_line_dim="Historic Time",
            linecolor_dim=linecolor_dim,
            subplot_dim=subplot_dim,
            x_array=hist_x_array,
            fig=fig,
            color_map=colors,
            **kwargs,
        )
        fig = ap_hist.plot()

        # Last historic year (black dot)
        ap_scatter = self.plotter_class(
            array=scatter,
            intra_line_dim="Last Historic Year",
            linecolor_dim=linecolor_dim,
            subplot_dim=subplot_dim,
            x_array=scatter_x_array,
            xlabel=x_label,
            ylabel=y_label,
            fig=fig,
            chart_type="scatter",
            color_map=colors,
            suppress_legend=True,
            **kwargs,
        )
        fig = ap_scatter.plot()

        return fig, ap_scatter
