import os
from matplotlib import pyplot as plt
from simson.common.base_model import SimsonBaseModel
import plotly.graph_objects as go
import flodym as fd
import flodym.export as fde


class CustomDataExporter(SimsonBaseModel):
    output_path: str
    do_save_figs: bool = True
    do_show_figs: bool = True
    plotting_engine: str = "plotly"
    _display_names: dict = {}
    do_export: dict = {"pickle": True, "csv": True}
    production: dict = {"do_visualize": True}
    stock: dict = {"do_visualize": True}
    sankey: dict = {"do_visualize": True}

    def export_mfa(self, mfa: fd.MFASystem):
        if self.do_export["pickle"]:
            fde.export_mfa_to_pickle(mfa=mfa, export_path=self.export_path("mfa.pickle"))
        if self.do_export["csv"]:
            dir_out = os.path.join(self.export_path(), "flows")
            fde.export_mfa_flows_to_csv(mfa=mfa, export_directory=dir_out)
            fde.export_mfa_stocks_to_csv(mfa=mfa, export_directory=dir_out)

    def export_path(self, filename: str = None):
        path_tuple = (self.output_path, "export")
        if filename is not None:
            path_tuple += (filename,)
        return os.path.join(*path_tuple)

    def figure_path(self, filename: str):
        return os.path.join(self.output_path, "figures", filename)

    def _show_and_save_plotly(self, fig: go.Figure, name):
        if self.do_save_figs:
            fig.write_image(self.figure_path(f"{name}.png"))
        if self.do_show_figs:
            fig.show()

    def visualize_sankey(self, mfa: fd.MFASystem):
        plotter = fde.PlotlySankeyPlotter(mfa=mfa, display_names=self._display_names, **self.sankey)
        fig = plotter.plot()
        self._show_and_save_plotly(fig, name="sankey")

    def figure_path(self, filename: str) -> str:
        return os.path.join(self.output_path, "figures", filename)

    def plot_and_save_figure(self, plotter: fde.ArrayPlotter, filename: str):
        plotter.plot(do_show=self.do_show_figs)
        if self.do_save_figs:
            plotter.save(self.figure_path(filename), width=2200, height=1300)

    def stop_and_show(self):
        if self.plotting_engine == "pyplot" and self.do_show_figs:
            plt.show()

    @property
    def plotter_class(self):
        if self.plotting_engine == "plotly":
            return fde.PlotlyArrayPlotter
        elif self.plotting_engine == "pyplot":
            return fde.PyplotArrayPlotter
        else:
            raise ValueError(f"Unknown plotting engine: {self.plotting_engine}")
