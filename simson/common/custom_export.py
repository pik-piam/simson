import logging
import os
from matplotlib import pyplot as plt
from pydantic import BaseModel as PydanticBaseModel
import plotly.graph_objects as go

from sodym import MFASystem
from sodym.export.data_writer import export_mfa_flows_to_csv, export_mfa_stocks_to_csv, export_mfa_to_pickle
from sodym.export.array_plotter import PlotlyArrayPlotter
from sodym.export.sankey import PlotlySankeyPlotter


class CustomDataExporter(PydanticBaseModel):

    output_path: str
    do_save_figs: bool = True
    do_show_figs: bool = True
    display_names: dict = {}
    do_export: dict = {'pickle': True, 'csv': True}
    production: dict = {'do_visualize': True}
    stock: dict = {'do_visualize': True}
    sankey: dict = {'do_visualize': True}

    def export_mfa(self, mfa: MFASystem):
        if self.do_export['pickle']:
            export_mfa_to_pickle(mfa=mfa, export_path=self.export_path('mfa.pickle'))
        if self.do_export['csv']:
            dir_out = os.path.join(self.export_path(), 'flows')
            export_mfa_flows_to_csv(mfa=mfa, export_directory=dir_out)
            export_mfa_stocks_to_csv(mfa=mfa, export_directory=dir_out)

    def export_path(self, filename: str = None):
        path_tuple = (self.output_path, 'export')
        if filename is not None:
            path_tuple += (filename,)
        return os.path.join(*path_tuple)

    def figure_path(self, filename: str):
        return os.path.join(self.output_path, 'figures', filename)

    def _show_and_save_pyplot(self, fig, name):
        if self.do_save_figs:
            plt.savefig(self.figure_path(f"{name}.png"))
        if self.do_show_figs:
            plt.show()

    def _show_and_save_plotly(self, fig: go.Figure, name):
        if self.do_save_figs:
            fig.write_image(self.figure_path(f"{name}.png"))
        if self.do_show_figs:
            fig.show()

    def visualize_results(self, mfa: MFASystem):
        if self.production['do_visualize']:
            self.visualize_production(mfa=mfa)
        if self.stock['do_visualize']:
            logging.info('Stock visualization functionality unavailable')
            #self.visualize_stock()
        if self.sankey['do_visualize']:
            plotter = PlotlySankeyPlotter(
                mfa=mfa,
                display_names = self.display_names,
                **self.sankey)
            fig = plotter.plot()
            self._show_and_save_plotly(fig, name="sankey")

    def visualize_production(self, mfa: MFASystem):
        ap_modeled = PlotlyArrayPlotter(
            array=mfa.stocks['in_use'].inflow['World'].sum_nda_over(('m', 'e')),
            intra_line_dim='Time',
            subplot_dim='Good',
            line_label='Modeled',
            display_names=self.display_names
        )
        fig = ap_modeled.plot()
        ap_historic = PlotlyArrayPlotter(
            array = mfa.parameters['production']['World'],
            intra_line_dim='Historic Time',
            subplot_dim='Good',
            line_label='Historic Production',
            fig=fig,
            xlabel='Year',
            ylabel='Production [t]',
            display_names=self.display_names
        )
        save_path = os.path.join(self.output_path, 'figures', 'production.png') if self.do_save_figs else None
        ap_historic.plot(save_path=save_path, do_show=self.do_show_figs)

    def visualize_stock(self, gdppc, historic_gdppc, stocks, historic_stocks, stocks_pc, historic_stocks_pc):

        if self.stock['per_capita']:
            stocks_plot = stocks_pc['World']
            historic_stocks_plot = historic_stocks_pc['World']
        else:
            stocks_plot = stocks['World']
            historic_stocks_plot = historic_stocks['World']

        if self.stock['over'] == 'time':
            x_array, x_array_hist = None, None
            xlabel = 'Year'
        elif self.stock['over'] == 'gdppc':
            x_array = gdppc['World']
            x_array_hist = historic_gdppc['World']
            xlabel = 'GDP per capita [USD]'

        ap_modeled = PlotlyArrayPlotter(
            array=stocks_plot,
            intra_line_dim='Time',
            x_array=x_array,
            subplot_dim='Good',
            line_label='Modeled',
            display_names=self.display_names)
        fig = ap_modeled.plot()
        ap_historic = PlotlyArrayPlotter(
            array=historic_stocks_plot,
            intra_line_dim='Historic Time',
            x_array=x_array_hist,
            subplot_dim='Good',
            line_label='Historic',
            fig=fig,
            xlabel=xlabel,
            ylabel='Stock [t]',
            display_names=self.display_names)
        save_path = os.path.join(self.output_path, 'figures', 'stock.png') if self.do_save_figs else None
        ap_historic.plot(save_path=save_path, do_show=self.do_show_figs)
