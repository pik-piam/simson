import numpy as np
from numpy.linalg import inv

from sodym import MFASystem

from simson.common.inflow_driven_mfa import InflowDrivenHistoricMFA

class InflowDrivenHistoricSteelMFASystem(InflowDrivenHistoricMFA):

    def compute(self):
        """
        Perform all computations for the MFA system.
        """
        self.compute_historic_flows()
        self.compute_historic_in_use_stock()
        self.check_mass_balance()


    def compute_historic_flows(self):
        prm = self.parameters
        flw = self.flows

        aux = {
            'net_intermediate_trade': self.get_new_array(dim_letters=('h','r','i')),
            'fabrication_by_sector': self.get_new_array(dim_letters=('h','r','g')),
            'fabrication_loss': self.get_new_array(dim_letters=('h','r','g')),
            'fabrication_error': self.get_new_array(dim_letters=('h','r'))
        }

        flw['sysenv => forming'][...] = prm['production_by_intermediate']
        flw['forming => ip_market'][...] = prm['production_by_intermediate'] * prm['forming_yield']
        flw['forming => sysenv'][...] = flw['sysenv => forming'] - flw['forming => ip_market']

        # Todo: add trade

        aux['net_intermediate_trade'][...] = flw['ip_trade => ip_market'] - flw['ip_market => ip_trade']
        flw['ip_market => fabrication'][...] = flw['forming => ip_market'] + aux['net_intermediate_trade']

        aux['fabrication_by_sector'][...] = self._calc_sector_flows(
            flw['ip_market => fabrication'],
            prm['good_to_intermediate_distribution'])

        aux['fabrication_error'] = flw['ip_market => fabrication'] - aux['fabrication_by_sector']

        flw['fabrication => use'][...] = aux['fabrication_by_sector'] * prm['fabrication_yield']
        aux['fabrication_loss'][...] = aux['fabrication_by_sector'] - flw['fabrication => use']
        flw['fabrication => sysenv'][...] = aux['fabrication_error'] + aux['fabrication_loss']
        # Todo: add intermediate trade

        return

    def _calc_sector_flows(self, intermediate_flow, gi_distribution):
        """
        Estimate the fabrication by in-use-good according to the inflow of intermediate products
        and the good to intermediate product distribution.
        """

        # The following calculation is based on
        # https://en.wikipedia.org/wiki/Overdetermined_system#Approximate_solutions
        # gi_values represents 'A', hence the variable at_a is A transposed times A
        # 'b' is the intermediate flow and x are the sector flows that we are trying to find out

        gi_values = gi_distribution.values.transpose()
        at_a = np.matmul(gi_values.transpose(), gi_values)
        inverse_at_a = inv(at_a)
        inverse_at_a_times_at = np.matmul(inverse_at_a, gi_values.transpose())
        sector_flow_values = np.einsum('gi,hri->hrg',inverse_at_a_times_at, intermediate_flow.values)

        # don't allow negative sector flows
        sector_flow_values = np.maximum(0, sector_flow_values)

        sector_flows = self.get_new_array(dim_letters=('h','r','g'))
        sector_flows.values = sector_flow_values

        return sector_flows

    def compute_historic_in_use_stock(self):
        flw = self.flows
        stk = self.stocks
        stk['in_use'].inflow[...] = flw['fabrication => use'] + flw['indirect_trade => use'] - flw['use => indirect_trade']
        stk['in_use'].compute()

        return

class StockDrivenSteelMFASystem(MFASystem):

    def compute(self):
        """
        Perform all computations for the MFA system.
        """
        self.compute_flows()
        self.compute_other_stocks()
        self.check_mass_balance()

    def compute_flows(self):
        # abbreviations for better readability
        prm = self.parameters
        flw = self.flows
        stk = self.stocks
        scp = self.scalar_parameters


        # auxiliary arrays;
        # It is important to initialize them to define their dimensions. See the NamedDimArray documentation for details.
        # could also be single variables instead of dict, but this way the code looks more uniform
        aux = {
            'total_fabrication': self.get_new_array(dim_letters=('t', 'e', 'r', 'g', 's')),
            'production': self.get_new_array(dim_letters=('t', 'e', 'r', 'i', 's')),
            'forming_outflow': self.get_new_array(dim_letters=('t', 'e', 'r', 's')),
            'scrap_in_production': self.get_new_array(dim_letters=('t', 'e', 'r', 's')),
            'available_scrap': self.get_new_array(dim_letters=('t', 'e', 'r', 's')),
            'eaf_share_production': self.get_new_array(dim_letters=('t', 'e', 'r', 's')),
            'production_inflow': self.get_new_array(dim_letters=('t', 'e', 'r', 's')),
            'max_scrap_production': self.get_new_array(dim_letters=('t', 'e', 'r', 's')),
            'scrap_share_production': self.get_new_array(dim_letters=('t', 'e', 'r', 's')),
            'bof_production_inflow': self.get_new_array(dim_letters=('t', 'e', 'r', 's')),
        }

        # Slicing on the left-hand side of the assignment (foo[...] = bar) is used to assign only the values of the flows, not the NamedDimArray object managing the dimensions.
        # This way, the dimensions of the right-hand side of the assignment can be automatically reduced and re-ordered to the dimensions of the left-hand side.
        # For further details on the syntax, see the NamedDimArray documentation.

        # Pre-use

        flw['fabrication => use'][...]                  = stk['use'].inflow
        aux['total_fabrication'][...]                   = flw['fabrication => use']             /   prm['fabrication_yield']
        flw['fabrication => fabrication_buffer'][...]   = aux['total_fabrication']              -   flw['fabrication => use']
        flw['ip_market => fabrication'][...]            = aux['total_fabrication']              *   prm['good_to_intermediate_distribution']
        flw['forming => ip_market'][...]                = flw['ip_market => fabrication']
        aux['production'][...]                          = flw['forming => ip_market']           /   prm['forming_yield']
        aux['forming_outflow'][...]                     = aux['production']                     -   flw['forming => ip_market']
        flw['forming => sysenv'][...]                   = aux['forming_outflow']                *   scp['forming_losses']
        flw['forming => fabrication_buffer'][...]       = aux['forming_outflow']                -   flw['forming => sysenv']

        # Post-use

        flw['use => outflow_buffer'][...]               = stk['use'].outflow
        flw['outflow_buffer => eol_market'][...]        = flw['use => outflow_buffer']          *   prm['recovery_rate']
        flw['outflow_buffer => obsolete'][...]          = flw['use => outflow_buffer']          -   flw['outflow_buffer => eol_market']
        flw['eol_market => recycling'][...]             = flw['outflow_buffer => eol_market']
        flw['recycling => scrap_market'][...]           = flw['eol_market => recycling']
        flw['fabrication_buffer => scrap_market'][...]  = flw['forming => fabrication_buffer']  +   flw['fabrication => fabrication_buffer']


        # PRODUCTION

        aux['production_inflow'][...]                   = aux['production']                     /   scp['production_yield']
        aux['max_scrap_production'][...]                = aux['production_inflow']              *   scp['max_scrap_share_base_model']
        aux['available_scrap'][...]                     = flw['recycling => scrap_market']      +   flw['fabrication_buffer => scrap_market']
        aux['scrap_in_production'][...]                 = aux['available_scrap'].minimum(aux['max_scrap_production'])  # using NumPy Minimum functionality
        flw['scrap_market => excess_scrap'][...]        = aux['available_scrap']                -   aux['scrap_in_production']
        aux['scrap_share_production']['Fe'][...]        = aux['scrap_in_production']['Fe']      /   aux['production_inflow']['Fe']
        aux['eaf_share_production'][...]                = aux['scrap_share_production']         -   scp['scrap_in_bof_rate']
        aux['eaf_share_production'][...]                = aux['eaf_share_production']           /   (1 - scp['scrap_in_bof_rate'])
        aux['eaf_share_production'][...]                = aux['eaf_share_production'].minimum(1).maximum(0)
        flw['scrap_market => eaf_production'][...]      = aux['production_inflow']              *   aux['eaf_share_production']
        flw['scrap_market => bof_production'][...]      = aux['scrap_in_production']            -   flw['scrap_market => eaf_production']
        aux['bof_production_inflow'][...]               = aux['production_inflow']              -   flw['scrap_market => eaf_production']
        flw['sysenv => bof_production'][...]            = aux['bof_production_inflow']          -   flw['scrap_market => bof_production']
        flw['bof_production => forming'][...]           = aux['bof_production_inflow']          *   scp['production_yield']
        flw['bof_production => sysenv'][...]            = aux['bof_production_inflow']          -   flw['bof_production => forming']
        flw['eaf_production => forming'][...]           = flw['scrap_market => eaf_production'] *   scp['production_yield']
        flw['eaf_production => sysenv'][...]            = flw['scrap_market => eaf_production'] -   flw['eaf_production => forming']

        return

    def compute_other_stocks(self):
        stk = self.stocks
        flw = self.flows

        # in-use stock is already computed in compute_in_use_stock

        stk['obsolete'].inflow[...] = flw['outflow_buffer => obsolete']
        stk['obsolete'].compute()

        stk['excess_scrap'].inflow[...] = flw['scrap_market => excess_scrap']
        stk['excess_scrap'].compute()

        # TODO: Delay buffers?

        stk['outflow_buffer'].inflow[...] = flw['use => outflow_buffer']
        stk['outflow_buffer'].outflow[...] = flw['outflow_buffer => eol_market'] + flw['outflow_buffer => obsolete']
        stk['outflow_buffer'].compute()

        stk['fabrication_buffer'].inflow[...] = flw['forming => fabrication_buffer'] + flw['fabrication => fabrication_buffer']
        stk['fabrication_buffer'].outflow[...] = flw['fabrication_buffer => scrap_market']
        stk['fabrication_buffer'].compute()
