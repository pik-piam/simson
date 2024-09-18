from sodym import MFASystem


class PlasticsMFASystem(MFASystem):

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

        # auxiliary arrays;
        # It is important to initialize them to define their dimensions. See the NamedDimArray documentation for details.
        # could also be single variables instead of dict, but this way the code looks more uniform
        aux = {
            'reclmech_loss':                self.get_new_array(dim_letters=('t','e','r','m')),
            'virgin_2_fabr_all_mat':        self.get_new_array(dim_letters=('t','e','r')),
            'virgin_material_shares':       self.get_new_array(dim_letters=('t','e','r','m')),
            'captured_2_virginccu_by_mat':  self.get_new_array(dim_letters=('t','e','r','m')),
            'ratio_nonc_to_c':              self.get_new_array(dim_letters=('m',)),
        }

        # Slicing on the left-hand side of the assignment (foo[...] = bar) is used to assign only the values of the flows, not the NamedDimArray object managing the dimensions.
        # This way, the dimensions of the right-hand side of the assignment can be automatically reduced and re-ordered to the dimensions of the left-hand side.
        # For further details on the syntax, see the NamedDimArray documentation.

        flw['fabrication => use'][...]           = stk['in_use'].inflow
        flw['use => eol'][...]                   = stk['in_use'].outflow

        flw['eol => reclmech'][...]              = flw['use => eol']               * prm['mechanical_recycling_rate']
        flw['reclmech => recl'][...]             = flw['eol => reclmech']          * prm['mechanical_recycling_yield']
        aux['reclmech_loss'][...]                = flw['eol => reclmech']          - flw['reclmech => recl']
        flw['reclmech => uncontrolled'][...]     = aux['reclmech_loss']            * prm['reclmech_loss_uncontrolled_rate']
        flw['reclmech => incineration'][...]     = aux['reclmech_loss']            - flw['reclmech => uncontrolled']

        flw['eol => reclchem'][...]              = flw['use => eol']               * prm['chemical_recycling_rate']
        flw['reclchem => recl'][...]             = flw['eol => reclchem']

        flw['eol => reclsolv'][...]              = flw['use => eol']               * prm['solvent_recycling_rate']
        flw['reclsolv => recl'][...]             = flw['eol => reclsolv']

        flw['eol => incineration'][...]          = flw['use => eol']               * prm['incineration_rate']
        flw['eol => uncontrolled'][...]          = flw['use => eol']               * prm['uncontrolled_losses_rate']

        flw['eol => landfill'][...]              = flw['use => eol']               - flw['eol => reclmech'] \
                                                                                   - flw['eol => reclchem'] \
                                                                                   - flw['eol => reclsolv'] \
                                                                                   - flw['eol => incineration'] \
                                                                                   - flw['eol => uncontrolled']

        flw['incineration => emission'][...]     = flw['eol => incineration']      + flw['reclmech => incineration']

        flw['emission => captured'][...]         = flw['incineration => emission'] * prm['emission_capture_rate']
        flw['emission => atmosphere'][...]       = flw['incineration => emission'] - flw['emission => captured']
        flw['captured => virginccu'][...]        = flw['emission => captured']

        flw['recl => fabrication'][...]          = flw['reclmech => recl']         + flw['reclchem => recl'] \
                                                                                   + flw['reclsolv => recl']
        flw['virgin => fabrication'][...]        = flw['fabrication => use']       - flw['recl => fabrication']

        flw['virgindaccu => virgin'][...]        = flw['virgin => fabrication']    * prm['daccu_production_rate']
        flw['virginbio => virgin'][...]          = flw['virgin => fabrication']    * prm['bio_production_rate']

        aux['virgin_2_fabr_all_mat'][...]        = flw['virgin => fabrication']
        aux['virgin_material_shares'][...]       = flw['virgin => fabrication']    / aux['virgin_2_fabr_all_mat']
        aux['captured_2_virginccu_by_mat'][...]  = flw['captured => virginccu']    * aux['virgin_material_shares']

        # The { ... } syntax is used to slice the NamedDimArray object to a subset of its dimensions. See the NamedDimArray documentation for details.
        flw['virginccu => virgin']['C']              = aux['captured_2_virginccu_by_mat']['C']
        aux['ratio_nonc_to_c'][...]                  = prm['carbon_content_materials']['Other Elements'] / prm['carbon_content_materials']['C']
        flw['virginccu => virgin']['Other Elements'] = flw['virginccu => virgin']['C']                   * aux['ratio_nonc_to_c']

        flw['virginfoss => virgin'][...]         = flw['virgin => fabrication']    - flw['virgindaccu => virgin'] \
                                                                                   - flw['virginbio => virgin'] \
                                                                                   - flw['virginccu => virgin']

        flw['sysenv => virginfoss'][...]         = flw['virginfoss => virgin']
        flw['atmosphere => virginbio'][...]      = flw['virginbio => virgin']
        flw['atmosphere => virgindaccu'][...]    = flw['virgindaccu => virgin']
        flw['sysenv => virginccu'][...]          = flw['virginccu => virgin']      - aux['captured_2_virginccu_by_mat']

        # non-C atmosphere & captured has no meaning & is equivalent to sysenv

        return


    def compute_other_stocks(self):

        stk = self.stocks
        flw = self.flows

        # in-use stock is already computed in compute_in_use_stock

        stk['landfill'].inflow[...] = flw['eol => landfill']
        stk['landfill'].compute()

        stk['uncontrolled'].inflow[...] = flw['eol => uncontrolled'] + flw['reclmech => uncontrolled']
        stk['uncontrolled'].compute()

        stk['atmospheric'].inflow[...] = flw['emission => atmosphere']
        stk['atmospheric'].outflow[...] = flw['atmosphere => virgindaccu'] + flw['atmosphere => virginbio']
        stk['atmospheric'].compute()
        return