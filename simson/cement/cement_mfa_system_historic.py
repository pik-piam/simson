import flodym as fd


class InflowDrivenHistoricCementMFASystem(fd.MFASystem):

    def compute(self):
        """
        Perform all computations for the MFA system.
        """
        self.compute_in_use_stock()
        self.compute_flows()
        self.check_mass_balance()

    def compute_in_use_stock(self):
        prm = self.parameters
        stk = self.stocks

        # in use
        stk["historic_in_use"].inflow[...] = (
            prm["cement_production"] * prm["use_split"] / prm["cement_ratio"]
        )
        stk["historic_in_use"].lifetime_model.set_prms(
            mean=prm["use_lifetime_mean"],
            std=prm["use_lifetime_std"],
        )
        stk["historic_in_use"].compute()

    def compute_flows(self):
        flw = self.flows
        stk = self.stocks

        flw["sysenv => use"][...] = stk["historic_in_use"].inflow
        flw["use => sysenv"][...] = stk["historic_in_use"].outflow
