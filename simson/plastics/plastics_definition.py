import flodym as fd

from simson.common.common_cfg import GeneralCfg


def get_definition(cfg: GeneralCfg):

    dimensions = [
        fd.DimensionDefinition(name="Time", dim_letter="t", dtype=int),
        fd.DimensionDefinition(name="Historic Time", dim_letter="h", dtype=int),
        fd.DimensionDefinition(name="Element", dim_letter="e", dtype=str),
        fd.DimensionDefinition(name="Region", dim_letter="r", dtype=str),
        fd.DimensionDefinition(name="Material", dim_letter="m", dtype=str),
        fd.DimensionDefinition(name="Good", dim_letter="g", dtype=str),
    ]

    processes = [
        "sysenv",
        "virginfoss",
        "virginbio",
        "virgindaccu",
        "virginccu",
        "virgin",
        "fabrication",
        "wasteimport",
        "wasteexport",
        "wastetrade",
        # "finalimport",
        # "finalexport",
        # "finaltrade",
        "recl",
        "reclmech",
        "reclchem",
        "use",
        "eol",
        "incineration",
        "landfill",
        "collected",
        "mismanaged",
        "uncontrolled",
        "emission",
        "captured",
        "atmosphere",
    ]

    # fmt: off
    # names are auto-generated, see Flow class documetation
    flows = [
        fd.FlowDefinition(from_process="sysenv", to_process="virginfoss", dim_letters=("t","e","r","m")),
        fd.FlowDefinition(from_process="sysenv", to_process="virginbio", dim_letters=("t","e","r","m")),
        fd.FlowDefinition(from_process="sysenv", to_process="virgindaccu", dim_letters=("t","e","r","m")),
        fd.FlowDefinition(from_process="sysenv", to_process="virginccu", dim_letters=("t","e","r","m")),
        fd.FlowDefinition(from_process="atmosphere", to_process="virginbio", dim_letters=("t","e","r")),
        fd.FlowDefinition(from_process="atmosphere", to_process="virgindaccu", dim_letters=("t","e","r")),
        fd.FlowDefinition(from_process="virginfoss", to_process="virgin", dim_letters=("t","e","r","m")),
        fd.FlowDefinition(from_process="virginbio", to_process="virgin", dim_letters=("t","e","r","m")),
        fd.FlowDefinition(from_process="virgindaccu", to_process="virgin", dim_letters=("t","e","r","m")),
        fd.FlowDefinition(from_process="virginccu", to_process="virgin", dim_letters=("t","e","r","m")),
        fd.FlowDefinition(from_process="virgin", to_process="fabrication", dim_letters=("t","e","r","m")),
        fd.FlowDefinition(from_process="fabrication", to_process="use", dim_letters=("t","e","r","m","g")),
        fd.FlowDefinition(from_process="use", to_process="eol", dim_letters=("t","e","r","m","g")),
        fd.FlowDefinition(from_process="eol", to_process="collected", dim_letters=("t","e","r","m")),
        fd.FlowDefinition(from_process="eol", to_process="mismanaged", dim_letters=("t","e","r","m")),
        fd.FlowDefinition(from_process="collected", to_process="reclmech", dim_letters=("t","e","r","m")),
        fd.FlowDefinition(from_process="collected", to_process="reclchem", dim_letters=("t","e","r","m")),
        fd.FlowDefinition(from_process="collected", to_process="landfill", dim_letters=("t","e","r","m")),
        fd.FlowDefinition(from_process="collected", to_process="incineration", dim_letters=("t","e","r","m")),
        fd.FlowDefinition(from_process="mismanaged", to_process="uncontrolled", dim_letters=("t","e","r","m")),
        fd.FlowDefinition(from_process="reclmech", to_process="recl", dim_letters=("t","e","r","m")),
        fd.FlowDefinition(from_process="reclchem", to_process="recl", dim_letters=("t","e","r","m")),
        fd.FlowDefinition(from_process="recl", to_process="fabrication", dim_letters=("t","e","r","m")),
        fd.FlowDefinition(from_process="reclmech", to_process="uncontrolled", dim_letters=("t","e","r","m")),
        fd.FlowDefinition(from_process="reclmech", to_process="incineration", dim_letters=("t","e","r","m")),
        fd.FlowDefinition(from_process="incineration", to_process="emission", dim_letters=("t","e","r")),
        fd.FlowDefinition(from_process="emission", to_process="captured", dim_letters=("t","e","r")),
        fd.FlowDefinition(from_process="emission", to_process="atmosphere", dim_letters=("t","e","r")),
        fd.FlowDefinition(from_process="captured", to_process="virginccu", dim_letters=("t","e","r")),

        # waste trade
        fd.FlowDefinition(from_process="wastetrade", to_process="wasteimport", dim_letters=("t","e","r","m")),
        fd.FlowDefinition(from_process="wasteimport", to_process="collected", dim_letters=("t","e","r","m")),
        fd.FlowDefinition(from_process="collected", to_process="wasteexport", dim_letters=("t","e","r","m")),
        fd.FlowDefinition(from_process="wasteexport", to_process="wastetrade", dim_letters=("t","e","r","m")),

        # final trade
        #fd.FlowDefinition(from_process="finaltrade", to_process="finalimport", dim_letters=("t","e","r","m")),
        #fd.FlowDefinition(from_process="finalimport", to_process="fabrication", dim_letters=("t","e","r","m")),
        #fd.FlowDefinition(from_process="fabrication", to_process="finalexport", dim_letters=("t","e","r","m")),
        #fd.FlowDefinition(from_process="finalexport", to_process="finaltrade", dim_letters=("t","e","r","m")),
    ]
    # fmt: on

    stocks = [
        fd.StockDefinition(
            name="in_use_historic",
            dim_letters=("h", "r", "g"),
            subclass=fd.InflowDrivenDSM,
            lifetime_model_class=cfg.customization.lifetime_model,
            time_letter="h",
        ),
        fd.StockDefinition(
            name="in_use_dsm",
            dim_letters=("t", "r", "g"),
            subclass=fd.StockDrivenDSM,
            lifetime_model_class=cfg.customization.lifetime_model,
        ),
        fd.StockDefinition(
            name="in_use",
            process="use",
            dim_letters=("t", "e", "r", "m", "g"),
            subclass=fd.SimpleFlowDrivenStock,
        ),
        fd.StockDefinition(
            name="wastetrade",
            process="wastetrade",
            dim_letters=("t", "e", "m"),
            subclass=fd.SimpleFlowDrivenStock,
        ),
        # fd.StockDefinition(
        #     name="finaltrade",
        #     process="finaltrade",
        #     dim_letters=("t", "e", "m"),
        #     subclass=fd.SimpleFlowDrivenStock,
        # ),
        fd.StockDefinition(
            name="atmospheric",
            process="atmosphere",
            dim_letters=("t", "e", "r"),
            subclass=fd.SimpleFlowDrivenStock,
        ),
        fd.StockDefinition(
            name="landfill",
            process="landfill",
            dim_letters=("t", "e", "r", "m"),
            subclass=fd.SimpleFlowDrivenStock,
        ),
        fd.StockDefinition(
            name="uncontrolled",
            process="uncontrolled",
            dim_letters=("t", "e", "r", "m"),
            subclass=fd.SimpleFlowDrivenStock,
        ),
    ]

    parameters = [
        # EOL rates
        fd.ParameterDefinition(name="collection_rate", dim_letters=("t", "r", "m")),
        fd.ParameterDefinition(name="mechanical_recycling_rate", dim_letters=("t", "r", "m")),
        fd.ParameterDefinition(name="chemical_recycling_rate", dim_letters=("t", "r", "m")),
        fd.ParameterDefinition(name="solvent_recycling_rate", dim_letters=("t", "r", "m")),
        fd.ParameterDefinition(name="incineration_rate", dim_letters=("t", "r", "m")),
        fd.ParameterDefinition(name="landfill_rate", dim_letters=("t", "r", "m")),
        fd.ParameterDefinition(name="wasteimport_rate", dim_letters=("t", "r", "g")),
        fd.ParameterDefinition(name="wasteexport_rate", dim_letters=("t", "r", "g")),
        # fd.ParameterDefinition(name="finalimport_rate", dim_letters=("t","r")),
        # fd.ParameterDefinition(name="finalexport_rate", dim_letters=("t","r")),
        # virgin production rates
        fd.ParameterDefinition(name="bio_production_rate", dim_letters=("t", "r", "m")),
        fd.ParameterDefinition(name="daccu_production_rate", dim_letters=("t", "r", "m")),
        # recycling losses
        fd.ParameterDefinition(name="mechanical_recycling_yield", dim_letters=("t", "r", "m")),
        fd.ParameterDefinition(name="reclmech_loss_uncontrolled_rate", dim_letters=("t", "r", "m")),
        # other
        fd.ParameterDefinition(name="material_shares_in_goods", dim_letters=("r", "m", "g")),
        fd.ParameterDefinition(name="emission_capture_rate", dim_letters=("t",)),
        fd.ParameterDefinition(name="carbon_content_materials", dim_letters=("e", "m")),
        # for in-use stock
        fd.ParameterDefinition(name="wasteimporttotal", dim_letters=("t", "g")),
        fd.ParameterDefinition(name="finalimporttotal", dim_letters=("t",)),
        fd.ParameterDefinition(name="production", dim_letters=("h", "r", "g")),
        fd.ParameterDefinition(name="lifetime_mean", dim_letters=("r", "g")),
        fd.ParameterDefinition(name="lifetime_std", dim_letters=("r", "g")),
        fd.ParameterDefinition(name="population", dim_letters=("t", "r")),
        fd.ParameterDefinition(name="gdppc", dim_letters=("t", "r")),
    ]

    return fd.MFADefinition(
        dimensions=dimensions,
        processes=processes,
        flows=flows,
        stocks=stocks,
        parameters=parameters,
    )
