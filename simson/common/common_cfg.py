from simson.common.base_model import SimsonBaseModel
import flodym as fd


IMPLEMENTED_MODELS = [
    "plastics",
    "steel",
]


class ModelCustomization(SimsonBaseModel):

    curve_strategy: str
    ldf_type: str
    _lifetime_model_class: type = None

    @property
    def lifetime_model(self):
        lifetime_model_classes = {
            "Fixed": fd.FixedLifetime,
            "Normal": fd.NormalLifetime,
            "FoldedNormal": fd.FoldedNormalLifetime,
            "LogNormal": fd.LogNormalLifetime,
            "Weibull": fd.WeibullLifetime,
        }
        return lifetime_model_classes[self.ldf_type]


class VisualizationCfg(SimsonBaseModel):

    stock: dict = {"do_visualize": False}
    production: dict = {"do_visualize": False}
    sankey: dict = {"do_visualize": False}
    do_show_figs: bool = True
    do_save_figs: bool = False
    plotting_engine: str = "plotly"


class SteelVisualizationCfg(VisualizationCfg):

    scrap_demand_supply: dict = {"do_visualize": False}
    sector_splits: dict = {"do_visualize": False}


class PlasticsVisualizationCfg(VisualizationCfg):

    pass


class GeneralCfg(SimsonBaseModel):

    model_class: str
    input_data_path: str
    customization: ModelCustomization
    visualization: VisualizationCfg
    output_path: str
    do_export: dict[str, bool]

    @classmethod
    def from_model_class(cls, **kwargs) -> "GeneralCfg":
        if "model_class" not in kwargs:
            raise ValueError("model_class must be provided.")
        model_class = kwargs["model_class"]
        subclasses = {
            "plastics": PlasticsCfg,
            "steel": SteelCfg,
        }
        if model_class not in subclasses:
            raise ValueError(f"Model class {model_class} not supported.")
        subcls = subclasses[model_class]
        return subcls(**kwargs)


class PlasticsCfg(GeneralCfg):

    visualization: PlasticsVisualizationCfg


class SteelCfg(GeneralCfg):

    visualization: SteelVisualizationCfg
