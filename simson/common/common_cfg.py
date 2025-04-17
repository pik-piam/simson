from simson.common.base_model import SimsonBaseModel
import flodym as fd

from .data_extrapolations import Extrapolation


IMPLEMENTED_MODELS = [
    "plastics",
    "steel",
    "cement",
]


def choose_subclass_by_name(name: str, parent: type) -> type:

    def recurse_subclasses(cls):
        return set(cls.__subclasses__()).union(
            [s for c in cls.__subclasses__() for s in recurse_subclasses(c)]
        )

    subclasses = {cls.__name__: cls for cls in recurse_subclasses(parent)}
    if name not in subclasses:
        raise ValueError(
            f"Subclass name for {parent.__name__} must be one of {list(subclasses.keys())}, but {name} was given."
        )
    return subclasses[name]


class ModelCustomization(SimsonBaseModel):

    stock_extrapolation_class_name: str
    lifetime_model_name: str
    do_stock_extrapolation_by_category: bool = False

    @property
    def lifetime_model(self) -> fd.LifetimeModel:
        return choose_subclass_by_name(self.lifetime_model_name, fd.LifetimeModel)

    @property
    def stock_extrapolation_class(self) -> Extrapolation:
        """Check if the given extrapolation class is a valid subclass of OneDimensionalExtrapolation and return it."""
        return choose_subclass_by_name(self.stock_extrapolation_class_name, Extrapolation)


class ExportCfg(SimsonBaseModel):
    csv: bool = True
    pickle: bool = True
    assumptions: bool = True


class VisualizationCfg(SimsonBaseModel):

    use_stock: dict = {"do_visualize": False}
    production: dict = {"do_visualize": False}
    sankey: dict = {"do_visualize": False}
    do_show_figs: bool = True
    do_save_figs: bool = False
    plotting_engine: str = "plotly"
    plotly_renderer: str = "browser"


class CementVisualizationCfg(VisualizationCfg):

    clinker_production: dict = {}
    cement_production: dict = {}
    concrete_production: dict = {}
    eol_stock: dict = {}
    extrapolation: dict = {}


class SteelVisualizationCfg(VisualizationCfg):

    scrap_demand_supply: dict = {"do_visualize": False}
    sector_splits: dict = {"do_visualize": False}
    trade: dict = {"do_visualize": False}
    consumption: dict = {"do_visualize": False}
    gdppc: dict = {"do_visualize": False}


class PlasticsVisualizationCfg(VisualizationCfg):

    pass


class GeneralCfg(SimsonBaseModel):

    model_class: str
    input_data_path: str
    customization: ModelCustomization
    visualization: VisualizationCfg
    output_path: str
    do_export: ExportCfg

    @classmethod
    def from_model_class(cls, **kwargs) -> "GeneralCfg":
        if "model_class" not in kwargs:
            raise ValueError("model_class must be provided.")
        model_class = kwargs["model_class"]
        subclasses = {
            "plastics": PlasticsCfg,
            "steel": SteelCfg,
            "cement": CementCfg,
        }
        if model_class not in subclasses:
            raise ValueError(f"Model class {model_class} not supported.")
        subcls = subclasses[model_class]
        return subcls(**kwargs)


class PlasticsCfg(GeneralCfg):

    visualization: PlasticsVisualizationCfg


class CementCfg(GeneralCfg):

    visualization: CementVisualizationCfg


class SteelCfg(GeneralCfg):

    visualization: SteelVisualizationCfg
