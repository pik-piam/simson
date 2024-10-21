from pydantic import BaseModel as PydanticBaseModel


class ModelCustomization(PydanticBaseModel):
    curve_strategy: str
    ldf_type: str


class VisualizationCfg(PydanticBaseModel):
    stock: dict = {'do_visualize': False}
    production: dict = {'do_visualize': False}
    sankey: dict = {'do_visualize': False}
    do_show_figs: bool = True
    do_save_figs: bool = False


class CommonCfg(PydanticBaseModel):
    input_data_path: str
    customization: ModelCustomization
    visualization: VisualizationCfg
    output_path: str
    do_export: dict[str, bool]
