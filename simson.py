import logging
import yaml

from sodym import MFASystem

from simson.plastics.plastics_model import PlasticsModel
from simson.common.common_cfg import CommonCfg
from simson.steel.steel_model import SteelModel


allowed_models = {
    'plastics': PlasticsModel,
    'steel': SteelModel,
}
configurations = {
    'plastics': CommonCfg,
    'steel': CommonCfg,
}


def get_model_config(filename):
    with open(filename, 'r') as stream:
        data = yaml.safe_load(stream)
    return {k: v for k, v in data.items()}


def init_mfa(cfg: dict) -> MFASystem:
    """Choose MFA subclass and return an initialized instance.
    """
    model_name = cfg['model_class']
    if model_name not in allowed_models:
        raise ValueError(f"Model class {model_name} not supported.")

    cfg = configurations[model_name](**cfg)
    mfa = allowed_models[model_name](cfg=cfg)
    return mfa


if __name__ == '__main__':
    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    cfg_file = 'config/steel.yml'
    model_config = get_model_config(cfg_file)
    mfa = init_mfa(cfg=model_config)
    logging.info(f'{type(mfa).__name__} instance created.')
    mfa.run()
    logging.info('Model computations completed.')
