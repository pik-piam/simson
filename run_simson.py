import logging
import yaml
import flodym as fd
import sys

from simson.common.common_cfg import GeneralCfg
from simson.plastics.plastics_model import PlasticsModel
from simson.steel.steel_model import SteelModel
from simson.cement.cement_model import CementModel


models = {
    "plastics": PlasticsModel,
    "steel": SteelModel,
    "cement": CementModel,
}


def get_model_config(filename):
    with open(filename, "r") as stream:
        data = yaml.safe_load(stream)
    return {k: v for k, v in data.items()}


def init_model(cfg: dict) -> fd.MFASystem:
    """Choose MFA subclass and return an initialized instance."""

    cfg = GeneralCfg.from_model_class(**cfg)
    mfa = models[cfg.model_class](cfg=cfg)
    return mfa


def calculate_model(model_config):
    mfa = init_model(cfg=model_config)
    logging.info(f"{type(mfa).__name__} instance created.")
    mfa.run()
    logging.info("Model computations completed.")


def run_simson(cfg_file: str):
    logging.basicConfig(
        format="%(asctime)s %(levelname)-8s %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    model_config = get_model_config(cfg_file)
    calculate_model(model_config)


if __name__ == "__main__":
    try:
        cfg_file = sys.argv[1]
    except IndexError:
        raise ValueError("Please provide a configuration file as an argument.")
    run_simson(cfg_file)
