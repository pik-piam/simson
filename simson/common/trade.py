from pydantic import BaseModel as PydanticBaseModel, model_validator
from typing import List
import numpy as np
from scipy.stats import gmean, hmean
import sys
import flodym as fd


class Trade(PydanticBaseModel):
    """A TradeModule handles the storing and calculation of trade data for a given MFASystem."""

    imports: fd.FlodymArray
    exports: fd.FlodymArray

    @model_validator(mode="after")
    def validate_region_dimension(self):
        assert "r" in self.imports.dims.letters, "Imports must have a Region dimension."
        assert "r" in self.exports.dims.letters, "Exports must have a Region dimension."

        return self

    @model_validator(mode="after")
    def validate_trade_dimensions(self):
        assert (
            self.imports.dims.letters == self.exports.dims.letters
        ), "Imports and exports must have the same dimensions."
        return self

    @property
    def net_imports(self):
        return self.imports - self.exports

    @property
    def net_exports(self):
        return self.exports - self.imports

    def balance(self, to: str = "hmean"):
        global_imports = self.imports.sum_over("r")
        global_exports = self.exports.sum_over("r")

        reference_trade = self.get_reference_trade(global_imports, global_exports, to)

        import_factor = reference_trade / global_imports.maximum(sys.float_info.epsilon)
        export_factor = reference_trade / global_exports.maximum(sys.float_info.epsilon)

        new_imports = self.imports * import_factor
        new_exports = self.exports * export_factor

        self.imports = new_imports
        self.exports = new_exports

    @staticmethod
    def get_reference_trade(
        global_imports: fd.FlodymArray, global_exports: fd.FlodymArray, to: str = "hmean"
    ):
        reference_trade_lookup = {
            "maximum": global_imports.maximum(global_exports),
            "minimum": global_imports.minimum(global_exports),
            "imports": global_imports,
            "exports": global_exports,
            # this is the same method as referenced in Michaja's paper
            "hmean": fd.FlodymArray(
                dims=global_exports.dims,
                values=hmean(np.stack([global_imports.values, global_exports.values])),
            ),
            "gmean": fd.FlodymArray(
                dims=global_exports.dims,
                values=gmean(np.stack([global_imports.values, global_exports.values])),
            ),
            "amean": (global_imports + global_exports) / 2,
        }
        if to not in reference_trade_lookup:
            raise ValueError(
                f"Extrenum {to} not recognized. Must be one of {list(reference_trade_lookup.keys())}"
            )
        return reference_trade_lookup[to]


class TradeSet(PydanticBaseModel):
    """A trade model for the steel sector storing the data and defining how trade is processed."""

    markets: dict[str, Trade]

    @classmethod
    def from_definitions(cls, definitions: List["TradeDefinition"], dims: fd.DimensionSet):
        markets = {}
        for d in definitions:
            markets[d.name] = Trade(
                imports=fd.FlodymArray(dims=dims[d.dim_letters]),
                exports=fd.FlodymArray(dims=dims[d.dim_letters]),
            )
        return cls(markets=markets)

    def __getitem__(self, item):
        return self.markets[item]

    def __setitem__(self, key: str, value: Trade):
        if not isinstance(value, Trade):
            raise ValueError("TradeSet can only store Trade objects.")
        if key not in self.markets:
            raise ValueError(
                f"TradeSet does not have a trade named {key}. You can add new trades using trade.markets[key] = value."
            )
        if self.markets[key].imports.dims.letters != value.imports.dims.letters:
            raise ValueError(
                f"Trade dimensions do not match. Expected {self.markets[key].imports.dims.letters}, got {value.imports.dims.letters}."
            )
        self.markets[key] = value

    def balance(self, to: str = None):
        for trade in self.markets.values():
            trade.balance(to=to) if to is not None else trade.balance()


class TradeDefinition(fd.ParameterDefinition):
    pass
