from pydantic import BaseModel as PydanticBaseModel
from pydantic import model_validator
from sodym.named_dim_arrays import NamedDimArray
from typing import Optional, Callable


class Trade(PydanticBaseModel):
    """ A TradeModule handles the storing and calculation of trade data for a given MFASystem."""

    imports: NamedDimArray
    exports: NamedDimArray
    balancer: Optional[Callable] = None  # e.g. functions defined in trade_balancers.py

    @model_validator(mode='after')
    def validate_region_dimension(self):
        assert 'r' in self.imports.dims.letters, "Imports must have a Region dimension."
        assert 'r' in self.exports.dims.letters, "Exports must have a Region dimension."

        return self

    @model_validator(mode='after')
    def validate_trade_dimensions(self):
        assert self.imports.dims == self.exports.dims, "Imports and Exports must have the same dimensions."

        return self

    def balance(self, **kwargs):
        if self.balancer is not None:
            self.balancer(self, **kwargs)
        else:
            raise NotImplementedError("No balancer function has been implemented for this Trade object.")

    def predict(self):
        if self.predictor is not None:
            assert 'h' in self.imports.dims.letters and 'h' in self.exports.dims.letters, \
                "Trade data must have a historic time dimension."
            self.predictor()
        else:
            raise NotImplementedError("No predictor function has been implemented for this Trade object.")

    def __getitem__(self, key):
        if key == 'Imports':
            return self.imports
        elif key == 'Exports':
            return self.exports
        else:
            raise KeyError(f"Key {key} not found in Trade data - has to be either 'Imports' or 'Exports'.")

    def __setitem__(self, key, value):
        if key == 'Imports':
            self.imports = value
        elif key == 'Exports':
            self.exports = value
        else:
            raise KeyError(f"Key {key} has to be either 'Imports' or 'Exports'.")
