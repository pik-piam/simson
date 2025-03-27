import flodym as fd
import numpy as np
from typing import Optional
from pydantic import model_validator

from simson.common.base_model import SimsonBaseModel


def broadcast_trailing_dimensions(array: np.ndarray, to_shape_of: np.ndarray) -> np.ndarray:
    """Broadcasts array to shape of to_shape_of, adding dimensions if necessary."""
    new_shape = array.shape + (1,) * (len(to_shape_of.shape) - len(array.shape))
    b_reshaped = np.reshape(array, new_shape)
    b_broadcast = np.broadcast_to(b_reshaped, to_shape_of.shape)
    return b_broadcast


class Bound(SimsonBaseModel):
    var_name: Optional[str]
    dims: fd.DimensionSet = fd.DimensionSet(dim_list=[])
    """Dimensions of the bounds. Not required if bounds are scalar."""
    lower_bound: fd.FlodymArray
    upper_bound: fd.FlodymArray

    @model_validator(mode="before")
    @classmethod
    def convert_to_fd_array(cls, data: dict):
        required_fields = ["var_name", "lower_bound", "upper_bound"]
        for field in required_fields:
            if field not in data:
                raise ValueError(f"Missing required field: {field}")

        var_name = data.get("var_name")
        dims = data.get("dims")
        lower = np.array(data.get("lower_bound"), dtype=float)
        upper = np.array(data.get("upper_bound"), dtype=float)

        if dims is None:
            dims = cls.model_fields.get("dims").default

        return {
            "var_name": var_name,
            "lower_bound": fd.FlodymArray(dims=dims, values=lower, name="lower_bound"),
            "upper_bound": fd.FlodymArray(dims=dims, values=upper, name="upper_bound"),
            "dims": dims,
        }

    @model_validator(mode="after")
    def validate_bounds(self):
        lb = self.lower_bound.values
        ub = self.upper_bound.values

        if lb.shape != ub.shape:
            raise ValueError("Lower and upper bounds must have the same shape")
        if np.any(lb > ub):
            raise ValueError("Lower bounds must be smaller than upper bounds")
        if self.dims.shape() != lb.shape:
            raise ValueError("Shape of given bounds and dims must match.")

        # Check if lower bound equals upper bound
        equal_mask = lb == ub
        if np.any(equal_mask):
            adjustment = 1e-10
            zero_mask = (lb == 0) & (ub == 0)

            # Handle case where both bounds are 0
            lb[zero_mask] = -adjustment
            ub[zero_mask] = adjustment

            # Handle general case where bounds are equal
            non_zero_mask = equal_mask & np.logical_not(zero_mask)
            lb[non_zero_mask] -= adjustment * np.abs(lb[non_zero_mask])
            ub[non_zero_mask] += adjustment * np.abs(ub[non_zero_mask])

        return self

    def extend_dims(self, target_dims: fd.DimensionSet):
        self.lower_bound = self.lower_bound.cast_to(target_dims)
        self.upper_bound = self.upper_bound.cast_to(target_dims)
        self.dims = target_dims
        return self


class BoundList(SimsonBaseModel):
    bound_list: list[Bound] = []
    target_dims: fd.DimensionSet = fd.DimensionSet(dim_list=[])
    """Dimension of the extrapolation to which the bounds are extended."""

    @model_validator(mode="after")
    def cast_bounds(self):
        for idx, bound in enumerate(self.bound_list):
            if set(bound.dims.letters).issubset(self.target_dims.letters):
                self.bound_list[idx] = bound.extend_dims(self.target_dims)
            else:
                raise ValueError(f"Bound {bound.var_name} has dimensions not in target_dims.")
        return self

    def create_bounds_arr(self, all_prm_names: list[str]) -> np.ndarray:
        """Creates bounds array where each element is tuple of lower and upper bounds for each parameter."""

        if self.bound_list == []:
            return None

        invalid_params = set(b.var_name for b in self.bound_list) - set(all_prm_names)
        if invalid_params:
            raise ValueError(f"Unknown parameters in bounds: {invalid_params}")

        bound_shape = self.bound_list[0].upper_bound.values.shape
        param_positions = {name: i for i, name in enumerate(all_prm_names)}

        lower_bounds = np.full(bound_shape + (len(all_prm_names),), -np.inf)
        upper_bounds = np.full(bound_shape + (len(all_prm_names),), np.inf)

        for bound in self.bound_list:
            pos = param_positions[bound.var_name]
            lower_bounds[..., pos] = bound.lower_bound.values
            upper_bounds[..., pos] = bound.upper_bound.values

        bounds = np.stack((lower_bounds, upper_bounds), axis=-2)
        return bounds
