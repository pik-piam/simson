import numpy as np
from scipy.optimize import Bounds
from typing import Union


def broadcast_trailing_dimensions(array: np.ndarray, to_shape_of: np.ndarray) -> np.ndarray:
    """Broadcasts array to shape of to_shape_of, adding dimensions if necessary."""
    new_shape = array.shape + (1,) * (len(to_shape_of.shape) - len(array.shape))
    b_reshaped = np.reshape(array, new_shape)
    b_broadcast = np.broadcast_to(b_reshaped, to_shape_of.shape)
    return b_broadcast


class Bound:
    """Class representing bounds for a single parameter."""

    def __init__(
        self,
        var_name: str,
        lower_bound: Union[float, np.ndarray],
        upper_bound: Union[float, np.ndarray],
    ):
        self.var_name = var_name
        self.lower_bound = np.array(lower_bound)
        self.upper_bound = np.array(upper_bound)

    @staticmethod
    def create_bounds_arr(
        bounds_list: list["Bound"], all_prm_names: list[str], bound_shape: tuple
    ) -> np.ndarray:
        """Creates bounds array where each element is tuple of lower and upper bounds for each parameter."""

        # Check bound shapes
        if bounds_list:
            if not bound_shape == bounds_list[0].lower_bound.shape:
                raise ValueError("Bounds shape must match target shape")

        if any(
            b.lower_bound.shape != bound_shape or b.upper_bound.shape != bound_shape
            for b in bounds_list
        ):
            raise ValueError("All bounds must have the same shape")

        # Check for invalid parameter names
        invalid_params = set(b.var_name for b in bounds_list) - set(all_prm_names)
        if invalid_params:
            raise ValueError(f"Unknown parameters in bounds: {invalid_params}")

        bounds = np.empty(bound_shape, dtype=object)
        param_positions = {name: i for i, name in enumerate(all_prm_names)}
        for index in np.ndindex(bound_shape):
            # Initialize default bounds
            lower_bounds = [-np.inf] * len(all_prm_names)
            upper_bounds = [np.inf] * len(all_prm_names)

            # Update bounds for parameters that have them
            for bound in bounds_list:
                pos = param_positions[bound.var_name]
                lower_bounds[pos] = bound.lower_bound[index]
                upper_bounds[pos] = bound.upper_bound[index]
                if lower_bounds[pos] == upper_bounds[pos]:
                    # avoid error in least_squares
                    lower_bounds[pos] = lower_bounds[pos] * 0.999999
                    upper_bounds[pos] = upper_bounds[pos] * 1.000001

            bounds[index] = (np.array(lower_bounds), np.array(upper_bounds))

        return bounds
