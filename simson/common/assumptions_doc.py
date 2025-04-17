from inspect import stack, getframeinfo
from pydantic import field_validator
from typing import ClassVar, Any, Optional
import os

from simson.common.base_model import SimsonBaseModel


_assumptions = []


def add_assumption_doc(
    type: str, name: str, description: str, value: str = None, source: str = None
):
    """
    Add an assumption to the list of assumptions. The assumption is stored in a global list
    and can be printed later using the print_assumptions() function.
    Args:
        type (str): The type of the assumption. Must be one of the allowed types:
            "ad-hoc fix", "model assumption", "integer number", "literature value".
        name (str): The name of the assumption. This should be a short, descriptive name.
        description (str): A thorough explanation of the assumption. Should be understandable for
            users of the model without knowledge of the code.
        value (str, optional): The value of the assumption, if applicable. Defaults to None.
        source (str, optional): The source for literature data. Defaults to None.
    """
    caller = getframeinfo(stack()[1][0])
    assumption = Assumption(
        type=type,
        name=name,
        value=value,
        description=description,
        filename=caller.filename,
        line_number=caller.lineno,
        source=source,
    )
    _assumptions.append(assumption)


class Assumption(SimsonBaseModel):
    type: str
    name: str
    description: str
    filename: str
    line_number: int
    value: Optional[Any] = None
    source: Optional[str] = None
    _allowed_types: ClassVar = [
        "ad-hoc fix",
        "model assumption",
        "integer number",
        "literature value",
    ]

    @field_validator("type", mode="after")
    @classmethod
    def check_type(cls, v):
        if v not in cls._allowed_types:
            raise ValueError("assumption type must be one of: " + str(cls._allowed_types))
        return v

    def __str__(self):
        str_out = "Assumption:\n"
        str_out += f"  Type: {self.type}\n"
        str_out += f"  Name: {self.name}\n"
        if self.value is not None:
            str_out += f"  Value: {self.value}\n"
        str_out += f"  Description: {self.description}\n"
        if self.source is not None:
            str_out += f"  Source: {self.source}\n"
        root_dir = os.path.abspath(os.curdir)
        filename_rel = os.path.relpath(self.filename, root_dir)
        str_out += f"  File: {filename_rel}, Line: {self.line_number}"
        return str_out


def assumptions_str() -> str:
    return "\n".join(str(a) for a in _assumptions)
