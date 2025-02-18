from pydantic import BaseModel, ConfigDict


class SimsonBaseModel(BaseModel):

    model_config = ConfigDict(extra="forbid", protected_namespaces=(), arbitrary_types_allowed=True)
