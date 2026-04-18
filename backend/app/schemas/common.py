from typing import Any

from pydantic import BaseModel


class APIResponse(BaseModel):
    code: int = 0
    msg: str = "success"
    data: Any = None

