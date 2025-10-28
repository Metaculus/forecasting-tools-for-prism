from typing import Literal

from pydantic import BaseModel


class ConditionalPrediction(BaseModel):
    parent: float | Literal["affirm"]
    child: float | Literal["affirm"]
    prediction_yes: float
    prediction_no: float
