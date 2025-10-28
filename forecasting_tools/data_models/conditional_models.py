from typing import Literal

from pydantic import BaseModel

from forecasting_tools.data_models.multiple_choice_report import PredictedOptionList
from forecasting_tools.data_models.numeric_report import NumericDistribution


class ConditionalPrediction(BaseModel):
    parent: NumericDistribution | PredictedOptionList | float | Literal["affirm"]
    child: NumericDistribution | PredictedOptionList | float | Literal["affirm"]
    prediction_yes: NumericDistribution | PredictedOptionList | float
    prediction_no: NumericDistribution | PredictedOptionList | float
