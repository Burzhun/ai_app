from typing import Literal
from pydantic import BaseModel


class NLPParams(BaseModel):
    request: str
    method_name: Literal[
        "predict",
        "predict_verbose",
    ] = "predict_verbose"


class TransformersParams(NLPParams):
    model_name: Literal[
        "ai-forever/sage-fredt5-distilled-95m",
        "ai-forever/sage-fredt5-large",
        "ai-forever/sage-m2m100-1.2B",
        "ai-forever/sage-mt5-large",
    ] = "ai-forever/sage-fredt5-distilled-95m"
