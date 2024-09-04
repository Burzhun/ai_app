from abc import ABC, abstractmethod
from typing import List
import pydantic


class BaseNLPPrediction(pydantic.BaseModel):
    index: int
    error: str
    suggestions: List[str]
    message: str


class BaseNLPVerbosePrediction(pydantic.BaseModel):
    corrections: List[BaseNLPPrediction]
    text: str


class BaseEvaluation(pydantic.BaseModel):
    precision: float
    recall: float
    F1: float


class BaseNLPModel(ABC):
    def __init__(self):
        pass

    @pydantic.validate_call(validate_return=True)
    @abstractmethod
    def predict(self, text: str) -> str:
        pass

    @pydantic.validate_call(validate_return=True)
    @abstractmethod
    def predict_verbose(self, text: str) -> BaseNLPVerbosePrediction:
        pass

    @pydantic.validate_call(validate_return=True)
    @abstractmethod
    def evaluate(self, text: str, answer: BaseNLPPrediction) -> BaseEvaluation:
        pass
