from abc import ABC, abstractmethod
import pydantic


class BaseCVPrediction(pydantic.BaseModel):
    text: str

    def __eq__(self, other):
        return self.text == other.text


class BaseCVVerbosePrediction(pydantic.BaseModel):
    json_loads: dict
    text: str

    def __eq__(self, other):
        return self.json_loads == other.json_loads and self.text == other.text


class BaseCVModel(ABC):
    def __init__(self):
        pass

    @pydantic.validate_call(validate_return=True)
    @abstractmethod
    def predict(self, file_name: str) -> BaseCVVerbosePrediction:
        pass

    @pydantic.validate_call(validate_return=True)
    @abstractmethod
    def predict_verbose(self, file_name: str) -> BaseCVPrediction:
        pass
