from typing import Literal
from pydantic import BaseModel


class CVParams(BaseModel):
    file_name: str
    method_name: Literal[
        "predict",
        "predict_verbose",
    ] = "predict_verbose"


class OCRParams(CVParams):
    model_name: Literal[
        "Yandex Vision OCR",
    ] = "Yandex Vision OCR"
