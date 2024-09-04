from typing import Any
from fastapi import APIRouter, Body
from src.ml_models.cv_models.yandex_vision_ocr import YandexVisionOcrModel
from src.server.models.cv_basemodels import CVParams as YandexParams

yandex_vision_ocr_router = APIRouter()


def get_model():
    return YandexVisionOcrModel


@yandex_vision_ocr_router.post("/check_image")
async def ocr_check_image(
    request_params: YandexParams = Body(),
) -> Any:
    """

    Args:
        request_params:

    Returns:

    """
    file_name = request_params.file_name
    model = get_model()
    method = model.__getattribute__(request_params.method_name)
    return method(file_name)
