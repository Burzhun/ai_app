from typing import Any
from fastapi import APIRouter, Body, Depends
from src.ml_models.nlp_models.yandex_speller import YandexSpellerModel
from src.server.models.nlp_basemodels import NLPParams as YandexParams

yandex_speller_router = APIRouter()


def get_model():
    return YandexSpellerModel()


@yandex_speller_router.post("/check_text")
async def transformers_check_text(
    request_params: YandexParams = Depends(),
) -> Any:
    """Исправить орфографию и пунктуацию, вернуть список ошибок с их
    метаданными

    Args:
        request (TextRequest): сам текст

    Returns:
        dict: словарь {"название модели": [ошибки и их данные]}
    """
    text = request_params.request
    model = get_model()
    method = model.__getattribute__(request_params.method_name)
    return method(text)
