from typing import Any
from fastapi import APIRouter, Depends
from src.ml_models.nlp_models import pylanguagetool
from src.server.models.nlp_basemodels import NLPParams

pylanguagetool_router = APIRouter()
active_model = None


def get_model() -> pylanguagetool.PyLanguageToolModel:
    global active_model
    if active_model is None:
        active_model = pylanguagetool.PyLanguageToolModel()
    return active_model


@pylanguagetool_router.post("/check_text")
async def transformers_check_text(
    request_params: NLPParams = Depends(),
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
