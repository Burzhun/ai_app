from typing import Any
from fastapi import APIRouter, Depends
from src.ml_models.nlp_models.transformers import TransformerModel
from src.server.models.nlp_basemodels import TransformersParams

transformers_router = APIRouter()
active_model_name, active_model = None, None


def get_model(model_name, **transformer_kwargs):
    global active_model, active_model_name
    if active_model_name != model_name:
        active_model = TransformerModel(model_name, **transformer_kwargs)
        active_model_name = model_name
    return active_model, active_model_name


@transformers_router.post("/check_text")
async def transformers_check_text(
    request_params: TransformersParams = Depends(),
) -> Any:
    """Исправить орфографию и пунктуацию, вернуть список ошибок с их
    метаданными
    """
    text = request_params.request
    model_name = request_params.model_name
    active_model, active_model_name = get_model(model_name)
    method = active_model.__getattribute__(request_params.method_name)
    return method(text)
