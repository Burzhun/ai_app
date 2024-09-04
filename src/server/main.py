from fastapi import FastAPI
from src.server.routers.cv_routers.yandex_vision_ocr_router import yandex_vision_ocr_router
from src.server.routers.nlp_routers.transformers_router import transformers_router
from src.server.routers.nlp_routers.yandex_speller_router import yandex_speller_router
from src.server.routers.nlp_routers.pylanguagetool_router import pylanguagetool_router


app = FastAPI()

app.include_router(
    pylanguagetool_router,
    prefix="/pylanguagetool",
    tags=["pylanguagetool"],
)

app.include_router(
    yandex_speller_router,
    prefix="/yandex",
    tags=["yandex"],
)

app.include_router(
    transformers_router,
    prefix="/transformers",
    tags=["transformers"],
)

app.include_router(
    yandex_vision_ocr_router, prefix="/yandex_vision_ocr", tags=["yandex_vision_ocr"]
)
