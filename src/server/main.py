from fastapi import FastAPI,File, UploadFile, Body
from os.path import abspath
from src.pipeline_init.united_pipeline import getImageText,getProcessedText,base64_to_image
from src.server.routers.nlp_routers import (
    pylanguagetool_router,
    yandex_speller_router,
    transformers_router,
)
from src.server.routers.cv_routers import (
    yandex_vision_ocr_router

)
from fastapi.responses import StreamingResponse
import time
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import random
import os
import re
from datetime import datetime
import asyncio
from src.loggers.basic_logger import get_logger
from dotenv import load_dotenv
from src.cv_utils.utils import textOneLine
import uvicorn

load_dotenv()

logger = get_logger(name='base_logger')
app = FastAPI()

origins = [
    os.environ.get("CLIENT_URL")
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



app.include_router(
    pylanguagetool_router.pylanguagetool_router,
    prefix="/pylanguagetool",
    tags=["pylanguagetool"],
)

app.include_router(
    yandex_speller_router.yandex_speller_router,
    prefix="/yandex",
    tags=["yandex"],
)

app.include_router(
    transformers_router.transformers_router,
    prefix="/transformers",
    tags=["transformers"],
)

app.include_router(
    yandex_vision_ocr_router.yandex_vision_ocr_router,
    prefix="/yandex_vision_ocr",
    tags=["yandex_vision_ocr"]
)

@app.get("/test")
async def root():
    return {"message": "test"}

class ImageData(BaseModel):
    image: str

@app.post("/upload")
def upload(payload: ImageData):
    s=''

    try:
        date = datetime.now().strftime('%Y-%m-%d__%H_%M_%S')
        print('date')
        image = base64_to_image(payload.image)
        print('image')
        s = 'images/img_'+str(date)+'.jpg'
        print('image s')
        image.save(s)
        return {"error":0,"message": s}

        
        
    except Exception as e:
        print(e)
        return {"error":2,"message": e}

    return {"error":1,"message": ''}


async def event_stream2(filePath: str):
    event_str = 'event:load_stage'
    data_str = f"data: Извлечение текста"
    yield f"{event_str}\n{data_str}\n\n"
    return


async def event_stream(filePath: str):
    s=''
    try:
        # contents = file.file.read()
        # with open('sample_img.jpg', 'wb') as f:
        #     f.write(contents)
        if filePath:
            event_str = 'event:load_stage'
            data_str = f"data: Извлечение текста"
            yield f"{event_str}\n{data_str}\n\n"
            s=getImageText(filePath)
            data_str = f"data: Проверка орфографии"
            yield f"{event_str}\n{data_str}\n\n"
            # get the result
            original_text=re.sub(r'(?<=[.,])(?=[^\s])', r' ', s)
            original_text = textOneLine(original_text)
            task = asyncio.to_thread(getProcessedText, original_text)
            text, errors = await task
            event_str = 'event:text_errors'
            data_str = f"data: {errors}"
            yield f"{event_str}\n{data_str}\n\n"
            event_str = 'event:original_text'
            data_str = f"data: {original_text}"
            yield f"{event_str}\n{data_str}\n\n"
            data_str = f"data: {text}"
            event_str = 'event:load_finished'
            yield f"{event_str}\n{data_str}\n\n"
            
            os.remove(filePath)
            return

        
    except Exception as e:
        event_str = 'event:load_error'
        data_str = f"data: Ошибка"
        logger.error('Error 1')
        yield f"{event_str}\n{data_str}\n\n"
        print(e)
        return

@app.get("/stream")
async def stream(name: str):
    return StreamingResponse(event_stream(name), media_type="text/event-stream")


if __name__ == '__main__':
    uvicorn.run(app, port=8000, host='0.0.0.0')
