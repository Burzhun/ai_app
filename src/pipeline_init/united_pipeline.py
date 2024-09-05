from src.ml_models.cv_models.yandex_vision_ocr import YandexVisionOcrModel
from src.ml_models.nlp_models.transformers import TransformerModel
from src.basemodels.nlp_models.base import BaseNLPVerbosePrediction
import base64
from PIL import Image
from io import BytesIO

def united_call_verbose(ocr: YandexVisionOcrModel = YandexVisionOcrModel(), nlp: TransformerModel = TransformerModel(),
                        file_name: str = '/Users/ilyakasimov/Documents/GitHub/Obuch2i-NLP/src/cv_utils/test_img.jpg') \
        -> BaseNLPVerbosePrediction:
    """
    Единый метод, который по изображению извлекает текст и получает список ошибок (BaseNLPVerbosePrediction)
    Args:
        ocr: ocr_model
        nlp: nlp_model
        file_name: путь до файла

    Returns: BaseNLPVerbosePrediction

    """
    text = ocr.predict(file_name=file_name).text
    verbose_prediction = nlp.predict_verbose(text)
    return verbose_prediction


def united_call(file_name: str = './test_img.jpg', ocr: YandexVisionOcrModel = YandexVisionOcrModel(), nlp: TransformerModel = TransformerModel()
                ) \
        -> str:
    """
    Единый метод, который по изображению извлекает текст и получает список ошибок (строка)
    Args:
        ocr: ocr_model
        nlp: nlp_model
        file_name: путь до файла

    Returns: str

    """
    text = ocr.predict(file_name=file_name).text
    
    prediction = nlp.predict(text)

    return prediction

def getImageText(file_name: str = './test_img.jpg', ocr: YandexVisionOcrModel = YandexVisionOcrModel()) -> str:
    #s = "листы бумаги.- лисстья березы.; волчьи зубы - зубья пилы; лисстья березы; корни дерева- марковные Упражнение 554. Много яблок - много помидоров, много яблок много помидоров. Упражнение 554. Пара ботинок, валенок, сапог, чулок, Много дел, яблок, макарон, мест,"
    #return s
    text = ocr.predict(file_name=file_name).text
    return text

def getProcessedText(text: str) -> str:
    nlp: TransformerModel = TransformerModel()
    prediction, errors = nlp.predict(text)
    return prediction, errors

def base64_to_image(base64_string):
    if "data:image" in base64_string:
        base64_string = base64_string.split(",")[1]

    image_bytes = base64.b64decode(base64_string)
    image_stream = BytesIO(image_bytes)

    image = Image.open(image_stream)
    return image
