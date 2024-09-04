from src.ml_models.cv_models.yandex_vision_ocr import YandexVisionOcrModel
from src.ml_models.nlp_models.transformers import TransformerModel
from src.basemodels.nlp_models.base import BaseNLPVerbosePrediction


def united_call_verbose(ocr: YandexVisionOcrModel = YandexVisionOcrModel(), nlp: TransformerModel = TransformerModel(),
                        file_name: str = '/Users/ilyakasimov/Documents/GitHub/Obuch2i-NLP/src/cv_utils/sample_img.jpg') \
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


def united_call(ocr: YandexVisionOcrModel = YandexVisionOcrModel(), nlp: TransformerModel = TransformerModel(),
                file_name: str = '/Users/ilyakasimov/Documents/GitHub/Obuch2i-NLP/src/cv_utils/sample_img.jpg') \
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
