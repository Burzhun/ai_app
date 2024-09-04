from src.basemodels.cv_models.base import (
    BaseCVPrediction,
    BaseCVVerbosePrediction,
    BaseCVModel,
)

from src.cv_utils.utils import api_request, json_to_text
from src.loggers.basic_logger import get_logger

logger = get_logger(name="base_logger")


class YandexVisionOcrModel(BaseCVModel):
    """
    Класс для оцифровки рукописного текста с использованием модели Yandex Vision OCR.

    Attributes:

    """

    def __init__(self):
        super().__init__()
        self.ocr_service_url = "https://ocr.api.cloud.yandex.net/ocr/v1/recognizeText"
        logger.info("Initialised YandexVisionOcrModel")

    def predict_verbose(
        self,
        file_name="/Users/ilyakasimov/Documents/GitHub/Obuch2i-NLP/src/cv_utils/sample_img.jpg",
    ) -> BaseCVVerbosePrediction:
        """
        Предсказание модели Yandex Vision OCR, полный ответ JSON
        Args:
            file_name: Имя файла картинки

        Returns: BaseCVVerbosePrediction

        """
        json_loads = api_request(file_name=file_name)
        text = json_to_text(json_loads)
        logger.info(f"Processed text from image(full JSON), image_path - {file_name}")
        return BaseCVVerbosePrediction(json_loads=json_loads, text=text)

    def predict(
        self,
        file_name="/Users/ilyakasimov/Documents/GitHub/Obuch2i-NLP/src/cv_utils/sample_img.jpg",
    ) -> BaseCVPrediction:
        """
        Предсказание модели Yandex Vision OCR, только текст на картинке полученный из JSON
        Args:
            file_name: Имя файла картинки

        Returns: BaseCVPrediction

        """
        verbose_prediction = self.predict_verbose(file_name)
        logger.info(f"Processed text from image, image_path - {file_name}")
        return BaseCVPrediction(text=verbose_prediction.text)
