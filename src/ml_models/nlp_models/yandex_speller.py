import requests
from typing import List, Dict
from src.basemodels.nlp_models.base import (
    BaseEvaluation,
    BaseNLPModel,
    BaseNLPPrediction,
    BaseNLPVerbosePrediction,
)
from src.loggers.basic_logger import get_logger

logger = get_logger(name="base_logger")


class YandexSpellerModel(BaseNLPModel):

    def __init__(self):
        super().__init__()
        self.spell_service_url = (
            "https://speller.yandex.net/services/spellservice.json/checkText"
        )
        logger.info("Initialised YandexSpellerModel")

    def _send_request(self, text: str) -> List[Dict]:
        """
        Отправляет запрос к Yandex Speller API для проверки текста.

        Args:
            text (str): Текст для проверки.

        Returns:
            List[Dict]: Список ошибок, найденных в тексте.
        """
        params = {"text": text}
        response = requests.get(self.spell_service_url, params=params)
        response.raise_for_status()
        logger.info(f"Sent request to YandexSpellerModel, text - {text}")
        return response.json()

    def _process_errors(
        self, text: str, errors: List[Dict]
    ) -> BaseNLPVerbosePrediction:
        """
        Обрабатывает ошибки и вносит исправления в текст.

        Args:
            text (str): Исходный текст.
            errors (List[Dict]): Список ошибок, найденных в тексте.

        Returns:
            BaseNLPVerbosePrediction: Объект с информацией об ошибках и исправленным текстом.
        """
        corrected_text = text
        corrections = []

        for error in errors:
            start_pos = error["pos"]
            end_pos = start_pos + error["len"]
            error_text = text[start_pos:end_pos]
            suggestions = error.get("s", [])

            if suggestions:
                corrected_text = (
                    corrected_text[:start_pos]
                    + suggestions[0]
                    + corrected_text[end_pos:]
                )

            correction = BaseNLPPrediction(
                index=start_pos,
                error=error_text,
                suggestions=suggestions,
                message=error.get("message", ""),
            )
            corrections.append(correction)
        logger.info(f"Processed errors, text - {text}, errors - {errors}")
        return BaseNLPVerbosePrediction(corrections=corrections, text=corrected_text)

    def predict_verbose(self, text: str) -> BaseNLPVerbosePrediction:
        """
        Проверяет текст на орфографические ошибки с использованием API Yandex Speller.

        Args:
            text (str): Текст для проверки.

        Returns:
            BaseNLPVerbosePrediction: Объект с информацией об ошибках и исправленным текстом.
        """
        errors = self._send_request(text)
        logger.info(
            f"Generated corrected text (with list of all corrections) from initial text: initial text - {text}"
        )
        return self._process_errors(text, errors)

    def predict(self, text: str) -> str:
        """
        Возвращает текст с исправленными ошибками.

        Args:
            text (str): Текст для проверки.

        Returns:
            str: Исправленный текст.
        """
        verbose_prediction = self.predict_verbose(text)
        logger.info(
            f"Generated corrected text from initial text: initial text - {text}"
        )
        return verbose_prediction.text

    def evaluate(self, text: str, answer: str) -> BaseEvaluation:
        """
        Оценка исправленного текста относительно ответа.
        Метод не реализован и должен быть переопределен в подклассах.

        Args:
            text (str): Исходный текст.
            answer (str): Ожидаемый текст.

        Returns:
            BaseEvaluation: Оценка результатов.
        """
        raise NotImplementedError("This method should be overridden in subclasses")
        logger.info("Evaluated YandexSpellerModel performance")
        return super().evaluate(text, answer)
