from typing import Callable, Dict, List
from src.basemodels.nlp_models.base import BaseNLPModel, BaseNLPVerbosePrediction


# TODO: параллельное выполнение всех моделей в process pool-e
class NLPipeline(BaseNLPModel):
    """
    Класс NLPipeline представляет конвейер для обработки текста с помощью нескольких NLP-моделей
    параллельно - то есть возвращая результат всех моделей.

    Атрибуты:
        models (List[BaseNLPModel]): Список NLP-моделей, которые будут использоваться для обработки текста.
    """

    def __init__(self, models: List[BaseNLPModel]):
        """
        Инициализирует NLPipeline с заданным списком моделей.

        Аргументы:
            models (List[BaseNLPModel]): Список NLP-моделей для использования в конвейере.
        """
        self.models = models

    def _concat_predictions(
        self, text: str, predict_func_name: str = "predict"
    ) -> Dict[str, Callable]:
        """
        Внутренний метод для объединения предсказаний всех моделей в один словарь.

        Аргументы:
            text (str): Текст, который нужно обработать.
            predict_func_name (str): Название метода предсказания, который будет вызываться у каждой модели
                (по умолчанию "predict").

        Возвращает:
            Dict[str, Callable]: Словарь, где ключом является имя модели, а значением — результат её предсказания.
        """
        predictions = {}
        for model in self.models:
            predict_func: Callable = getattr(model, predict_func_name)
            predictions[model.__class__.__name__] = predict_func(text)
        return predictions

    def predict(self, text: str) -> Dict[str, str]:
        """
        Возвращает предсказания всех моделей для заданного текста в виде исправленного текста.

        Аргументы:
            text (str): Текст, который нужно обработать.

        Возвращает:
            Dict[str, str]: Словарь, где ключом является имя модели, а значением — исправленный текст.
        """
        return self._concat_predictions(text, predict_func_name="predict")

    def predict_verbose(self, text: str) -> Dict[str, BaseNLPVerbosePrediction]:
        """
        Возвращает подробные предсказания всех моделей для заданного текста, включая исправления и сообщения.

        Аргументы:
            text (str): Текст, который нужно обработать.

        Возвращает:
            Dict[str, BaseNLPVerbosePrediction]: Словарь, где ключом является имя модели, а значением —
            подробное предсказание, содержащее исправления, сообщения и другие метаданные.
        """
        return self._concat_predictions(text, predict_func_name="predict_verbose")

    def evaluate(self, text: str, answer: str) -> Dict:
        """
        Оценивает производительность моделей на основе предоставленного текста и правильного ответа.

        Аргументы:
            text (str): Текст, который нужно оценить.
            answer (str): Правильный ответ для сравнения с результатами моделей.

        Возвращает:
            Dict: Словарь с результатами оценки для каждой модели (пока не реализовано).

        Вызывает:
            NotImplementedError: Этот метод ещё не реализован.
        """
        raise NotImplementedError
        return super().evaluate(text, answer)
