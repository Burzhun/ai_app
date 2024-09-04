from src.basemodels.nlp_models.base import BaseNLPModel, BaseNLPVerbosePrediction
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from typing import List, Dict
from src.loggers.basic_logger import get_logger

logger = get_logger(name="base_logger")


class TransformerModel(BaseNLPModel):
    """
    Класс для проверки орфографии и пунктуации с использованием модели на основе Transformers.

    Attributes:
        tokenizer (AutoTokenizer): Токенизатор для модели.
        model (AutoModelForSeq2SeqLM): Модель для генерации исправленного текста. Одна из:
            "ai-forever/sage-fredt5-distilled-95m",
            "ai-forever/sage-fredt5-large",
            "ai-forever/sage-m2m100-1.2B",
            "ai-forever/sage-mt5-large",
    """

    MODEL_NAMES: List[str] = [
        "ai-forever/sage-fredt5-distilled-95m",
        "ai-forever/sage-fredt5-large",
        "ai-forever/sage-m2m100-1.2B",
        "ai-forever/sage-mt5-large",
    ]

    def __init__(
        self,
        model_name: str = "ai-forever/sage-fredt5-distilled-95m",
        device: str = "cuda",
    ):
        """
        Инициализация токенизатора и модели.

        Args:
            model_name (str): Название предобученной модели. По умолчанию "ai-forever/sage-fredt5-distilled-95m".
            device (str): Устройство для вычислений ("cuda" или "cpu"). По умолчанию "cuda".
        """
        assert model_name in self.MODEL_NAMES
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Для развернутого понимания что находится под капотом модели
        # logger.info(f'Initialised NLP model with following params: tokenizer - {self.tokenizer}, model - {self.model}'
        #             f', device - {self.device}')

        # Для короткого вывода
        logger.info(
            f"Initialised NLP model with following params: model - {model_name}"
        )

    def predict_verbose(self, text: str) -> BaseNLPVerbosePrediction:
        """
        Возвращает исправленный текст и список всех предложенных исправлений.

        Args:
            text (str): Входной текст для проверки.

        Returns:
            Tuple[List[Dict[str, str]], str]: Исправленный текст и список всех предложенных исправлений.
        """
        inputs = self.tokenizer(
            text,
            max_length=None,
            padding="longest",
            truncation=False,
            return_tensors="pt",
        )
        outputs = self.model.generate(
            **inputs.to(self.model.device), max_length=inputs["input_ids"].size(1) * 1.5
        )
        corrected_text = self.tokenizer.batch_decode(
            outputs,
            skip_special_tokens=True,
        )[0]

        corrections = self._generate_corrections(text, corrected_text)

        logger.info(
            f"Generated corrected text (with list of all corrections) from initial text: initial text - {text}"
        )
        return BaseNLPVerbosePrediction(corrections=corrections, text=corrected_text)

    def _generate_corrections(
        self, original_text: str, corrected_text: str
    ) -> List[Dict[str, str]]:
        """
        Генерирует список исправлений на основе оригинального и исправленного текста.

        Args:
            original_text (str): Оригинальный текст.
            corrected_text (str): Исправленный текст.

        Returns:
            List[Dict[str, str]]: Список исправлений.
        """
        corrections = []
        original_words = original_text.split()
        corrected_words = corrected_text.split()

        # Идем по минимальной длине списков
        for idx, (original_word, corrected_word) in enumerate(
            zip(original_words, corrected_words)
        ):
            if original_word != corrected_word:
                correction = {
                    "index": original_text.index(original_word),
                    "error": original_word,
                    "suggestions": [corrected_word],
                    "message": "",
                }
                corrections.append(correction)

        # Если есть оставшиеся слова в оригинальном тексте
        if len(original_words) > len(corrected_words):
            for word in original_words[len(corrected_words) :]:
                correction = {
                    "index": original_text.index(word),
                    "error": word,
                    "suggestions": [],
                    "message": "",
                }
                corrections.append(correction)

        # Если есть добавленные слова в исправленном тексте
        if len(corrected_words) > len(original_words):
            for word in corrected_words[len(original_words) :]:
                correction = {
                    "index": len(original_text),
                    "error": "",
                    "suggestions": [word],
                    "message": "",
                }
                corrections.append(correction)

        logger.info(
            f"Generated list of all corrections based on text: text - {original_text}"
        )
        return corrections

    def predict(self, text: str) -> str:
        """
        Возвращает исправленный текст.

        Args:
            text (str): Входной текст для проверки.

        Returns:
            str: Исправленный текст.
        """

        logger.info(
            f"Generated corrected text from initial text: initial text - {text}"
        )
        return self.predict_verbose(text).text

    def evaluate(self, *args, **kwargs):
        raise NotImplementedError()
