import language_tool_python

from src.basemodels.nlp_models.base import (
    BaseEvaluation,
    BaseNLPModel,
    BaseNLPPrediction,
    BaseNLPVerbosePrediction,
)

from src.loggers.basic_logger import get_logger

logger = get_logger(name="base_logger")


class PyLanguageToolModel(BaseNLPModel):
    """
    A language model utilizing the LanguageTool library to check grammar
    and spelling in Russian text.

    Attributes:
        language_tool (LanguageTool): An instance of the LanguageTool class
            for Russian grammar checking.
    """

    def __init__(self):
        """
        Initializes the PyLanguageToolModel with the Russian language setting
        for grammar checking.
        """
        super().__init__()
        self.language_tool = language_tool_python.LanguageTool("ru")
        logger.info("Initialized PyLanguageToolModel")

    def predict_verbose(self, text: str) -> BaseNLPVerbosePrediction:
        """
        Corrects the input text using the LanguageTool library and returns detailed information
        about the corrections made, including suggestions and messages.

        Args:
            text (str): The input text to be corrected.

        Returns:
            BaseNLPVerbosePrediction: A dataclass containing the corrected text and
            a list of all corrections with detailed information.
        """
        matches = self.language_tool.check(text)
        corrections = []
        for match in matches:
            error = text[match.offset : match.offset + match.errorLength + 1]
            error = error.replace(" ", "")
            corrections.append(
                BaseNLPPrediction(
                    **{
                        "index": match.offset,
                        "error": error,
                        "suggestions": match.replacements,
                        "message": match.message,
                    }
                )
            )
            try:
                repl = match.replacements[0]
            except IndexError:
                repl = error
            text = text.replace(error, repl)

        logger.info(
            f"Generated corrected text (with list of all corrections) from initial text: initial text - {text}"
        )
        return BaseNLPVerbosePrediction(corrections=corrections, text=text)

    def predict(self, text: str) -> str:
        """
        Corrects the input text using the LanguageTool library and returns only the corrected text.

        Args:
            text (str): The input text to be corrected.

        Returns:
            str: The corrected version of the input text.
        """
        logger.info(
            f"Generated corrected text from initial text: initial text - {text}"
        )
        return self.predict_verbose(text).text

    def evaluate(self, text: str, answer: str) -> BaseEvaluation:
        """
        Evaluates the performance of the PyLanguageToolModel on the given text.
        Currently, this method is not implemented.

        Args:
            text (str): The input text to evaluate.
            answer (str): The reference text for evaluation.

        Returns:
            BaseEvaluation: The evaluation metrics, including precision, recall, and F1-score.

        Raises:
            NotImplementedError: This method is not implemented yet.
        """
        raise NotImplementedError
        logger.info("Evaluated PyLanguageToolModel performance")
        return super().evaluate(text, answer)
