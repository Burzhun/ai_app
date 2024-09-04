from dataclasses import dataclass
from datasets import load_dataset, DatasetDict
from typing import Tuple


@dataclass
class DatasetConfig:
    """
    Конфигурация для загрузки датасета.

    Attributes:
        path (str): Путь к датасету.
        name (str): Название датасета.
        trust_remote_code (bool): Доверять ли удаленному коду. По умолчанию False.
    """

    path: str
    name: str
    trust_remote_code: bool = False


class DatasetLoader:
    """
    Класс для загрузки орфографических и пунктуационных датасетов.

    Attributes:
        orpho_dataset_config (DatasetConfig): Конфигурация для орфографического датасета.
        punct_dataset_config (DatasetConfig): Конфигурация для пунктуационного датасета.
    """

    def __init__(
        self,
        orpho_dataset_config: DatasetConfig = DatasetConfig(
            path="ai-forever/spellcheck_benchmark",
            name="RUSpellRU",
            trust_remote_code=False,
        ),
        punct_dataset_config: DatasetConfig = DatasetConfig(
            path="ai-forever/spellcheck_punctuation_benchmark",
            name="RUSpellRU",
            trust_remote_code=True,
        ),
    ):
        """
        Инициализация DatasetLoader с конфигурациями датасетов.

        Args:
            orpho_dataset_config (DatasetConfig): Конфигурация для орфографического датасета.
            punct_dataset_config (DatasetConfig): Конфигурация для пунктуационного датасета.
        """
        self.orpho_dataset_config = orpho_dataset_config
        self.punct_dataset_config = punct_dataset_config

    @staticmethod
    def load_dataset(dscfg: DatasetConfig) -> DatasetDict:
        """
        Загружает датасет по заданной конфигурации.

        Args:
            dscfg (DatasetConfig): Конфигурация датасета для загрузки.

        Returns:
            DatasetDict: Загруженный датасет.
        """
        return load_dataset(
            dscfg.path, dscfg.name, trust_remote_code=dscfg.trust_remote_code
        )

    def load_datasets(self) -> Tuple[DatasetDict, DatasetDict]:
        """
        Загружает орфографический и пунктуационный датасеты.

        Returns:
            Tuple[DatasetDict, DatasetDict]: Кортеж из двух загруженных датасетов.
        """
        orpho_dataset = self.load_dataset(self.orpho_dataset_config)
        punct_dataset = self.load_dataset(self.punct_dataset_config)
        return orpho_dataset, punct_dataset


def load_datasets() -> Tuple[DatasetDict, DatasetDict]:
    """
    Загружает орфографический и пунктуационный датасеты с использованием DatasetLoader.

    Returns:
        Tuple[DatasetDict, DatasetDict]: Кортеж из двух загруженных датасетов.
    """
    return DatasetLoader().load_datasets()
