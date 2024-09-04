import abc
import wandb
from typing import Iterable
import warnings
import pandas as pd
from tqdm import tqdm
from src.base import BaseModel
from sage.evaluation.scorer import Scorer


class BaseModelScorer(abc.ABC):

    @abc.abstractmethod
    def score(self, predictions: Iterable, **scoring_kwargs):
        pass


class SageModelScorer(BaseModelScorer):

    scoring_model_class = Scorer
    predictions_columns = ["Source", "Truth", "Model_result", "Model_is_correct"]

    @staticmethod
    def _check_init_args(dataset, scoring_model):
        assert hasattr(scoring_model, "score")

    def __init__(
        self,
        dataset,
        scoring_model=scoring_model_class(load_errant=True),
    ):
        self._check_init_args(dataset, scoring_model)
        self.dataset = dataset
        self.scoring_model = scoring_model

    def _score(self, predictions: Iterable, metrics, **scoring_kwargs):
        result = self.scoring_model.score(
            sources=self.dataset["source"],
            corrections=self.dataset["correction"],
            answers=predictions,
            metrics=metrics,
            **scoring_kwargs,
        )
        return result

    def _get_predictions(self, model: BaseModel):
        return [model.predict(i) for i in tqdm(self.dataset["source"])]

    def _explain_predictions(self, predictions):
        if len(predictions) != len(self.dataset):
            warnings.warn(
                f"Длина датасета {len(self.dataset)} не равна длине предсказаний {len(predictions)}"
            )
        raw_df_data = [
            [
                self.dataset["source"][i],
                self.dataset["correction"][i],
                prediction,
                prediction.lower() == self.dataset["correction"][i].lower(),
            ]
            for i, prediction in enumerate(predictions)
        ]
        df = pd.DataFrame(raw_df_data, columns=self.predictions_columns)
        return df

    def _get_score_and_predictions(
        self, model: BaseModel, metrics=["errant", "ruspelleval"], **scoring_kwargs
    ):
        predictions = self._get_predictions(model)
        score = self._score(predictions, metrics, **scoring_kwargs)
        return score, predictions

    def score(
        self, model: BaseModel, metrics=["errant", "ruspelleval"], **scoring_kwargs
    ):
        score, predictions = self._get_score_and_predictions(
            model, metrics, **scoring_kwargs
        )
        return score

    def score_explain(
        self, model: BaseModel, metrics=["errant", "ruspelleval"], **scoring_kwargs
    ):
        score, predictions = self._get_score_and_predictions(
            model, metrics, **scoring_kwargs
        )
        explanation = self._explain_predictions(predictions)
        return score, explanation


class WandbSageModelScorer(SageModelScorer):

    def __init__(self, *args, project: str, run_suffix: str = "", **kwargs):
        super().__init__(*args, **kwargs)
        wandb.init(project=project, config=kwargs)
        self.run_suffix = run_suffix

    def _get_score_and_predictions(
        self, model: BaseModel, metrics=["errant", "ruspelleval"], **scoring_kwargs
    ):
        score, predictions = super()._get_score_and_predictions(
            model, metrics, **scoring_kwargs
        )
        wandb.run.summary["score" + self.run_suffix] = score
        return score, predictions

    def _explain_predictions(self, predictions):
        df = super()._explain_predictions(predictions)
        wandb.log({"explanation" + self.run_suffix: wandb.Table(dataframe=df)})
        return df
