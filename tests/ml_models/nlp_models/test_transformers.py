import pytest
from src.basemodels.nlp_models.base import BaseNLPPrediction
from src.ml_models.nlp_models.transformers import TransformerModel


@pytest.fixture
def speller_model():
    return TransformerModel()


@pytest.fixture
def text():
    return "превет как дила"


@pytest.fixture
def expected_corrected_text():
    return "Привет, как дела!"


@pytest.fixture
def expected_corrections():
    return [
        BaseNLPPrediction(
            index=0,
            error="превет",
            suggestions=["Привет,"],
            message="",
        ),
        BaseNLPPrediction(
            index=11,
            error="дила",
            suggestions=["дела!"],
            message="",
        ),
    ]


def test_predict(speller_model, text, expected_corrected_text):
    response = speller_model.predict(text)
    assert response == expected_corrected_text


def test_predict_verbose(
    speller_model, text, expected_corrected_text, expected_corrections
):
    response = speller_model.predict_verbose(text)
    print(response)
    assert response.text == expected_corrected_text
    assert response.corrections == expected_corrections
