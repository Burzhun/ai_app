import pytest
import requests as m
from src.basemodels.nlp_models.base import BaseNLPPrediction
from src.ml_models.nlp_models.yandex_speller import YandexSpellerModel


@pytest.fixture
def speller_model():
    return YandexSpellerModel()


@pytest.fixture
def text():
    return "превет как дила"


@pytest.fixture
def expected_corrected_text():
    return "привет как дела"


@pytest.fixture
def expected_response():
    return [
        {
            "code": 1,
            "pos": 0,
            "row": 0,
            "col": 0,
            "len": 6,
            "word": "превет",
            "s": ["привет", "превет", "превед"],
        },
        {
            "code": 1,
            "pos": 11,
            "row": 0,
            "col": 11,
            "len": 4,
            "word": "дила",
            "s": ["дела", "дила"],
        },
    ]


@pytest.fixture
def expected_corrections():
    return [
        BaseNLPPrediction(
            index=0,
            error="превет",
            suggestions=["привет", "превет", "превед"],
            message="",
        ),
        BaseNLPPrediction(
            index=11, error="дила", suggestions=["дела", "дила"], message=""
        ),
    ]


def test_send_request(speller_model, text, expected_response):
    m.get(speller_model.spell_service_url, json=expected_response)
    response = speller_model._send_request(text)
    assert response == expected_response


def test_predict(speller_model, text, expected_corrected_text):
    response = speller_model.predict(text)
    assert response == expected_corrected_text


def test_predict_verbose(
    speller_model, text, expected_corrected_text, expected_corrections
):
    response = speller_model.predict_verbose(text)
    assert response.text == expected_corrected_text
    assert response.corrections == expected_corrections
