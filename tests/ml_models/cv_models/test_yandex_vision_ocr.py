import pytest
import requests as r
from src.basemodels.cv_models.base import BaseCVPrediction, BaseCVVerbosePrediction
from src.ml_models.cv_models.yandex_vision_ocr import YandexVisionOcrModel


@pytest.fixture
def vision_model():
    return YandexVisionOcrModel()


@pytest.fixture
def text():
    return 'Напишу рукописный\nтекст с фото, скрима,\nпечатного текста или\nфотоскана\n'


@pytest.fixture
def json_loads():
    json_loads = {'result':
                      {'textAnnotation':
                           {'width': '660', 'height': '440', 'blocks':
                               [{'boundingBox': {'vertices': [{'x': '59', 'y': '102'}, {'x': '59', 'y': '333'},
                                                              {'x': '580', 'y': '333'}, {'x': '580', 'y': '102'}]},
                                 'lines': [{'boundingBox': {
                                     'vertices': [{'x': '141', 'y': '102'}, {'x': '141', 'y': '168'},
                                                  {'x': '573', 'y': '168'}, {'x': '573', 'y': '102'}]},
                                     'text': 'Напишу рукописный', 'words': [{'boundingBox': {
                                         'vertices': [{'x': '141', 'y': '121'}, {'x': '141', 'y': '173'},
                                                      {'x': '298', 'y': '173'}, {'x': '298', 'y': '121'}]},
                                         'text': 'Напишу',
                                         'entityIndex': '-1',
                                         'textSegments': [{'startIndex': '0',
                                                           'length': '6'}]},
                                         {'boundingBox': {'vertices': [
                                             {'x': '329', 'y': '120'},
                                             {'x': '329', 'y': '172'},
                                             {'x': '573', 'y': '172'},
                                             {'x': '573', 'y': '120'}]},
                                             'text': 'рукописный',
                                             'entityIndex': '-1',
                                             'textSegments': [{'startIndex': '7',
                                                               'length': '10'}]}],
                                     'textSegments': [{'startIndex': '0', 'length': '17'}],
                                     'orientation': 'ANGLE_0'}, {'boundingBox': {
                                     'vertices': [{'x': '83', 'y': '183'}, {'x': '83', 'y': '223'},
                                                  {'x': '580', 'y': '223'}, {'x': '580', 'y': '183'}]},
                                     'text': 'текст с фото, скрима,', 'words': [{
                                         'boundingBox': {
                                             'vertices': [
                                                 {
                                                     'x': '83',
                                                     'y': '191'},
                                                 {
                                                     'x': '83',
                                                     'y': '227'},
                                                 {
                                                     'x': '208',
                                                     'y': '227'},
                                                 {
                                                     'x': '208',
                                                     'y': '191'}]},
                                         'text': 'текст',
                                         'entityIndex': '-1',
                                         'textSegments': [
                                             {
                                                 'startIndex': '18',
                                                 'length': '5'}]},
                                         {
                                             'boundingBox': {
                                                 'vertices': [
                                                     {
                                                         'x': '240',
                                                         'y': '189'},
                                                     {
                                                         'x': '240',
                                                         'y': '223'},
                                                     {
                                                         'x': '279',
                                                         'y': '223'},
                                                     {
                                                         'x': '279',
                                                         'y': '189'}]},
                                             'text': 'с',
                                             'entityIndex': '-1',
                                             'textSegments': [
                                                 {
                                                     'startIndex': '24',
                                                     'length': '1'}]},
                                         {
                                             'boundingBox': {
                                                 'vertices': [
                                                     {
                                                         'x': '302',
                                                         'y': '184'},
                                                     {
                                                         'x': '302',
                                                         'y': '221'},
                                                     {
                                                         'x': '419',
                                                         'y': '221'},
                                                     {
                                                         'x': '419',
                                                         'y': '184'}]},
                                             'text': 'фото,',
                                             'entityIndex': '-1',
                                             'textSegments': [
                                                 {
                                                     'startIndex': '26',
                                                     'length': '5'}]},
                                         {
                                             'boundingBox': {
                                                 'vertices': [
                                                     {
                                                         'x': '436',
                                                         'y': '179'},
                                                     {
                                                         'x': '436',
                                                         'y': '217'},
                                                     {
                                                         'x': '580',
                                                         'y': '217'},
                                                     {
                                                         'x': '580',
                                                         'y': '179'}]},
                                             'text': 'скрима,',
                                             'entityIndex': '-1',
                                             'textSegments': [
                                                 {
                                                     'startIndex': '32',
                                                     'length': '7'}]}],
                                     'textSegments': [
                                         {'startIndex': '18', 'length': '21'}],
                                     'orientation': 'ANGLE_0'}, {'boundingBox': {
                                     'vertices': [{'x': '59', 'y': '242'}, {'x': '59', 'y': '279'},
                                                  {'x': '498', 'y': '279'}, {'x': '498', 'y': '242'}]},
                                     'text': 'печатного текста или',
                                     'words': [{
                                         'boundingBox': {
                                             'vertices': [
                                                 {
                                                     'x': '59',
                                                     'y': '246'},
                                                 {
                                                     'x': '59',
                                                     'y': '283'},
                                                 {
                                                     'x': '240',
                                                     'y': '283'},
                                                 {
                                                     'x': '240',
                                                     'y': '246'}]},
                                         'text': 'печатного',
                                         'entityIndex': '-1',
                                         'textSegments': [
                                             {
                                                 'startIndex': '40',
                                                 'length': '9'}]},
                                         {
                                             'boundingBox': {
                                                 'vertices': [
                                                     {
                                                         'x': '264',
                                                         'y': '242'},
                                                     {
                                                         'x': '264',
                                                         'y': '277'},
                                                     {
                                                         'x': '400',
                                                         'y': '277'},
                                                     {
                                                         'x': '400',
                                                         'y': '242'}]},
                                             'text': 'текста',
                                             'entityIndex': '-1',
                                             'textSegments': [
                                                 {
                                                     'startIndex': '50',
                                                     'length': '6'}]},
                                         {
                                             'boundingBox': {
                                                 'vertices': [
                                                     {
                                                         'x': '424',
                                                         'y': '239'},
                                                     {
                                                         'x': '424',
                                                         'y': '273'},
                                                     {
                                                         'x': '498',
                                                         'y': '273'},
                                                     {
                                                         'x': '498',
                                                         'y': '239'}]},
                                             'text': 'или',
                                             'entityIndex': '-1',
                                             'textSegments': [
                                                 {
                                                     'startIndex': '57',
                                                     'length': '3'}]}],
                                     'textSegments': [{
                                         'startIndex': '40',
                                         'length': '20'}],
                                     'orientation': 'ANGLE_0'},
                                     {'boundingBox': {
                                         'vertices': [{'x': '170', 'y': '299'}, {'x': '170', 'y': '333'},
                                                      {'x': '385', 'y': '333'}, {'x': '385', 'y': '299'}]},
                                         'text': 'фотоскана', 'words': [{'boundingBox': {
                                         'vertices': [{'x': '170', 'y': '296'}, {'x': '170', 'y': '337'},
                                                      {'x': '385', 'y': '337'}, {'x': '385', 'y': '296'}]},
                                         'text': 'фотоскана', 'entityIndex': '-1',
                                         'textSegments': [
                                             {'startIndex': '61', 'length': '9'}]}],
                                         'textSegments': [{'startIndex': '61', 'length': '9'}],
                                         'orientation': 'ANGLE_0'}], 'languages': [{'languageCode': 'ru'}],
                                 'textSegments': [{'startIndex': '0', 'length': '70'}]}], 'entities': [], 'tables': [],
                            'fullText': 'Напишу рукописный\nтекст с фото, скрима,\nпечатного текста или\nфотоскана\n',
                            'rotate': 'ANGLE_0'}, 'page': '0'}}
    return json_loads


def test_predict(vision_model, text):
    response = vision_model.predict()
    assert response == BaseCVPrediction(text=text),\
        'Несовпадение текста - модель должна была вернуть другой текст!'


def test_predict_verbose(vision_model, text, json_loads):
    response = vision_model.predict_verbose()
    assert response == BaseCVVerbosePrediction(json_loads=json_loads, text=text), \
        'Несовпадение текста или JSON-а - модель должна была вернуть другой текст!'
