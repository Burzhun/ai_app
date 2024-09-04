import base64
import requests
import subprocess
import json


def get_iam_token():
    """
    Метод для получения IAM-токена

    Returns:
        IAM-Token

    """
    output = subprocess.check_output("yc iam create-token", shell=True).decode('UTF-8').rstrip()
    return output


# YA_DIR_ID нужно подставить свою, от продовой учетки
def api_request(file_name='sample_img.jpg', YA_DIR_ID='b1ghkpruumbocmh45sf2') -> json:
    """
    Метод для обращения к API Yandex vision OCR
    Returns:
        JSON-Response полученный yandex vision OCR
    """
    IAM_TOKEN = get_iam_token()
    with open(file_name, "rb") as f:
        encoded_image = base64.b64encode(f.read())

    json_body = f'{{ \n \t "mimeType": "JPEG", \n \
        "languageCodes": ["ru","en"], \n \
        "model": "handwritten", \n \
        "content": "{encoded_image.decode("utf-8")}"}}'

    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {IAM_TOKEN}',
        'x-folder-id': f'{YA_DIR_ID}',
        'x-data-logging-enabled': 'true',
    }

    response = requests.post('https://ocr.api.cloud.yandex.net/ocr/v1/recognizeText', headers=headers,
                             data=json_body)
    json_loads = json.loads(response.text)
    return json_loads


def json_to_text(json_loads) -> str:
    """
   Метод для извлечения текста из JSON

      Returns:
          Строка текста из JSON
   """
    res = """"""
    for i in json_loads["result"]['textAnnotation']['fullText']:
        res += i
    return res

