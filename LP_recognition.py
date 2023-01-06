import base64
import requests

API_KEY = "ZTsSxXhOflazTWsbmgNpKqeD"
SECRET_KEY = "1IYWWGAkiR3j1xQHoHAdgt3sQlwqQv5K"

'''
车牌识别
'''


def fetch_token():
    host = 'https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id=' + API_KEY + '&client_secret=' + SECRET_KEY
    response = requests.get(host)
    if response:
        result = response.json()
        return result['access_token']


def read_file(image_path1):
    f1 = open(image_path1, 'rb')
    return f1.read()


def license_match(token, picture_file1):
    img1 = base64.b64encode(picture_file1).decode('utf-8')
    # 注意要用 json.dumps
    params = {"image": img1}
    access_token = token
    request_url = "https://aip.baidubce.com/rest/2.0/ocr/v1/license_plate"
    request_url = request_url + '?access_token=' + access_token
    headers = {'content-type': 'application/x-www-form-urlencoded'}
    response = requests.post(request_url, data=params, headers=headers)
    if response:
        return response.json()


def recognition(url):
    file = read_file(url)
    token = fetch_token()
    result_json = license_match(token, file)
    result = result_json.get('words_result', {})

    return result


if __name__ == '__main__':
    token = fetch_token()
    picture_file1 = read_file('./img/car.jpg')
    result_json = license_match(token, picture_file1)
    result = result_json.get('words_result', {})
    print(result.get('number', ""))
