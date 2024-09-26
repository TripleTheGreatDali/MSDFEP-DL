import base64
import hashlib
import hmac
import io
import json
import os
from datetime import datetime
import requests
from ultralytics import YOLO
from PIL import Image
import http.client
import ssl


# 获取蛋链视频流
def get_video(camera_id):
    # 示例API请求参数
    method = 'POST'
    path = '*******'
    host = '*******'
    AK = '*******'
    SK = '*******'
    content_type = 'application/json'

    # 构造待签名的字符串，根据错误信息来调整
    string_to_sign = '\n'.join([method, '*/*', content_type, path])

    # 使用SK生成HMAC SHA256签名
    signature = base64.b64encode(hmac.new(SK.encode(), string_to_sign.encode(), hashlib.sha256).digest()).decode()

    # 构造请求头
    headers = {
        'Content-Type': content_type,
        'X-Ca-Key': AK,
        'X-Ca-Signature': signature,
        # 根据错误信息，API似乎不要求在签名中包括正文，但如果需要，请确保包括'Accept'头
        'Accept': '*/*'
    }

    # 请求体数据
    data = {
        "indexCode": camera_id,  # 鸡蛋栋舍
        "transmode": 1,
        "streamType": 0,
        "protocol": "hls",
        "expireTime": -1,
        "expand": "transcode=1&streamform=rtp"
    }

    # 发送POST请求
    response = requests.post('http://' + host + path, headers=headers, data=json.dumps(data))

    # 解析响应内容
    response_data = json.loads(response.text)

    # 提取URL
    stream_url = response_data['data']['url'] if 'data' in response_data and 'url' in response_data['data'] else None

    return stream_url

def send_message(config, url):
    # 动态生成timestamp和token
    timestamp = str(int(datetime.now().timestamp()))
    salt = "bdbf3bdfdabbb5a525e31356d7503ded"  # 根据您的实际情况提供盐值
    hash_obj = hashlib.md5(f"{timestamp}{salt}".encode())
    token = hash_obj.hexdigest()

    # 使用params传递查询参数
    params = {
        "token": token,
        "timestamp": timestamp
    }
    # 构造payload
    payload = config

    headers = {
        'Content-Type': 'application/json'
    }

    response = requests.post(url, params=params, headers=headers, data=payload)

    if response.status_code == 200:
        try:
            return response.json()  # 尝试解析JSON
        except ValueError:
            return {'status': 'error', 'message': 'Invalid JSON format'}
    else:
        return {'status': 'error', 'message': f'Server error with status code {response.status_code}'}

# 获取token
def acquire_token():
    login_url = '*******'
    login_data = {
        "username": "*******'",
        "password": "*******'",
        "code": "*******'"
    }

    # 发送登录请求
    response = requests.post(login_url, data=json.dumps(login_data), verify=False,
                             headers={'Content-Type': 'application/json'})
    login_response = response.json()
    # 授权令牌
    token = login_response["data"]["token"]
    return token

# 获取艾瑞数据
def acquire_data(pageNum):
    # 创建一个不验证证书的SSL上下文
    context = ssl.create_default_context()
    context.check_hostname = False
    context.verify_mode = ssl.CERT_NONE
    # 连接到服务器
    conn = http.client.HTTPSConnection("47.105.91.104", 9443, context=context)  # 使用实际的IP地址或域名

    headers = {
        'Authorization': f'Bearer {acquire_token()}',
        'User-Agent': 'Apifox/1.0.0 (https://apifox.com)'
    }

    params = "*******"+ str(pageNum)
    # 发送GET请求
    conn.request("GET", params, headers=headers)

    # 获取响应
    res = conn.getresponse()
    data = res.read()

    # 打印响应
    return json.loads(data.decode("utf-8"))

# 获取艾瑞图片
def acquire_img(ID):
    # 创建一个不验证证书的SSL上下文
    context = ssl.create_default_context()
    context.check_hostname = False
    context.verify_mode = ssl.CERT_NONE
    conn = http.client.HTTPSConnection("*******'", 9443, context=context)
    payload = ''
    headers = {
        'Authorization': f'Bearer {acquire_token()}',
        'User-Agent': 'Apifox/1.0.0 (https://apifox.com)'
    }
    conn.request("GET", "*******'" + ID, payload, headers)
    res = conn.getresponse()
    data = res.read().decode("utf-8")
    data = json.loads(data)
    return '*******' + data['data'][0]['imageUrl']

