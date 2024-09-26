import hashlib
import shutil
import sqlite3
import zipfile
import time
from flask_socketio import SocketIO, emit
from ultralytics import YOLO
import threading
from PIL import Image
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
from functools import wraps
import os
from utils.Test_model import *

from flask_cors import CORS
app = Flask(__name__)
app.secret_key = 'your_secret_key'
socketio = SocketIO(app, cors_allowed_origins="*")
CORS(app)

# 定义装饰器
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if session.get('logged_in') != True:
            return redirect(url_for('login'))
        return f(*args, **kwargs)

    return decorated_function


# 提前运行
@app.before_request
def before_request():
    if 'logged_in' not in session:
        session['logged_in'] = False


# 登录界面
@app.route('/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username == 'admin' and password == 'admin':
            session['logged_in'] = True  # 设置 session 标志为已登录
            return redirect(url_for('index'))  # 登录成功，重定向到 main 页面
        else:
            flash('Incorrect password or username, please try again！')
            return redirect(url_for('login', error='true'))
    return render_template('login.html')


# 训练主界面
@app.route('/index')
@login_required
def index():
    return render_template('index.html')


# 模型图片测试
@app.route('/index/model_img_test')
@login_required
def model_img_test():
    # 连接到 SQLite 数据库
    # conn = sqlite3.connect('Sort/Egg_Count/SQL/egg_count_datasets.db')
    # # 创建游标对象
    # cur = conn.cursor()
    # # 执行查询
    # cur.execute("SELECT * FROM egg_count_model")
    # # 获取所有数据
    # models = cur.fetchall()
    # # 关闭连接
    # conn.close()
    return render_template('model_img_test.html')


@app.route('/index/model_img_test/recognize', methods=['POST'])
@login_required
def model_img_test_recognize():
    img = request.files['file']
    img_source = Image.open(img.stream).convert('RGB').resize((900, 600), Image.LANCZOS)

    # 读取其他表单字段
    confidence_degree = float(int(request.form.get('Confidence_degree')) / 100)

    result_text, img_url, crop_img_url = model_recognize(img_source, confidence_degree)
    return {'status': 'success', 'result_text': result_text, 'img_url': img_url, 'crop_img_url': crop_img_url}


@app.route('/logout')
def logout():
    session.pop('logged_in', None)
    return redirect(url_for('login'))


# 服务器端的SALT应该与客户端一致
SALT = '*******' # Note 


def verify_token(client_token, client_timestamp):
    """验证客户端的Token是否有效"""
    current_time = time.time()
    if abs(current_time - int(client_timestamp)) > 300:  # 允许5分钟的时间差异
        return False
    expected_token = hashlib.md5((client_timestamp + SALT).encode()).hexdigest()
    return expected_token == client_token


@app.route('/test')
def test():
    # 示例数据集
    data = {
        'labels': [i for i in range(1, 101)],  # X轴标签
        'datasets': [
            {'label': 'Loss 1', 'data': [i ** 0.5 for i in range(1, 101)], 'borderColor': 'red', 'fill': False},
            {'label': 'Loss 2', 'data': [i ** 0.4 for i in range(1, 101)], 'borderColor': 'green', 'fill': False},
            {'label': 'Loss 3', 'data': [i ** 0.3 for i in range(1, 101)], 'borderColor': 'blue', 'fill': False}
        ]
    }
    return "aaaaa"


if __name__ == '__main__':
    app.config['DEBUG'] = True
    app.run(debug=True, host='0.0.0.0', port=5000)
