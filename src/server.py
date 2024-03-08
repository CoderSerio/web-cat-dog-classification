#
from flask import Flask, request, jsonify
from flask_cors import CORS
from util import formatImageSize
import tensorflow as tf  # 如果模型是TensorFlow保存的，需要导入tf
import json
from PIL import Image
from io import BytesIO
import numpy as np


app = Flask(__name__)
CORS(app)  # 在全局启用CORS


model = tf.keras.models.load_model(
    'models/tra(0.9285)-val(0.8110).keras')  # TensorFlow模型示例


@app.route('/predict', methods=['POST'])
def predict():
    # 接收前端发送的请求数据，处理成二进制
    data = request.files['image']
    print('这是什么', data)

    dataBytes = BytesIO()
    data.save(dataBytes)
    dataBytes.seek(0)
    image = Image.open(dataBytes)
    formattedImage = formatImageSize(image)
    imageList = []
    imageList.append(formattedImage)
    # print('这又是什么', formatImage.shape)
    prediction = model.predict(np.array(imageList))
    # 返回预测结果
    return json.dumps({'prediction': prediction.tolist()})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
