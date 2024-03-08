import os
from PIL import Image
import numpy as np


def formatImageSize(image):
    resizedImage = image.resize((64, 64), Image.Resampling.LANCZOS)
    # 归一化
    imageNDArray = np.array(resizedImage) / 255
    formattedImage = imageNDArray.reshape((64, 64, 3))

    return formattedImage


def readImageDataFile(folderPath, label):
    images = []
    labels = []

    for imageFileName in os.listdir(folderPath):
        imagePath = os.path.join(folderPath, imageFileName)
        image = Image.open(imagePath).convert('RGB')

        # 将图片缩放到指定大小
        formattedImage = formatImageSize(image)
        images.append(formattedImage)
        labels.append(label)

    imagesNDArray = np.array(images)
    labelsNDArray = np.array(labels)

    return imagesNDArray, labelsNDArray
