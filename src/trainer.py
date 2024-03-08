
import tensorflow as tf
import numpy as np
from util import readImageDataFile

xTrainCat, yTrainCat = readImageDataFile('data/train/cat', 0)
xTrainDog, yTrainDog = readImageDataFile('data/train/dog', 1)
xValidationCat, yValidationCat = readImageDataFile('data/validate/cat', 0)
xValidationDog, yValidationDog = readImageDataFile('data/validate/dog', 1)


xTrain = np.concatenate([xTrainCat, xTrainDog])
yTrain = tf.keras.utils.to_categorical(np.concatenate([yTrainCat, yTrainDog]))
xTest = np.concatenate([xValidationCat, xValidationDog])
yTest = tf.keras.utils.to_categorical(
    np.concatenate([yValidationCat, yValidationDog]))

# if tf.config.list_physical_devices('GPU'):
#     with tf.device('/GPU:0'):
print('\n\nSTART!!!\n\n')
# VGG-like
model = tf.keras.models.Sequential([
    # 第一层卷积 + 池化
    tf.keras.layers.Conv2D(
        32, (3, 3),
        activation='relu',
        input_shape=(64, 64, 3),
    ),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

    # 第二层卷积 + 池化
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu',
                           kernel_regularizer=tf.keras.regularizers.l2(0.0002)),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

    # 第三层卷积 + 池化
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu',
                           kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

    # 扁平化并加入全连接层
    tf.keras.layers.Flatten(),

    # 全连接层
    tf.keras.layers.Dense(512, activation='relu',
                          kernel_regularizer=tf.keras.regularizers.l2(0.003)),
    # dropout
    tf.keras.layers.Dropout(0.4),

    # 输出层，假设类别数为2（猫和狗）
    tf.keras.layers.Dense(2, activation='sigmoid')
])

# 编译模型，假设损失函数选用 categorical_crossentropy，优化器使用 Adam
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

info = model.fit(
    xTrain,
    yTrain,
    verbose=1,
    epochs=60,
    batch_size=15,
    validation_split=0.2,
    validation_data=(xTest, yTest),
    callbacks=[tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=15, restore_best_weights=True)]
)

acc = info.history['accuracy'][-1]
valAcc = info.history['val_accuracy'][-1]

model.save(f'models/tra({acc:.4f})-val({valAcc:.4f}).keras')
print('ALL DONE')
# else:
#     print('\n\nNOT SUPPORT GPU!!!\n\n')
