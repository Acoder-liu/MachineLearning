# python-38
# ------------------------------------------------
# 作者     :刘想
# 时间     :2021/6/5 16:30
# ------------------------------------------------
# -*- coding:utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model, load_model
from keras.datasets import mnist
import time  # 引入time模块


def get_data():
    (trainSet, _), (testSet, _) = mnist.load_data()
    trainSet = trainSet.astype('float32') / 255.
    testSet = testSet.astype('float32') / 255.
    trainSet = np.reshape(trainSet, (len(trainSet), 28, 28, 1))
    testSet = np.reshape(testSet, (len(testSet), 28, 28, 1))
    print('x_train.shape:', trainSet.shape)
    print('x_test.shape:', testSet.shape)
    return trainSet, testSet


def add_noise(train, test):
    """随机添加噪音"""
    ticks = [5, 6, 2, 34, 12, 34, 66, 74]
    np.random.seed(655)
    noise_factor = 0.5
    trainNoisy = train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=train.shape)
    testNoisy = test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=test.shape)
    # 值仍在0-1之间
    trainNoisy = np.clip(trainNoisy, 0., 1.)
    testNoisy = np.clip(testNoisy, 0., 1.)
    return trainNoisy, testNoisy


def buildModel(train_noisy, test_noisy):
    """去燥"""
    input_img = Input(shape=(28, 28, 1,))  # N * 28 * 28 * 1
    # 实现 encoder 部分，由两个 3 × 3 × 32 的卷积和两个 2 × 2 的最大池化组 成。
    x = Conv2D(32, (3, 3), padding='same', activation='relu')(input_img)  # 28 * 28 * 32
    encoded = MaxPooling2D((2, 2), padding='same')(x)  # 14 * 14 * 32
    # x = Conv2D(32, (3, 3), padding='same', activation='relu')(x)  # 14 * 14 * 32
    # encoded = MaxPooling2D((2, 2), padding='same')(x)  # 7 * 7 * 32
    # 实现 decoder 部分，由两个 3 × 3 × 32 的卷积和两个 2 × 2 的上采样组成。
    # 7 * 7 * 32
    x = Conv2D(32, (3, 3), padding='same', activation='relu')(encoded)  # 7 * 7 * 32
    x = UpSampling2D((2, 2))(x)  # 14 * 14 * 32
    # x = Conv2D(32, (3, 3), padding='same', activation='relu')(x)  # 14 * 14 * 32
    # x = UpSampling2D((2, 2))(x)  # 28 * 28 * 32
    decoded = Conv2D(1, (3, 3), padding='same', activation='sigmoid')(x)  # 28 * 28 *

    autoEncoder = Model(input_img, decoded)
    autoEncoder.compile(optimizer='adadelta', loss='binary_crossentropy')

    autoEncoder.fit(train_noisy, train,
                    epochs=200,
                    batch_size=128,
                    shuffle=True,
                    validation_data=(test_noisy, test))

    autoEncoder.save('autoEncoder.h5')


def remove_noisy(test_noisy):
    autoEncoder = load_model('autoEncoder.h5')
    decodedImages = autoEncoder.predict(test_noisy)
    return decodedImages


def plot1(x_data):
    """画图"""
    n = 10
    plt.figure(figsize=(20, 2))
    for i in range(n):
        ax = plt.subplot(1, n, i + 1)
        plt.imshow(x_data[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()


def plot2(test_noisy, image2):
    """画图"""
    n = 10
    plt.figure(figsize=(20, 4))
    for i in range(n):
        # display original
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(test_noisy[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(image2[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()


train, test = get_data()
trainNoisy, testNoisy = add_noise(train, test)
buildModel(trainNoisy, testNoisy)
decoded_images = remove_noisy(testNoisy)
plot2(testNoisy, decoded_images)
