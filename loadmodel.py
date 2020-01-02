import os

import numpy
import yaml
# model.load('E:/workspace/HOG_SVM/saved_models/cifar10_ResNet20v1_model_50.h5')
from keras.engine.saving import load_model
from sklearn import svm
import keras
from keras.datasets import cifar10
import numpy as np
import os

# class Starter:
#     def __init__(self):
#         from cifar10 import Dataloader
#         # 读取配置
#         config_path = os.path.join('configs/resnet.yaml')
#         self.option = yaml.load(open(config_path, 'r'))
#         self.dataloader = Dataloader(
#             data_path=self.option['data_path'],
#             config_path=config_path)
#         # print(self.dataloader.test_images)


# Load the CIFAR10 data.
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Input image dimensions.
input_shape = x_train.shape[1:]

# Normalize data.
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# If subtract pixel mean is enabled
if True:
    x_train_mean = np.mean(x_train, axis=0)
    x_train -= x_train_mean
    x_test -= x_train_mean

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
print('y_train shape:', y_train.shape)

# Convert class vectors to binary class matrices.
# y_train = keras.utils.to_categorical(y_train, 10)
# y_test = keras.utils.to_categorical(y_test, 10)


if __name__ == '__main__':

    model = load_model('saved_models/cifar10_ResNet20v1_model_100.h5')

    clf = svm.SVC(C=0.5, kernel='linear', decision_function_shape='ovo')
    # clf = svm.LinearSVC(C=1.0, class_weight=None, dual=True, fit_intercept=True,
    #                     intercept_scaling=1, loss='squared_hinge', max_iter=1000,
    #                     multi_class='ovr', penalty='l2', random_state=None, tol=0.0001,
    #                     verbose=0)

    train_x = model.predict(x_train)
    train_y = y_train
    clf.fit(train_x, train_y.ravel())
    test_x = model.predict(x_test)
    test_y = y_test
    result = clf.predict(test_x)
    right_res = 0
    for idx in range(len(result)):
        # print("{0}:{1}".format(numpy.where(test_x[idx] == (max(test_x[idx]))), test_y[idx]))
        if result[idx] == test_y[idx]:
            right_res += 1
    print("ACC is : {0} %".format(right_res * 100 / len(test_y)))
    # print(clf.fit(train_x, train_y))
