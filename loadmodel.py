import os
import platform

import joblib
import numpy
import glob
import yaml
import time


# model.load('E:/workspace/HOG_SVM/saved_models/cifar10_ResNet20v1_model.h5')
from keras.engine.saving import load_model
from sklearn.svm import LinearSVC


class Starter:

    # 训练和测试
    def train_and_test(self):
        t0 = time.time()
        correct_number = 0
        total = 0
        # 下面的代码是加载模型  可以注释上面的代码   直接进行加载模型  不进行训练
        clf = joblib.load('cifar10_%s_model_ResNet20v1.h5')
        print("训练之后的模型存放在model文件夹中")
        # exit()
        result_list = []
        for feat_path in glob.glob(os.path.join(test_feat_path, '*.feat')):
            total += 1
            if platform.system() == 'Windows':
                symbol = '\\'
            else:
                symbol = '/'
            image_name = feat_path.split(symbol)[1].split('.feat')[0]
            data_test = joblib.load(feat_path)
            data_test_feat = data_test[:-1].reshape((1, -1)).astype(numpy.np.float64)
            result = clf.predict(data_test_feat)
            result_list.append(image_name + ' ' + label_map[int(result[0])] + '\n')
            if int(result[0]) == int(data_test[-1]):
                correct_number += 1
        write_to_txt(result_list)
        rate = float(correct_number) / total
        t1 = time.time()
        print('准确率是： %f' % rate)
        print('耗时是 : %f' % (t1 - t0))

    def __init__(self):
        from cifar10 import Dataloader
        # 读取配置
        config_path = os.path.join('configs/resnet.yaml')
        self.option = yaml.load(open(config_path, 'r'))
        self.dataloader = Dataloader(
            data_path=self.option['data_path'],
            config_path=config_path)
        # print(self.dataloader.test_images)


if __name__ == '__main__':
    model = load_model('saved_models/cifar10_ResNet20v1_model.h5')
    starter = Starter()
    clf = LinearSVC()
    train_x = model.predict(numpy.reshape(starter.dataloader.train_images, (45000, 32, 32, 3)))
    train_y = starter.dataloader.train_labels
    clf.fit(train_x, train_y)
    test_x = model.predict(numpy.reshape(starter.dataloader.test_images, (10000, 32, 32, 3)))
    test_y = starter.dataloader.test_labels
    result = clf.predict(test_x)
    right_res = 0
    for idx in range(len(result)):
        if result[idx] == test_y[idx]:
            right_res += 1
    print("ACC : {0}".format(right_res/len(test_y)))
    # print(clf.fit(train_x, train_y))
