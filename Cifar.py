import pickle


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def testUnpickle():
    dict = unpickle("image/data_batch_1")
    print(dict.keys())
    print(dict[b'labels'])
