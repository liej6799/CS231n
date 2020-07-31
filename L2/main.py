import matplotlib.pyplot as plt
import numpy as np
import method
from cs231n.data_utils import load_CIFAR10
from cifar10_web import cifar10

print('Prep to load data')
Xtrain, Ytrain, Xtest, Ytest = load_CIFAR10(
    'D:\Datasets\Cifar 10\cifar-10-batches-py')
print('Load data success')
# Xtr_rows becomes 50000 x 3072
Xtr_rows = Xtrain.reshape(Xtrain.shape[0], 32 * 32 * 3)
# Xte_rows becomes 10000 x 3072
Xte_rows = Xtest.reshape(Xtest.shape[0], 32 * 32 * 3)
print('Data reshape success')
NearestNeighbor = method.NearestNeighbor()

print('Start train')

NearestNeighbor.train(Xtr_rows, Ytrain)
print('Train success')


Yval_predict = NearestNeighbor.predict(Xte_rows[:100])
# print('predict success')
# print('accuracy: %f' % (np.mean(Yval_predict == Ytest[:100])))
