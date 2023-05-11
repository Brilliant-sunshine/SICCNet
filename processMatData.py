# coding=UTF8
from sklearn.model_selection import KFold
import numpy as np

from scipy.io import loadmat
import scipy

# 将WI_real数据分成每个人一个文件，用来进行FBCNet
dataset = "WI_real"
for subject_id in [1,2,3,4,5,6,7,8,9,10]:
    # load original data(preprocessed)
    testData = scipy.io.loadmat("./data/" + dataset + "/original/testdata.mat")
    trainData = scipy.io.loadmat("./data/" + dataset + "/original/traindata.mat")
    DataName = "data" + str(subject_id) + "_X"
    LabelName = "data" + str(subject_id) + "_y"
    traindata = trainData[DataName]
    trainlabel = trainData[LabelName]
    testdata = testData[DataName]
    testlabel = testData[LabelName]

    testlabel[testlabel<0] = 0
    trainlabel[trainlabel<0] = 0

    data = np.concatenate([traindata, testdata], axis=2)
    label = np.concatenate([trainlabel, testlabel], axis=1)

    scipy.io.savemat("E:\lzg1\对比算法\FBCNet-master-WI_real\data\\3a\\rawMat\S000"+str(subject_id)+".mat",
                     {'x':data, 'y':label})
    print(subject_id)
