# -*- coding: utf-8 -*-
"""
   The trained model is used here to obtain test results.
"""
import warnings

from braindecode.visualization.perturbation import compute_amplitude_prediction_correlations

warnings.filterwarnings('ignore')
from braindecode.torch_ext.util import np_to_var, var_to_np
import time
from torch.autograd import Variable
from until.ConvNet import ShallowFBCSPNet
import argparse
from until.loadData import *
from until.plot_head import *

parser = argparse.ArgumentParser(description='input the dataset dir path.')

parser.add_argument('--backbone_net_all', type=str, default=['shallow'], help='choose model: shallow, DeepNet, EEGNet')
parser.add_argument('--test_interval', type=int, default=1, help='iter')
parser.add_argument('--max_epochs', type=int, default=1500)
parser.add_argument('--subject', type=int, default=[], help='3a=[1 2 3]')
parser.add_argument('--dataset', type=str, default="2a", help='choose: 2a, 3a, WI_real')
parser.add_argument('--foldname', type=str, default='plotHead.log', help='resultFileName')
parser.add_argument('--model_folder_shallow', type=str, default="shallow")
parser.add_argument('--model_folder_SICCNet', type=str, default="")
parser.add_argument('--head_path', type=str, default="E:\image\\tsne\\2a\\head")
parser.add_argument('--Is', type=int, default=[[], # S1
                                              [],  # S2
                                              [],  # S3
                                              [],  # S4
                                              [],  # S5
                                              [],  # S6
                                              [],  # S7
                                              [],  # S8
                                              []]) # S9
parser.add_argument('--folds', type=int, default=[[],  # S1
                                                  [],  # S2
                                                  [],  # S3
                                                  [],  # S4
                                                  [],  # S5
                                                  [],  # S6
                                                  [],  # S7
                                                  [],  # S8
                                                  []]) # S9
args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cuda = True

torch.cuda.set_device(0)

time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

map_avg = []
map2 = []
trainTest = False
for backbone_net in args.backbone_net_all:
    for subject_id in args.subject:
        testData = scipy.io.loadmat("./splitData/" + args.dataset + "/original/S" + str(subject_id) + "Test.mat")
        trainData = scipy.io.loadmat("./splitData/" + args.dataset + "/original/S" + str(subject_id) + "Train.mat")

        for index, fold in enumerate(args.folds[subject_id - 1]):

            for _, after_num in enumerate(args.Is[subject_id - 1]):

                path_shallow = "E:\model_pth\\" + args.model_folder_shallow + "\\" + args.dataset + "\S" + str(subject_id) + \
                               "\S" + str(subject_id) + "F" + str(fold) + "epoch_" + str(
                                args.max_epochs - args.Is[subject_id - 1][after_num-1]) + ".pth"
                path_SICCNet = "E:\model_pth\\" + args.model_folder_SICCNet + "\\" + args.dataset + "\S" + str(subject_id) + \
                               "\S" + str(subject_id) + "F" + str(fold) + "epoch_" + str(
                    args.max_epochs - args.Is[subject_id - 1][after_num-1]) + ".pth"

                after_n = args.max_epochs - args.Is[subject_id - 1][after_num-1]


                print(
                    "Dataset = " + str(args.dataset) + " ; " + str(args.folds) + " folds cross-validation: Subject=" + str(
                        subject_id) + ", fold=" + str(fold))

                # train data
                trainDataName = "S" + str(subject_id) + "F" + str(fold) + "Data"
                trainLabelName = "S" + str(subject_id) + "F" + str(fold) + "Label"
                traindata = torch.from_numpy(np.transpose(trainData[trainDataName], (2, 1, 0)))

                if (args.dataset == "WI_real"):
                    trainlabel = torch.from_numpy(np.transpose(trainData[trainLabelName], (1, 0)))
                else:
                    trainlabel = torch.from_numpy(trainData[trainLabelName])

                DataName = "F" + str(fold) + "Data"
                LabelName = "F" + str(fold) + "Label"

                # test data
                testDataName = "S" + str(subject_id) + "F" + str(fold) + "Data"
                testLabelName = "S" + str(subject_id) + "F" + str(fold) + "Label"
                testdata = torch.from_numpy(np.transpose(testData[testDataName], (2, 1, 0)))
                if (args.dataset == "WI_real"):
                    testlabel = torch.from_numpy(np.transpose(testData[testLabelName], (1, 0)))
                else:
                    testlabel = torch.from_numpy(testData[testLabelName])

                X_test = Variable(testdata)
                X_test = torch.unsqueeze(X_test, dim=3)
                testinglabel = Variable(testlabel)

                allData = torch.cat([traindata,testdata], dim=0)
                allLabel = torch.cat([trainlabel, testlabel], dim=0)
                if (trainTest):
                    X_train = Variable(traindata)
                    X_train = torch.unsqueeze(X_train, dim=3)
                    X_train_label = Variable(trainlabel)

                # set base network
                classes = torch.unique(testlabel, sorted=False).numpy()
                n_classes = len(classes)
                n_chans = int(testdata.shape[1])
                input_time_length = testdata.shape[2]

                if backbone_net == 'shallow':
                    model_shallow = ShallowFBCSPNet(n_chans, n_classes, input_time_length=input_time_length,
                                            final_conv_length='auto')
                    model_SICCNet = ShallowFBCSPNet(n_chans, n_classes, input_time_length=input_time_length,
                                                final_conv_length='auto')
                if cuda:
                    model_shallow.cuda()
                    model_SICCNet.cuda()

                model_shallow.eval()
                model_SICCNet.eval()
                b_x = X_test.float()
                b_y = testinglabel.long()
                model_shallow.load_state_dict(torch.load(path_shallow))

                model_SICCNet.load_state_dict(torch.load(path_SICCNet))

                test_set = data(allData.float(), allLabel)
                iterator = BalancedBatchSizeIterator(batch_size=40)
                train_batches = list(iterator.get_batches(test_set, shuffle=False))
                train_X_batches = np.concatenate(list(zip(*train_batches))[0])
                pred_fn = lambda x: var_to_np(model_shallow(np_to_var(x).cuda()))
                amp_pred_corrs = compute_amplitude_prediction_correlations(pred_fn, train_X_batches, n_iterations=12,
                                                                           batch_size=50)
                map_avg.append(amp_pred_corrs)

                pred_fn1 = lambda x: var_to_np(model_SICCNet(np_to_var(x).cuda()))
                amp_pred_corrs = compute_amplitude_prediction_correlations(pred_fn1, train_X_batches, n_iterations=12,
                                                                           batch_size=50)
                map2.append(amp_pred_corrs)

                plot_head_show(test_set, map_avg, map2, subject_id, fold, after_n)