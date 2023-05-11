# coding=UTF8
import warnings
warnings.filterwarnings('ignore')
from braindecode.torch_ext.util import np_to_var
import torch.optim as optim
from torch.autograd import Variable
from sklearn import metrics
from until.ConvNet import ShallowFBCSPNet
from until.eegnet import eegnet
from until.deep4_net import Deep4Net_base
from until.constraints import MaxNormDefaultConstraint
from until.constraints_EEGNet import MaxNormDefaultConstraint_EEGNet
from until.validation import plot_confusion_matrix
import argparse
from until.loadData import *
from cutmix import *
from loguru import logger

parser = argparse.ArgumentParser(description='input the dataset dir path.')

parser.add_argument('--backbone_net_all', type=str, default=['shallow'], help='choose model: shallow, DeepNet, EEGNet')
parser.add_argument('--test_interval', type=int, default=1, help='iter')
parser.add_argument('--max_epochs', type=int, default=1500)
parser.add_argument('--subject', type=int, default=[1], help='3a=[1 2 3]')
parser.add_argument('--dataset', type=str, default="3a", help='choose: 2a, 3a, WI_real')
parser.add_argument('--foldname', type=str, default='ConfusionMatrix.log', help='resultFileName')
parser.add_argument('--model_folder', type=str, default="Shallow")
parser.add_argument('--tsne_path', type=str, default="E:\lzg1\image\\tsne\\2a\\shallow")
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

import os
if not os.path.exists("./result/"+args.dataset+"/"):
    os.mkdir("./result/"+args.dataset+"/")
logfile = "./result/"+args.dataset+"/"+args.foldname
logger.remove(handler_id=None)
logger.add(logfile, encoding='utf_8')

logger.info(args)

optim_dict = {"Adam": optim.Adam, "SGD": optim.SGD}
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cuda = True

torch.cuda.set_device(0)

for backbone_net in args.backbone_net_all:
    for subject_id in args.subject:

        testData = scipy.io.loadmat("./splitData/"+args.dataset+"/original/S" + str(subject_id) + "Test.mat")
        trainData = scipy.io.loadmat("./splitData/"+args.dataset+"/original/S" + str(subject_id) + "Train.mat")
        fold = 0

        for index, fold in enumerate(args.folds[subject_id-1]):
            for IS in args.Is[subject_id-1]:
                path = "E:\lzg1\model_pth\\" + args.model_folder + "\\" + args.dataset + "\S" + str(subject_id) + \
                       "\S" + str(subject_id) + "F" + str(fold) + "epoch_" + str(args.max_epochs - IS) + ".pth"

                print("Dataset = "+str(args.dataset)+" ; "+str(args.folds) + " folds cross-validation: Subject=" + str(subject_id) + ", fold=" + str(fold))

                # train data
                trainDataName = "S" + str(subject_id) + "F" + str(fold) + "Data"
                trainLabelName = "S" + str(subject_id) + "F" + str(fold) + "Label"
                traindata = torch.from_numpy(np.transpose(trainData[trainDataName], (2, 1, 0)))

                if(args.dataset == "WI_real"):
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
                    testlabel = torch.from_numpy(np.transpose(testData[testLabelName],(1,0)))
                else:
                    testlabel = torch.from_numpy(testData[testLabelName])

                X_test = Variable(testdata)
                X_test = torch.unsqueeze(X_test, dim=3)
                testinglabel = Variable(testlabel)

                # set base network
                classes = torch.unique(testlabel, sorted=False).numpy()
                n_classes = len(classes)
                n_chans = int(testdata.shape[1])
                input_time_length = testdata.shape[2]

                if backbone_net == 'shallow':
                    model = ShallowFBCSPNet(n_chans, n_classes, input_time_length=input_time_length,
                                            final_conv_length='auto')
                elif backbone_net == 'DeepNet':
                    model = Deep4Net_base(n_chans, n_classes, input_time_length=input_time_length,
                                          final_conv_length='auto')
                elif backbone_net == 'EEGNet':
                    kernel_length = 125
                    model = eegnet(n_classes=n_classes, in_chans=n_chans, input_time_length=input_time_length,
                                   kernel_length=kernel_length)
                if cuda:
                    model.cuda()

                if backbone_net == 'EEGNet':
                    model_constraint = MaxNormDefaultConstraint_EEGNet()
                else:
                    model_constraint = MaxNormDefaultConstraint()

                model.load_state_dict(torch.load(path))
                with torch.no_grad():
                    model.eval()
                    # test in the train
                    b_x = X_test.float()
                    b_y = testinglabel.long()
                    input_vars = np_to_var(b_x, pin_memory=False).float()
                    labels_vars_test = np_to_var(b_y, pin_memory=False).type(torch.LongTensor)
                    input_vars = input_vars.cuda()
                    labels_vars_test = labels_vars_test.cuda()

                    outputs, f, _ = model(input_vars)

                    y_test_pred = torch.max(outputs, 1)[1].cpu().data.numpy().squeeze()

                    acc = metrics.accuracy_score(labels_vars_test.cpu().data.numpy(), y_test_pred)
                    recall = metrics.recall_score(labels_vars_test.cpu().data.numpy(), y_test_pred,
                                                  average='macro')
                    f1 = metrics.f1_score(labels_vars_test.cpu().data.numpy(), y_test_pred, average='macro')
                    preci = metrics.precision_score(labels_vars_test.cpu().data.numpy(), y_test_pred,
                                                    average='macro')
                    kappa = metrics.cohen_kappa_score(labels_vars_test.cpu().data.numpy(), y_test_pred)

                    confusion_matrix = metrics.confusion_matrix(labels_vars_test.cpu().data.numpy(), y_test_pred)
                    plot_confusion_matrix(epoch=IS, cm=confusion_matrix, fold=fold, dataset=args.dataset,
                                            subject=subject_id)