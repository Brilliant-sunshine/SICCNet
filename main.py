# coding=UTF8
import warnings
warnings.filterwarnings('ignore')
from until.moment_update import moment_update_ema
from braindecode.torch_ext.util import np_to_var
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset
from sklearn import metrics
from tqdm import trange
from until.ConvNet import ShallowFBCSPNet
from until.eegnet import eegnet
from until.deep4_net import Deep4Net_base
import until.lr_schedule as lr_schedule
from until.constraints import MaxNormDefaultConstraint
from loss import losses_supcon, losses_supcon_queue
import argparse
from until.loadData import *
from until.tsne import tsne
from until.validation import plot_confusion_matrix
from cutcat import *
from loguru import logger

parser = argparse.ArgumentParser(description='input the dataset dir path.')

parser.add_argument('--backbone_net_all', type=str, default=['shallow'], help='choose model: shallow, DeepNet, EEGNet')
parser.add_argument('--test_interval', type=int, default=1, help='iter')
parser.add_argument('--max_epochs', type=int, default=1500)
parser.add_argument('--folds', type=int, default=10, help='x fold cross validation')
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--subject', type=int, default=[1,2,3], help='3a=[1 2 3]')
parser.add_argument('--dataset', type=str, default="3a", help='choose: 2a, 3a, WI_real')
parser.add_argument('--foldname', type=str, default='result.log', help='resultFileName')

parser.add_argument('--alpha1', type=float, default=0.01, help='parameter of co-contrastive loss')
parser.add_argument('--alpha2', type=float, default=4, help='parameter of mse_loss')
parser.add_argument('--nSeg', type=int, default=4)
parser.add_argument('--alpha', type=int, default=2.0)
parser.add_argument('--queue', type=int, default=64, help='length of queue instance contrastive')
parser.add_argument('--isQueue', type=bool, default=False)
parser.add_argument('--isCollaborative', type=bool, default=False)
parser.add_argument('--is2DA', type=bool, default=False)

parser.add_argument('--isSaveModel', type=bool, default=False)
parser.add_argument('--model_folder', type=str, default="SICCNet-Model")

parser.add_argument('--isNA', type=bool, default=False)
parser.add_argument('--isWS', type=bool, default=False)
parser.add_argument('--isGAN', type=bool, default=False)

parser.add_argument('--tsne', type=bool, default=False)
parser.add_argument('--confusion', type=bool, default=False)


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

epo_acc_train = list()
epo_acc = list()
epo_recall = list()
epo_f1 = list()
epo_preci = list()
epo_kappa = list()

ACC = list()
kappa_all = list()
recall_all = list()
preci_all = list()
F1_all = list()

trainTest = False

for backbone_net in args.backbone_net_all:
    for subject_id in args.subject:

        testData = scipy.io.loadmat("./splitData/"+args.dataset+"/original/S"+str(subject_id)+"Test.mat")
        trainData = scipy.io.loadmat("./splitData/"+args.dataset+"/original/S"+str(subject_id)+"Train.mat")

        for fold in range(args.folds):
            fold = fold + 1
            print(str(args.folds) + " folds cross-validation: Subject=" + str(subject_id) + ", fold=" + str(fold))

            DataName = "F"+str(fold)+"Data"
            LabelName = "F"+str(fold)+"Label"

            trainDataName = "S" + str(subject_id) + "F" + str(fold) + "Data"
            trainLabelName = "S" + str(subject_id) + "F" + str(fold) + "Label"
            traindata = torch.from_numpy(np.transpose(trainData[trainDataName], (2, 1, 0)))
            if (args.dataset == "WI_real"):
                trainlabel = torch.from_numpy(np.transpose(trainData[trainLabelName], (1, 0)))
            else:
                trainlabel = torch.from_numpy(trainData[trainLabelName])

            X_train = Variable(torch.unsqueeze(traindata, dim=3))
            trainlabel = trainlabel.reshape((-1, 1))
            X_train_label = trainlabel
            trainset = TensorDataset(X_train, X_train_label)
            train_iterator = DataLoader(dataset=trainset, batch_size=args.batch_size, shuffle=True, drop_last=True)

            # test data
            testDataName = "S"+str(subject_id)+"F"+str(fold)+"Data"
            testLabelName = "S"+str(subject_id)+"F"+str(fold)+"Label"
            testdata = torch.from_numpy(np.transpose(testData[testDataName], (2, 1, 0)))
            testlabel = torch.from_numpy(np.transpose(testData[testLabelName],(1,0)))

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
                model_ema = ShallowFBCSPNet(n_chans, n_classes, input_time_length=input_time_length,
                                            final_conv_length='auto')
            elif backbone_net == 'DeepNet':
                model = Deep4Net_base(n_chans, n_classes, input_time_length=input_time_length,
                                      final_conv_length='auto')
                model_ema = Deep4Net_base(n_chans, n_classes, input_time_length=input_time_length,
                                      final_conv_length='auto')
            elif backbone_net == 'EEGNet':
                kernel_length = 125
                model = eegnet(n_classes=n_classes, in_chans=n_chans, input_time_length=input_time_length,
                               kernel_length=kernel_length)
                model_ema = eegnet(n_classes=n_classes, in_chans=n_chans, input_time_length=input_time_length,
                               kernel_length=kernel_length)

            if cuda:
                model.cuda()
                model_ema.cuda()

            for param in model_ema.parameters():
                param.detach_()

            model_constraint = MaxNormDefaultConstraint()

            cl_loss = nn.CrossEntropyLoss()

            if(args.isQueue):
                loss_shallow = losses_supcon_queue.SupConLoss(k=args.queue)
                loss_shallow_ema = losses_supcon_queue.SupConLoss(k=args.queue)
            else:
                loss_shallow = losses_supcon.SupConLoss()
                loss_shallow_ema = losses_supcon.SupConLoss()

                test_accuracy = np.zeros(shape=[0], dtype=float)
                test_recall = np.zeros(shape=[0], dtype=float)
                test_f1 = np.zeros(shape=[0], dtype=float)
                test_preci = np.zeros(shape=[0], dtype=float)
                test_kappa = np.zeros(shape=[0], dtype=float)
                test_loss = np.zeros(shape=[0], dtype=float)

            parameter_list = [{"params": model.parameters()}]
            optimizer = optim_dict["Adam"](parameter_list, **({"lr": 0.001, "weight_decay": 0}))

            schedule_param = {"init_lr": 0.001, "gamma": 10, "power": 0.75}
            lr_scheduler = lr_schedule.schedule_dict["inv"]

            for i in trange(args.max_epochs):
                model.train()
                model_ema.train()
                optimizer = lr_scheduler(optimizer, i / args.max_epochs, **schedule_param)

                for step, batch in enumerate(train_iterator):
                    inputs, labels = batch
                    optimizer.zero_grad()

                    input_vars, labels_vars = Variable(inputs).cuda().float(), Variable(labels).cuda().long()

                    input_vars_ema = cutcat_mixup_random(input_vars, labels_vars, args.alpha, args.nSeg)

                    if(args.isCollaborative):
                        if(args.is2DA):
                            input_vars_ema_1 = cutcat_mixup_random(input_vars, labels_vars, args.alpha, args.nSeg)
                            input = torch.cat((input_vars_ema_1, input_vars_ema), dim=0)
                        else:
                            input = torch.cat((input_vars, input_vars_ema), dim=0)
                        preds, feature, _ = model(input)
                        preds_ema, feature_ema, _ = model_ema(input)
                    else:
                        preds, feature, _ = model(input_vars)
                        preds_ema, feature_ema, _ = model_ema(input_vars_ema)


                    preds_ema = preds_ema.detach()
                    feature_ema = feature_ema.detach()

                    l2_norm = torch.sqrt(torch.sum(feature ** 2, dim=1))
                    features = feature / torch.unsqueeze(l2_norm, dim=1)
                    l2_norm_2 = torch.sqrt(torch.sum(feature_ema ** 2, dim=1))
                    feature_ema = feature_ema / torch.unsqueeze(l2_norm_2, dim=1)

                    if(args.isCollaborative):
                        feature_ori, feature_aug = torch.split(features, int(input_vars.shape[0]), dim=0)
                        feature_ema_ori, feature_ema_aug = torch.split(feature_ema, int(input_vars.shape[0]), dim=0)
                        feature_cat1 = torch.cat(
                                [torch.unsqueeze(feature_ori, dim=1), torch.unsqueeze(feature_ema_aug, dim=1)], dim=1)
                        feature_cat2 = torch.cat(
                                [torch.unsqueeze(feature_aug, dim=1), torch.unsqueeze(feature_ema_ori, dim=1)], dim=1)
                        loss1 = loss_shallow(feature_cat1, labels_vars, epoch=i, step=step)
                        loss2 = loss_shallow_ema(feature_cat2, labels_vars, epoch=i, step=step)
                        loss_co = loss1 + loss2
                    else:
                        feature_cat1 = torch.cat(
                            [torch.unsqueeze(features, dim=1), torch.unsqueeze(feature_ema, dim=1)], dim=1)
                        loss_co = loss_shallow(feature_cat1, labels_vars, epoch=i, step=step)

                    if(args.isCollaborative):
                        labels_vars = torch.cat((labels_vars, labels_vars), dim=0)

                    labels_vars_size = int(labels_vars.shape[0])
                    loss3 = cl_loss(preds, labels_vars.reshape((labels_vars_size)))

                    alpha1 = (2. / (1. + np.exp(-10 * i / args.max_epochs)) - 1) * args.alpha1

                    loss = loss3 + alpha1 * loss_co

                    loss.backward()
                    optimizer.step()

                    moment_update_ema(model, model_ema, 0.999)

                    if model_constraint is not None:
                        model_constraint.apply(model)

                if(args.isSaveModel and i >= args.max_epochs-10):
                    torch.save(model.state_dict(), "E:\lzg1\model_pth\\"+args.model_folder+"\\"+args.dataset+"\\S" + str(subject_id)
                               + "\S" + str(subject_id)+"F" + str(fold)+"epoch_"+str(i)+".pth")

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

                    # tsne
                    if(args.tsne):
                        f1_norm = torch.sqrt(torch.sum(outputs ** 2, dim=1))
                        f1 = outputs / torch.unsqueeze(f1_norm, dim=1)
                        if ((i == 0 or i == args.max_epochs - 1)):
                            tsne(f1.cpu().detach().numpy(), labels_vars_test.view(-1, 1).cpu().detach().numpy(), i,
                                  subject_id, fold, args.dataset)

                    y_test_pred = torch.max(outputs, 1)[1].cpu().data.numpy().squeeze()

                    acc = metrics.accuracy_score(labels_vars_test.cpu().data.numpy(), y_test_pred)
                    recall = metrics.recall_score(labels_vars_test.cpu().data.numpy(), y_test_pred,
                                                  average='macro')
                    f1 = metrics.f1_score(labels_vars_test.cpu().data.numpy(), y_test_pred, average='macro')
                    preci = metrics.precision_score(labels_vars_test.cpu().data.numpy(), y_test_pred,
                                                    average='macro')
                    kappa = metrics.cohen_kappa_score(labels_vars_test.cpu().data.numpy(), y_test_pred)

                    if(args.confusion):
                        if(i == 0 or i == args.max_epochs - 1):
                            confusion_matrix = metrics.confusion_matrix(labels_vars_test.cpu().data.numpy(), y_test_pred)
                            plot_confusion_matrix(epoch = i+1, cm = confusion_matrix, fold=fold, dataset=args.dataset, subject = subject_id)

                    epo_acc = np.append(epo_acc, acc)
                    epo_recall = np.append(epo_recall, recall)
                    epo_f1 = np.append(epo_f1, f1)
                    epo_preci = np.append(epo_preci, preci)
                    epo_kappa = np.append(epo_kappa, kappa)

            last10_acc = epo_acc[-10:]
            last10_recall = epo_recall[-10:]
            last10_f1 = epo_f1[-10:]
            last10_preci = epo_preci[-10:]
            last10_kappa = epo_kappa[-10:]

            for i in range(10):
                logger.info(
                    "subject = " + str(subject_id) + " Fold = " + str(fold) + " " + str(
                        i - 10) + " result: ACC: " + "%.4f" % last10_acc[i] +
                    " kappa: " + "%.4f" % last10_kappa[i] + " recall: " + "%.4f" % last10_recall[i] +
                    " F1: " + "%.4f" % last10_f1[i] + " precision " + "%.4f" % last10_preci[i])

            test_accuracy = np.append(test_accuracy, np.mean(last10_acc))
            test_recall = np.append(test_recall, np.mean(last10_recall))
            test_f1 = np.append(test_f1, np.mean(last10_f1))
            test_preci = np.append(test_preci, np.mean(last10_preci))
            test_kappa = np.append(test_kappa, np.mean(last10_kappa))

            epo_acc = list()
            epo_recall = list()
            epo_f1 = list()
            epo_preci = list()
            epo_kappa = list()

        # save ten-folds result
        acc_result = test_accuracy[-10:]
        recall_result = test_recall[-10:]
        f1_result = test_f1[-10:]
        preci_result = test_preci[-10:]
        kap_result = test_kappa[-10:]

        print("S=" + str(subject_id))
        print("all 10 folds ACC:", acc_result)

        ACC.append(acc_result.mean())
        kappa_all.append(kap_result.mean())
        F1_all.append(f1_result.mean())
        preci_all.append(preci_result.mean())
        recall_all.append(recall_result.mean())

nums = len(args.subject)
logger.info("====================================================================================================")
for s in range(nums):
    logger.info(
        "subject "+str(args.subject[s])+" result: ACC: " + "%.4f" % ACC[s] +" kappa: " + "%.4f" % kappa_all[s]+
        " recall: " + "%.4f" % recall_all[s]+" F1: " + "%.4f" % F1_all[s]+" precision " + "%.4f" % preci_all[s])

logger.info(
        "all subject mean result: ACC: " + "%.4f" % np.mean(ACC) +" kappa: " + "%.4f" % np.mean(kappa_all)+
        " recall: " + "%.4f" % np.mean(recall_all)+" F1: " + "%.4f" % np.mean(F1_all)+" precision " +
        "%.4f" % np.mean(preci_all))

print("subject = " + str(args.subject) + " results:")
print(ACC)
print("all subject average result of ACC：" + str(np.mean(ACC)))

