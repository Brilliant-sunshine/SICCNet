# coding=UTF8
import torch
import numpy as np

'''
    refer：CutCat：An Augmentation Method for EEG Classification
'''
def cutcat(data, label, nSeg=4):

    swapSeg = (int)(nSeg / 2)
    segLen = (int)(data.shape[2] / nSeg)
    data, label = np.array(data.cpu()), np.array(label.cpu())
    new_data = np.ones_like(data)
    for n in range(data.shape[0]):
        label_n = label[n, :]
        index = np.array(np.where(label == label_n))[0]
        delete_index = np.array(np.where(index == n))[0]
        index = np.delete(index, delete_index, 0)
        data_n = data[index]
        idx = np.random.randint(0, data_n.shape[0])

        random_Seg = np.random.permutation(nSeg)
        for index, seg in enumerate(random_Seg):
            if index < swapSeg:
                if(seg == nSeg-1):
                    new_data[n, :, segLen * seg: , :] = data_n[idx, :, segLen * seg: , :]
                else:
                    new_data[n, :, segLen * seg: segLen * (seg + 1), :] = data_n[idx, :,
                                                                          segLen * seg: segLen * (seg + 1), :]
            else:
                if (seg == nSeg - 1):
                    new_data[n, :, segLen * seg:, :] = data[n, :, segLen * seg:, :]
                else:
                    new_data[n, :, segLen * seg: segLen * (seg + 1), :] = data[n, :, segLen * seg : segLen * (seg + 1), :]

    return torch.from_numpy(new_data).cuda()

def cutcat_mixup(data, label, alpha, nSeg=4):
    segLen = (int)(data.shape[2] / nSeg)
    data, label = np.array(data.cpu()), np.array(label.cpu())
    lam_mu = np.random.beta(alpha, alpha)
    new_data = np.ones_like(data)
    for n in range(data.shape[0]):
        label_n = label[n, :]
        index = np.array(np.where(label == label_n))[0]
        data_n = data[index]
        idx = np.random.randint(0, data_n.shape[0])

        new_data[n, :, : segLen, : ] = data[n, :, : segLen, : ]
        new_data[n, :, segLen : segLen*2, : ] = lam_mu * data[n, :, segLen : segLen*2, : ] + \
                                                (1-lam_mu) * data_n[idx, :, segLen : segLen*2, : ]
        new_data[n, :, segLen*2 : segLen*3, : ] = data[n, :, segLen*2 : segLen*3, : ]
        new_data[n, :, segLen*3 : , : ] = lam_mu * data[n, :, segLen*3 : , : ] + \
                                          (1-lam_mu) * data_n[idx, :, segLen*3 : , : ]

    return torch.from_numpy(new_data).cuda()

def cutcat_mixup_random(data, label, alpha, nSeg=4):
    swapSeg = (int)(nSeg / 2)
    segLen = (int)(data.shape[2] / nSeg)
    data, label = np.array(data.cpu()), np.array(label.cpu())
    lam_mu = np.random.beta(alpha, alpha)
    new_data = np.ones_like(data)
    for n in range(data.shape[0]):
        label_n = label[n, :]
        index = np.array(np.where(label == label_n))[0]
        delete_index = np.array(np.where(index == n))[0]
        index = np.delete(index, delete_index, 0)
        data_n = data[index]
        try:
            idx = np.random.randint(0, data_n.shape[0])
        except:
            print("Exception: cutcat_mixup_random.py")
        random_Seg = np.random.permutation(nSeg)
        for index, seg in enumerate(random_Seg):
            if index < swapSeg:
                if (seg == nSeg - 1):
                    new_data[n, :, segLen * seg: , :] = \
                        lam_mu * data[n, :, segLen * seg : , :] + (1 - lam_mu) * data_n[idx, :, segLen * seg : , :]
                else:
                    new_data[n, :, segLen * seg:segLen * (seg + 1), :] = \
                        lam_mu * data[n, :, segLen * seg:segLen * (seg + 1), :] + (1 - lam_mu) * data_n[idx, :, segLen * seg:segLen * (seg + 1), :]
            else:
                if (seg == nSeg - 1):
                    new_data[n, :, segLen * seg:, :] = data[n, :, segLen * seg:, :]
                else:
                    new_data[n, :, segLen * seg: segLen * (seg + 1), :] = data[n, :, segLen * seg: segLen * (seg + 1),:]

    return torch.from_numpy(new_data).cuda()


