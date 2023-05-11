import torch
import numpy as np

def MixUp(data, label, alpha):
    data, label = np.array(data.cpu()), np.array(label.cpu())
    lam_mu = np.random.beta(alpha, alpha)
    new_data = np.zeros_like(data)
    for n in range(data.shape[0]):
        label_n = label[n, :]
        index = np.array(np.where(label == label_n))[0]
        data_n = data[index]
        idx = np.random.randint(0, data_n.shape[0])
        new_data[n, :] = lam_mu * data[n, :] + (1 - lam_mu) * data_n[idx, :]

    return torch.from_numpy(new_data).cuda(), torch.from_numpy(label).cuda()