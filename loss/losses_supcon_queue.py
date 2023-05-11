"""
Author: Yonglong Tian (yonglong@mit.edu)
Date: May 07, 2020
"""
from __future__ import print_function

import torch
import torch.nn as nn


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""

    def __init__(self, temperature=0.1, contrast_mode='all',
                 base_temperature=0.07, k=128):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        self.k = k

    def update_feature(self, features, labels, epoch, step):
        if (epoch == 0 and step == 0):
            self.update_f = features
            self.update_l = labels
        else:
            self.update_f = torch.cat([self.update_f, features], dim=0)
            self.update_l = torch.cat([self.update_l, labels], dim=0)
        if (self.update_f.shape[0] > self.k):
            f = self.update_f[-self.k:, :]
            l = self.update_l[-self.k:, :]
            self.update_f = self.update_f[-self.k:, :]
            self.update_l = self.update_l[-self.k:, :]
        else:
            f = self.update_f
            l = self.update_l

        return f, l

    def forward(self, features, labels=None, mask=None, epoch=None, step=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))
        labels = labels.reshape(-1,1)
        feature_ori, feature_aug = torch.split(features, 1, dim=1)
        feature_ori, feature_aug = torch.squeeze(feature_ori, dim=1), torch.squeeze(feature_aug, dim=1)
        feature_aug, labels_aug = self.update_feature(feature_aug.detach(), labels, epoch, step)
        feature_contrast = torch.cat((feature_ori, feature_aug), dim=0)
        labels = torch.cat((labels, labels_aug), dim=0)

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            # if labels.shape[0] != batch_size:
            #     raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        # contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        contrast_feature = feature_contrast
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability  数值稳定性
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        # mask = mask.repeat(anchor_count, contrast_count)  # 原始特征的标签取的是一组数据的，而计算需要两组的标签，这里需要重复，而在当前特征中标签已经合成双份的，故不需要
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(labels.shape[0]).view(-1, 1).to(device),
            # torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos

        # loss = loss.view(anchor_count, batch_size).mean()
        loss = loss.mean()

        return loss
