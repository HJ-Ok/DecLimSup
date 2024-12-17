import math
import re
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import cohen_kappa_score, f1_score


def calc_f1_acc(pred, df):
    values = np.arange(3, -1 - 0.25, -0.25)
    pred = pred[:, :, 1]
    _, idx = pred.max(1)

    y = idx.cpu().numpy()
    correct_count = 0
    for idx, tmp in enumerate(y):
        tmp_alpha = values[tmp]
        column_name = f"alpha{tmp_alpha}_is_correct"
        if df.at[idx, column_name] == 1:
            correct_count += 1
    return correct_count / len(df)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return "%dm %ds" % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return "%s (remain %s)" % (asMinutes(s), asMinutes(rs))
