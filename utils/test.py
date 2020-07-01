import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
import util
from torch.nn import functional as F
import sys

def right_comp(att, label, data):

    true_label = data.unseenclasses
    true_att = data.attribute[true_label]
    right_num = 0
    i = 0
    for right_label in label:
        atts = np.tile(att[i], (true_label.size(0), 1))
        pred_att = torch.from_numpy(atts)
        dist = F.pairwise_distance(true_att, pred_att, p=2)
        sorted, indices = torch.sort(dist, 0)
        idx = indices[0]

        pred_label = true_label[idx].numpy()

        if right_label == pred_label:
            right_num = right_num + 1
        i = i+1

    return right_num






