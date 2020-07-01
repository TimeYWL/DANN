import torch.nn as nn
from functions import ReverseLayerF
import torch


class Mapper(nn.Module):
    def __init__(self, opt):
        super(Mapper, self).__init__()
        self.fc1 = nn.Linear(opt.attSize + opt.nz, opt.ngh)
        self.fc2 = nn.Linear(opt.ngh, opt.resSize)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.relu = nn.ReLU(True)

    def forward(self, noise, att, alpha):
        h = torch.cat((noise, att), 1)
        h1 = self.lrelu(self.fc1(h))
        h2 = self.relu(self.fc2(h1))
        feature = h2
        reverse_feature = ReverseLayerF.apply(feature, alpha)
        return feature, reverse_feature

class Spredictor(nn.Module):
    def __init__(self, opt):
        super(Spredictor, self).__init__()
        self.fc1 = nn.Linear(opt.resSize, opt.nsh)
        self.fc2 = nn.Linear(opt.nsh, opt.attSize)
        self.relu = nn.ReLU(True)

    def forward(self, x):
        h1 = self.relu(self.fc1(x))
        h2 = self.relu(self.fc2(h1))

        return h2

class Dpredictor(nn.Module):
     def __init__(self, opt):
         super(Dpredictor, self).__init__()
         self.fc1 = nn.Linear(opt.resSize, opt.ndh)
         self.fc2 = nn.Linear(opt.ndh, 2)
         self.relu = nn.ReLU(True)
         self.log = nn.LogSoftmax(dim=1)

     def forward(self, x):
         h1 = self.relu(self.fc1(x))
         h2 = self.log(self.fc2(h1))
         return h2
