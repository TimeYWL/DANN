import torch
import torch.nn as nn

from DANN.util import tool

class Mapper(nn.Module):
    def __init__(self, args, data):
        super(Mapper, self).__init__()
        self.fc1 = nn.Linear(data.sf_size + args.nz, 4096)
        self.fc2 = nn.Linear(4096, data.vf_size)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.relu = nn.ReLU(True)

    def forward(self, noise, sf, alpha):
        h = torch.cat((noise, sf), 1)
        h1 = self.lrelu(self.fc1(h))
        h2 = self.relu(self.fc2(h1))
        feature = h2
        reverse_feature = tool.ReverseLayerF.apply(feature, alpha)
        return feature, reverse_feature

class SemanticPredictor(nn.Module):
    def __init__(self, args, data):
        super(SemanticPredictor, self).__init__()
        self.fc1 = nn.Linear(data.vf_size, 4096)
        self.fc2 = nn.Linear(4096, data.sf_size)
        self.relu = nn.ReLU(True)

    def forward(self, x):
        h1 = self.relu(self.fc1(x))
        h2 = self.relu(self.fc2(h1))

        return h2

class DomainPredictor(nn.Module):
    def __init__(self, args, data):
        super(DomainPredictor, self).__init__()
        self.fc1 = nn.Linear(data.vf_size, 4096)
        self.fc2 = nn.Linear(4096, 2)
        self.relu = nn.ReLU(True)
        self.log = nn.LogSoftmax(dim=1)

    def forward(self, x):
        h1 = self.relu(self.fc1(x))
        h2 = self.log(self.fc2(h1))
        return h2
