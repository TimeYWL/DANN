from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import math
import util 
import sys
import model
import loss
import test

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='CUB')
parser.add_argument('--dataroot', default='./data/', help='path to dataset')
parser.add_argument('--matdataset', default=True, help='Data in matlab format')
parser.add_argument('--image_embedding', default='res101')
parser.add_argument('--class_embedding', default='att')
parser.add_argument('--syn_num', type=int, default=100, help='number features to generate per class')
parser.add_argument('--gzsl', action='store_true', default=False, help='enable generalized zero-shot learning')
parser.add_argument('--preprocessing', action='store_true', default=False, help='enbale MinMaxScaler on visual features')
parser.add_argument('--standardization', action='store_true', default=False)
parser.add_argument('--validation', action='store_true', default=False, help='enable cross validation mode')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batch_size', type=int, default=64, help='input batch size')
parser.add_argument('--resSize', type=int, default=2048, help='size of visual features')
parser.add_argument('--attSize', type=int, default=1024, help='size of semantic features')
parser.add_argument('--nz', type=int, default=1024, help='size of the latent z vector')
parser.add_argument('--ngh', type=int, default=4096, help='size of the hidden units')
parser.add_argument('--ndh', type=int, default=4096, help='size of the hidden units')
parser.add_argument('--nsh', type=int, default=4096, help='size of the hidden units')
parser.add_argument('--nepoch', type=int, default=2000, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate to train GANs ')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', default=False, help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--outf', default='./checkpoint/', help='folder to output data and model checkpoints')
parser.add_argument('--outname', help='folder to output data and model checkpoints')
parser.add_argument('--save_every', type=int, default=100)
parser.add_argument('--print_every', type=int, default=1)
parser.add_argument('--val_every', type=int, default=10)
parser.add_argument('--start_epoch', type=int, default=0)
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--nclass_all', type=int, default=200, help='number of all classes')
parser.add_argument('--gama', type=float, default=0.5)
parser.add_argument('--alpha', type=float, default=0.001)

opt = parser.parse_args()
print(opt)

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
torch.cuda.manual_seed_all(opt.manualSeed)

cudnn.benchmark = True

data = util.DATA_LOADER(opt)
print("# of training samples: ", data.ntrain)

netS = model.Spredictor(opt)
print(netS)

netD = model.Dpredictor(opt)
print(netD)

netM = model.Mapper(opt)
print(netM)

input_res = torch.FloatTensor(opt.batch_size, opt.resSize)
input_att = torch.FloatTensor(opt.batch_size, opt.attSize)
input_label = torch.LongTensor(opt.batch_size)
one = torch.ones(opt.batch_size)
one = one.long()
mone = torch.zeros(opt.batch_size)
noise = torch.FloatTensor(opt.batch_size, opt.nz)
loss_domain = torch.nn.NLLLoss()

optS = optim.Adam(netS.parameters(), lr = opt.lr)
optD = optim.Adam(netD.parameters(), lr = opt.lr)
optM = optim.Adam(netM.parameters(), lr = opt.lr)

if opt.cuda:
    netD.cuda()
    netS.cuda()
    netM.cuda()
    input_res = input_res.cuda()
    input_att = input_att.cuda()
    input_label = input_label.cuda()
    loss_domain = loss_domain.cuda()
    one = one.cuda()
    mone = mone.cuda()
    noise = noise.cuda()


def sample():
    batch_feature, batch_label, batch_att = data.next_batch(opt.batch_size)
    input_res.copy_(batch_feature)
    input_att.copy_(batch_att)
    input_label.copy_(util.map_label(batch_label, data.seenclasses))

def comp_acc():

    start = 0
    rights = 0
    ntest = data.test_unseen_feature.size(0)
    for i in range(0, ntest-opt.batch_size, opt.batch_size):
        end = start+opt.batch_size
        input_res.copy_(data.test_unseen_feature[start:end])
        output = netS(Variable(input_res))
        label = data.test_unseen_label[start:end]

        right = test.right_comp(output.data, label, data)
        rights = rights + right 

        start = end

    acc = rights / data.test_unseen_feature.size(0)
    return acc

for epoch in range(opt.nepoch):

    for i in range(0, data.ntrain, opt.batch_size):

        netD.zero_grad()
        netS.zero_grad()
        netM.zero_grad()

        sample()
        input_attv = Variable(input_att, requires_grad=True)
        input_resv = Variable(input_res, requires_grad=True)
        noise.normal_(0, 1)
        noisev = Variable(noise, requires_grad=True)

        feature, re_feature = netM(noisev, input_attv, opt.alpha)

        att_val = netS(feature)
        dom_val = netD(re_feature)

        dist = loss.cos_loss(att_val, input_attv)
        cosine = loss.euc_loss(att_val, input_attv)
        att_loss = (cosine * opt.gama + dist * (1 - opt.gama)).mean()
        att_loss.backward(retain_graph=True)
    
        res_loss = loss.euc_loss(feature, input_resv)
        res_loss = res_loss.mean()
        res_loss.backward(retain_graph=True)

        onev = Variable(one)
        s_dom = loss_domain(dom_val, onev)

        s_dom.backward()

        optM.step()
        optD.step()
        optS.step()

    acc = comp_acc()
    print(acc)













