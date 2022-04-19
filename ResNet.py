from __future__ import absolute_import

import torch
from torch import nn
from torch.nn import init
from torch.nn import functional as F
from torch.nn import Parameter
import torchvision

from .resnets1 import resnet34, resnet50, resnet101

__all__ = ['ResNet50', 'ResNet50norm', 'ResNet101norm', 'MBCResNet50']


class AngleLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(AngleLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        self.weight.data.uniform_(-1, 1).renorm_(2,0,1e-5).mul_(1e5) # ?????????????

    def forward(self, input):
        x = input   # size=(B,F)  B is batchsize, F is feature len
        w = self.weight # size=(Classnum, F) 

        xnorm = torch.norm(x, p=2, dim=1, keepdim=True)
        wnorm = torch.norm(w, p=2, dim=1, keepdim=True)
        x = x.div(xnorm.expand_as(x))
        w = w.div(wnorm.expand_as(w))

        return F.linear(x, w)


class ResNet50(nn.Module):
    def __init__(self, num_classes, loss={'xent'}, dropout=0, **kwargs):
        super(ResNet50, self).__init__()
        self.loss = loss
        resnet50_ft = resnet34(pretrained=True)
        self.base = nn.Sequential(*list(resnet50_ft.children())[:-2])

        num_ftrs = resnet50_ft.fc.in_features

        # add_block = []
        # add_block += [nn.BatchNorm1d(num_ftrs)]
        # add_block += [nn.LeakyReLU(0.1)]
        # add_block += [nn.Dropout(p=dropout)]
        # add_block = nn.Sequential(*add_block)
        # self.bn = add_block
        self.bn = nn.BatchNorm1d(num_ftrs)
        # self.relu = nn.LeakyReLU(0.1)
        # self.dropout = nn.Dropout(p=dropout)

        self.classifier = nn.Linear(num_ftrs, num_classes)

        self.feat_dim = 2048 # feature dimension

    def forward(self, x):
        x = self.base(x)
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(x.size(0), -1)
            
        f = self.bn(x)
        y = self.classifier(f)

        return y, f


class ResNet50norm(nn.Module):
    def __init__(self, num_classes, loss={'xent'}, dropout=0, **kwargs):
        super(ResNet50norm, self).__init__()
        self.loss = loss
        resnet50_ft = resnet34(pretrained=True)
        self.base = nn.Sequential(*list(resnet50_ft.children())[:-2])

        num_ftrs = resnet50_ft.fc.in_features


        self.feat_dim = 512 # feature dimension
        self.embedding = nn.Linear(2048, self.feat_dim)

        self.bn = nn.BatchNorm1d(self.feat_dim)
        self.classifier = AngleLinear(self.feat_dim, num_classes)
        

    def forward(self, x):
        x = self.base(x)
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(x.size(0), -1)
        x = self.embedding(x)    
        f = self.bn(x)

        y = self.classifier(f)

        return y, f

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, 0, 0.001)
        if m.bias:
            init.zeros_(m.bias.data)

class ResNet101norm(nn.Module):
    def __init__(self, num_classes, loss={'xent'}, dropout=0, **kwargs):
        super(ResNet101norm, self).__init__()
        self.loss = loss
        resnet101_ft = resnet101(pretrained=True)
        self.base = nn.Sequential(*list(resnet101_ft.children())[:-2])

        num_ftrs = resnet101_ft.fc.in_features


        self.feat_dim = 512 # feature dimension
        self.embedding = nn.Linear(2048, self.feat_dim)

        self.bn = nn.BatchNorm1d(self.feat_dim)
        self.classifier = AngleLinear(self.feat_dim, num_classes)
        

    def forward(self, x):
        x = self.base(x)
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(x.size(0), -1)
        x = self.embedding(x)    
        f = self.bn(x)

        y = self.classifier(f)

        return y, f


class MBCResNet50(nn.Module):
    def __init__(self, dataset, dropout=0, **kwargs):
        super(MBCResNet50, self).__init__()
        resnet50_ft = resnet34(pretrained=True)
        self.base = nn.Sequential(*list(resnet50_ft.children())[:-2])

        num_ftrs = resnet50_ft.fc.in_features


        self.feat_dim = 512 # feature dimension
        self.embedding = nn.Linear(512, self.feat_dim)

        self.bn = nn.BatchNorm1d(self.feat_dim)

        num_classes = dataset.num_train_pids
        self.num_cameras = dataset.num_cameras 

        self.MBC = nn.ModuleList([AngleLinear(self.feat_dim, num_classes) 
                                  for _ in range(self.num_cameras)])

        self.classifier = nn.Linear(self.feat_dim, num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)

    def forward(self, x, cids, pids):
        x = self.base(x)
        x = F.avg_pool2d(x, x.size()[2:])

        x = x.view(x.size(0), -1)
        x = self.embedding(x)    
        f = self.bn(x)

        mbf = [f[cids == i] for i in range(self.num_cameras)]
        mby = [self.MBC[i](mbf[i]) for i in range(self.num_cameras)] 
        labels = [pids[cids == i] for i in range(self.num_cameras)]

        return mby, f, labels, self.classifier(f) 
