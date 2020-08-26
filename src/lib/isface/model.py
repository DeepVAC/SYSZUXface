import math
import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import namedtuple
from torch.nn import Linear, Conv2d, BatchNorm1d, BatchNorm2d, PReLU, ReLU, Sigmoid, Dropout2d, Dropout, AvgPool2d, MaxPool2d, AdaptiveAvgPool2d, Sequential, Module, Parameter

def norm1(input):
    norm = torch.norm(input,2,1,True)
    output = torch.div(input, norm)
    return output

def norm(input,axis=1):
    norm = torch.norm(input,2,axis,True)
    output = torch.div(input, norm)
    return output

class SEModule(Module):
    def __init__(self, channels, reduction):
        super(SEModule, self).__init__()
        self.avg_pool = AdaptiveAvgPool2d(1)
        self.fc1 = Conv2d(
            channels, channels // reduction, kernel_size=1, padding=0 ,bias=False)
        self.relu = ReLU(inplace=True)
        self.fc2 = Conv2d(
            channels // reduction, channels, kernel_size=1, padding=0 ,bias=False)
        self.sigmoid = Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x

class BottleneckIr(Module):
    def __init__(self, in_channel, depth, stride):
        super(BottleneckIr, self).__init__()
        if in_channel == depth:
            self.shortcut_layer = MaxPool2d(1, stride)
        else:
            self.shortcut_layer = Sequential(
                Conv2d(in_channel, depth, (1, 1), stride ,bias=False), BatchNorm2d(depth))
        self.res_layer = Sequential(
            BatchNorm2d(in_channel),
            Conv2d(in_channel, depth, (3, 3), (1, 1), 1 ,bias=False), PReLU(depth),
            Conv2d(depth, depth, (3, 3), stride, 1 ,bias=False), BatchNorm2d(depth))

    def forward(self, x):
        shortcut = self.shortcut_layer(x)
        res = self.res_layer(x)
        return res + shortcut

class BottleneckIrSe(Module):
    def __init__(self, in_channel, depth, stride):
        super(BottleneckIrSe, self).__init__()
        if in_channel == depth:
            self.shortcut_layer = MaxPool2d(1, stride)
        else:
            self.shortcut_layer = Sequential(
                Conv2d(in_channel, depth, (1, 1), stride ,bias=False), 
                BatchNorm2d(depth))
        self.res_layer = Sequential(
            BatchNorm2d(in_channel),
            Conv2d(in_channel, depth, (3,3), (1,1),1 ,bias=False),
            PReLU(depth),
            Conv2d(depth, depth, (3,3), stride, 1 ,bias=False),
            BatchNorm2d(depth),
            SEModule(depth,16)
            )
    def forward(self,x):
        shortcut = self.shortcut_layer(x)
        res = self.res_layer(x)
        return res + shortcut

class Bottleneck(namedtuple('Block', ['in_channel', 'depth', 'stride'])):
    '''A named tuple describing a ResNet block.'''

class Backbone(Module):
    def __init__(self, num_layers, drop_ratio, embbedings, mode='ir'):
        super(Backbone, self).__init__()
        
        all_layers = {
            '50': [3,4,14,3],
            '100': [3,13,30,3],
            '152': [3,8,36,3]
        }
        self.layers = all_layers[str(num_layers)]
        self.block = lambda channel, depth, layer, stride: [Bottleneck(channel, depth, stride)] + [Bottleneck(depth, depth, 1) for i in range(layer-1)]
        self.module = BottleneckIr if mode == 'ir' else BottleneckIrSe

        self.conv1 = Conv2d(3, 64, (3, 3), 1, 1 ,bias=False)
        self.bn1 = BatchNorm2d(64)
        self.prelu1 = PReLU(64)

        self.layer1 = self.makeLayers_(64, 64, self.layers[0], 2)
        self.layer2 = self.makeLayers_(64, 128, self.layers[1], 2)
        self.layer3 = self.makeLayers_(128, 256, self.layers[2], 2)
        self.layer4 = self.makeLayers_(256, 512, self.layers[3], 2)

        self.bn2 = BatchNorm2d(512)
        self.dp1 = Dropout(drop_ratio)
        self.fc1 = Linear(512 * 7 * 7, embbedings)
        self.bn3 = BatchNorm1d(embbedings, affine=False)

        self.initWeights_()

    def makeLayers_(self, channel, depth, layers, stride):
        modules = []
        layer_block = self.block(channel, depth, layers, stride)
        for bottleneck in layer_block:
            modules.append(self.module(bottleneck.in_channel,
                            bottleneck.depth,
                            bottleneck.stride))
        return Sequential(*modules)

    def initWeights_(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self,x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.prelu1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.bn2(x)
        x = self.dp1(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.bn3(x)

        return norm1(x)

class Arcface(Module):
    def __init__(self, embedding_size=512, classnum=10575, s=32.0, m=0.50, easy_margin=False):
        super(Arcface, self).__init__()
        self.in_feature = embedding_size
        self.out_feature = classnum
        self.s = s
        self.m = m
        self.kernel = Parameter(torch.Tensor(self.out_feature, self.in_feature))
        nn.init.xavier_uniform_(self.kernel)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)

        # make the function cos(theta+m) monotonic decreasing while theta in [0°,180°]
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, x, label):
        cosine = F.linear(F.normalize(x), F.normalize(self.kernel))
        # cos(theta + m)
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m

        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where((cosine - self.th) > 0, phi, cosine - self.mm)

                    #one_hot = torch.zeros(cosine.size(), device='cuda' if torch.cuda.is_available() else 'cpu')
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output = output * self.s

        return output

class CurricularFace(Module):
    def __init__(self, embedding_size=512, classnum=10575, m = 0.5, s = 64.):
        super(CurricularFace, self).__init__()
        self.in_features = embedding_size
        self.out_features = classnum
        self.m = m
        self.s = s
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.threshold = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m
        self.kernel = Parameter(torch.Tensor(self.in_features, self.out_features))
        self.register_buffer('t', torch.zeros(1))
        nn.init.normal_(self.kernel, std=0.01)

    def forward(self, embbedings, label):
        kernel_norm = norm(self.kernel, axis = 0)
        cos_theta = torch.mm(embbedings, kernel_norm)
        cos_theta = cos_theta.clamp(-1, 1)  # for numerical stability
        with torch.no_grad():
            origin_cos = cos_theta.clone()
        target_logit = cos_theta[torch.arange(0, embbedings.size(0)), label].view(-1, 1)
        sin_theta = torch.sqrt(1.0 - torch.pow(target_logit, 2))
        cos_theta_m = target_logit * self.cos_m - sin_theta * self.sin_m #cos(target+margin)
        mask = cos_theta > cos_theta_m
        final_target_logit = torch.where(target_logit > self.threshold, cos_theta_m, target_logit - self.mm)
        hard_example = cos_theta[mask]
        with torch.no_grad():
            self.t = target_logit.mean() * 0.01 + (1 - 0.01) * self.t
        cos_theta[mask] = hard_example * (self.t + hard_example)
        cos_theta.scatter_(1, label.view(-1, 1).long(), final_target_logit)
        output = cos_theta * self.s
        return output

class BackboneDDP(Backbone):
    def __init__(self, num_layers, drop_ratio, embbedings, mode='ir'):
        super(BackboneDDP, self).__init__(num_layers, drop_ratio, embbedings, mode)
        self.head = None

    def forward(self,x, label):
        x = super(BackboneDDP, self).forward(x)
        if label is None:
            return x
        return self.head(x, label)

    def extract(self,x):
        x = super(BackboneDDP, self).forward(x)
        return x

if __name__ == "__main__":
    model = Backbone(50, 0.4, 512)
    input = torch.ones((10,3,112,112))
    model(input)
