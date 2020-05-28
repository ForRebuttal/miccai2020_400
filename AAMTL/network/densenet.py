import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from collections import OrderedDict
from network.modules import *

__all__ = ['DenseNet', 'densenet121', 'densenet169', 'densenet201', 'densenet161']


model_urls = {
    'densenet121': 'https://download.pytorch.org/models/densenet121-a639ec97.pth',
    'densenet169': 'https://download.pytorch.org/models/densenet169-b2777c0a.pth',
    'densenet201': 'https://download.pytorch.org/models/densenet201-c1103571.pth',
    'densenet161': 'https://download.pytorch.org/models/densenet161-8d451a50.pth',
}


class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size *
                        growth_rate, kernel_size=1, stride=1, bias=False)),
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                        kernel_size=3, stride=1, padding=1, bias=False)),
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))

class _Attention_Pooling(nn.Module):
    def __init__(self, num_features):
        super(_Attention_Pooling, self).__init__()
        self.attention2one = nn.Linear(num_features, 1)
        
        #initially the layer equals to avg_pool
        self.attention2one.weight.data.zero_()
        self.attention2one.bias.data.zero_()
        
    def forward(self, feature):
        feature = feature.permute(0, 2, 3, 1)
        feature = feature.view(feature.size()[0], -1, feature.size()[3])
        feature_fc = self.attention2one(feature)
        feature_fc = feature_fc.permute(0, 2, 1)
        weight = F.softmax(feature_fc, dim = 2)
        feature = torch.matmul(weight, feature)
        return feature#, weight

class _Mask_Pooling(nn.Module):
    def forward(self, fmap, mask):
        #print("--mask--")
        #print(mask)
        #print("--fmap--")
        #print(fmap.size())      
        mask = mask.unsqueeze(1)
        mask_sum = torch.sum(mask,dim=[2,3])
        #print(mask_sum)
        #print(mask.size())
        mask_seq = mask.repeat(1,1920,1,1)
        mask_sum_seq = mask_sum.repeat(1,1920)
        #print(mask_seq.size())
        sum_fmap = fmap.mul(mask_seq)
        sum_fmap = torch.sum(sum_fmap, dim=[2,3])
        #print("--sum_fmap--")
        #print(sum_fmap)
        result = sum_fmap/mask_sum_seq
        #print("--sum_fmap--")
        #print(result)
        return result
        #exit()

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)
        
class Up2(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)

    def forward(self, x):
        return self.up(x)

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)
        
class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            DoubleConv(in_channels, out_channels),
            nn.AvgPool2d(2)
        )

    def forward(self, x):
        return self.maxpool_conv(x)     
        
class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
class SetTransformer(nn.Module):
    def __init__(self, dim_input, dim_output,
            num_inds=32, dim_hidden=128, num_heads=5, ln=False):
        super(SetTransformer, self).__init__()
        # self.enc = nn.Sequential(
        #         ISAB(dim_input, dim_hidden, num_heads, num_inds, ln=ln),
        #         ISAB(dim_hidden, dim_hidden, num_heads, num_inds, ln=ln))
        # self.dec = nn.Sequential(
        #         PMA(dim_hidden, num_heads, ln=ln),
        #         SAB(dim_hidden, dim_hidden, num_heads, ln=ln),
        #         SAB(dim_hidden, dim_hidden, num_heads, ln=ln),
        #         nn.Linear(dim_hidden, dim_output))
        self.enc = nn.Sequential(
                SAB(dim_hidden, dim_hidden, num_heads, ln=ln),
                SAB(dim_hidden, dim_hidden, num_heads, ln=ln),
        )
        self.dec = PMA(dim_hidden, num_heads, ln=ln)

    def forward(self, Q, X):
        return self.dec(Q, self.enc(X))
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)
class ClassBlock(nn.Module):
    def __init__(self, num_features, n_class):
        super(ClassBlock, self).__init__()
        typeList = [2, 2, 2, 2]
        self.typeLen = len(typeList)
        self.adaptive_pool = nn.AdaptiveAvgPool2d(1)
        self.channel_attention = nn.ModuleList([
            ChannelAttention(num_features)
            for typeIndex in range(self.typeLen)
        ])
        self.space_attention = nn.ModuleList([
            SpatialAttention()
            for typeIndex in range(self.typeLen)
        ])
        self.set_transformer = SetTransformer(num_features, num_features, dim_hidden=num_features)
        self.normalize = nn.LayerNorm(num_features)
        # self.y_transformer = nn.Linear(num_features, n_class)
        self.y_transformer = nn.Sequential(nn.Linear(num_features, num_features), nn.ReLU(), nn.Dropout(0.5),nn.Linear(num_features, n_class))
        self.classifier_stack = nn.ModuleList([
            nn.Linear(num_features, typeList[typeIndex])
            for typeIndex in range(len(typeList))])
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)
    def getLarger(self, out, out_input):
        # k = out.max(-1, keepdim=True)[1]  # y_soft.max(-1)
        # y_hard = torch.zeros_like(out).scatter_(-1, k, 1.0)
        # out_y = y_hard - out.detach() + out
        # out_y = torch.matmul(out_y.unsqueeze(1), out_input).squeeze(1)

        out_y = torch.matmul(out.unsqueeze(1), out_input).squeeze(1)

        return out_y

    def normalize_channels(self, attention_channels):
        max = torch.max(attention_channels, -1)[0].unsqueeze(2).repeat(1,1,attention_channels.size(2))
        min = torch.min(attention_channels, -1)[0].unsqueeze(2).repeat(1,1,attention_channels.size(2))
        return (attention_channels-min)/(max-min)

    def deal_task_layer(self, task_layer, out, count, encoder_input):
        tem_fea = out
        tem_out = task_layer(out)  # self.extract_feature(out,task_layer)
        # out_list.append(tem_out)

        tem_weight = self.state_dict()['classifier_stack.' + str(count) + '.weight']
        # tem_bias = self.state_dict()['classifier_stack.' + str(count) + '.bias']
        two_fea = tem_fea.unsqueeze(1).repeat(1, tem_weight.size(0), 1)

        out_input = (two_fea * F.softmax(
            two_fea * tem_weight.unsqueeze(0).repeat(tem_fea.size(0), 1, 1), -1))
        out_input = self.getLarger(tem_out, out_input)



        if encoder_input is None:
            encoder_input = out_input.unsqueeze(1)
        else:
            encoder_input = torch.cat((encoder_input, out_input.unsqueeze(1)), 1)

        return encoder_input, tem_out

    def forward(self, x, feature_maps):
        #分来计算28个分类
        count = -1
        out_input = None
        out_list = []
        fea_set = [x]
        for space in self.space_attention:
            count+=1
            maps = self.channel_attention[count](feature_maps) * feature_maps
            maps = space(maps) * maps
            x_attr = self.adaptive_pool(maps).view(maps.size(0), maps.size(1))
            fea_set.append(x_attr)
            # print(x_attr.size())
            if out_input is None:
                out_input = x_attr.unsqueeze(1)
            else:
                out_input = torch.cat((out_input, x_attr.unsqueeze(1)), 1)

        out_input = torch.cat((x.unsqueeze(1), out_input), 1)
        x2 = self.set_transformer(out_input, out_input)
        out_y = self.y_transformer(x2[:,0,:])
        count = 0
        for enc_layer in self.classifier_stack:
            count+=1
            out_list.append(enc_layer(x2[:, count, :]))
        return out_y, out_list

class DenseNet(nn.Module):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_
    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
    """

    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16),
                 n_channels=3, num_init_features=64, bn_size=4, drop_rate=0, num_classes=1, bilinear=True):
        self.n_channels = n_channels
        self.n_classes = num_classes
        self.bilinear = bilinear
        
        block_set=[]
        super(DenseNet, self).__init__()

        # First convolution
        self.block_0 = (nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(self.n_channels, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ])))
        #self.block_0 = DoubleConv(n_channels, 64)
        
        #self.inc   = DoubleConv(n_channels, 64)
        #self.down1 = Down(64,64)
        #self.down2 = Down(64,64)
        
        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            t_feature = nn.Sequential()
            block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
            t_feature.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
                t_feature.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2
            block_set.append(t_feature) 
        
        self.block_1 = block_set[0]
        self.block_2 = block_set[1]
        self.block_3 = block_set[2]
        self.block_4 = block_set[3]
        
        # Final batch norm
        self.block_4.add_module('norm5', nn.BatchNorm2d(num_features))
        self.block_4.add_module('relu5', nn.ReLU(inplace=True))
        
        self.up1 = Up(2816, 896, bilinear)
        self.up2 = Up(1152, 256, bilinear)
        self.up3 = Up(384, 64, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.up5 = Up2(64, 64, bilinear)
        self.up6 = Up2(64, 64, bilinear)
        self.outc = OutConv(64, self.n_classes)
        self.classifier = ClassBlock(num_features, 2)
        #self.classifier = nn.linear(num_features, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)
    def save_gradient(self, grad):
        # if hasattr(self, "gradients"):
        #     self.gradients.append(grad)
        # else:
        #     self.gradients = []
        #     self.gradients.append(grad)

        self.gradients = grad
         
    def forward(self, x):
        x0 = self.block_0(x)    #64              *56
        x1 = self.block_1(x0)   #128             *28
        x2 = self.block_2(x1)   #256             *14
        x3 = self.block_3(x2)   #896             *7
        x4 = self.block_4(x3)   #1920            *3
        x4.register_hook(self.save_gradient)
        out_vector = F.adaptive_avg_pool2d(x4, (1, 1)).view(x4.size(0), -1)
        out, out_list = self.classifier(out_vector, x4)

        x = self.up1(x4, x3)    #1920+896 ->896  *7
        x = self.up2(x, x2)     #896+256  ->128  *14
        x = self.up3(x, x1)     #256+128  ->64   *28        
        x = self.up4(x, x0)     #64+64    ->64   *56        
        x = self.up5(x)         #64+64    ->64   *112        
        x = self.up6(x)         #64+64    ->64   *224        
        logits = self.outc(x)   #64->2
      
        
        #out_vector = F.adaptive_avg_pool2d(out, (1, 1)).view(features.size(0), -1)

        #x, out_list, fea_set = self.classifier(out_vector, out)

        return logits, (out, out_list[0], out_list[1], out_list[2], out_list[3]), x4

    def extract_feature(self, out,net):
        for name, midlayer in net._modules.items():
            out = midlayer(out)
            if name=="1":
                #print(name)
                fea = out
        return F.softmax(out),fea

def _load_state_dict(model, model_url):
    # '.'s are no longer allowed in module names, but pervious _DenseLayer
    # has keys 'norm.1', 'relu.1', 'conv.1', 'norm.2', 'relu.2', 'conv.2'.
    # They are also in the checkpoints in model_urls. This pattern is used
    # to find such keys.
    pattern = re.compile(
        r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
    state_dict = model_zoo.load_url(model_url)
    for key in list(state_dict.keys()):
        res = pattern.match(key)
        if res:
            new_key = res.group(1) + res.group(2)
            state_dict[new_key] = state_dict[key]
            del state_dict[key]
    model.load_state_dict(state_dict)


def densenet121(pretrained=False, **kwargs):
    r"""Densenet-121 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = DenseNet(num_init_features=64, growth_rate=32, block_config=(6, 12, 24, 16),
                     **kwargs)
    if pretrained:
        _load_state_dict(model, model_urls['densenet121'])
    return model


def densenet169(pretrained=False, **kwargs):
    r"""Densenet-169 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = DenseNet(num_init_features=64, growth_rate=32, block_config=(6, 12, 32, 32),
                     **kwargs)
    if pretrained:
        _load_state_dict(model, model_urls['densenet169'])
    return model


def densenet201(pretrained=False, **kwargs):
    r"""Densenet-201 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = DenseNet(num_init_features=64, growth_rate=32, block_config=(6, 12, 48, 32),
                     **kwargs)
    if pretrained:
        _load_state_dict(model, model_urls['densenet201'])
    return model


def densenet161(pretrained=False, **kwargs):
    r"""Densenet-161 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = DenseNet(num_init_features=96, growth_rate=48, block_config=(6, 12, 36, 24),
                     **kwargs)
    if pretrained:
        _load_state_dict(model, model_urls['densenet161'])
    return model
    
if __name__=="__main__":
    net = densenet201()
    print(net.block_set)