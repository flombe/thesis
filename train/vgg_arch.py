import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import math
import numpy as np

__all__ = [
    'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19_bn', 'vgg19',
]

model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}


class VGG(nn.Module):

    def __init__(self, features, num_classes=1000, init_weights=True):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(512*7*7, 4096),     # just 512 for cifar10
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    # Sequential(
    # (0): Linear(in_features=25088, out_features=4096)
    # (1): ReLU(inplace)
    # (2): Dropout(p=0.5)
    # (3): Linear(in_features=4096, out_features=4096)
    # (4): ReLU(inplace)
    # (5): Dropout(p=0.5)
    # (6): Linear(in_features=4096, out_features=2)
    # )

    def extract(self, x, verbose=False):
        out1 = self.features(x)
        out2 = F.relu(self.classifier[0](out1.view(out1.size(0), -1)))
        out3 = F.relu(self.classifier[3](out2))
        out4 = F.relu(self.classifier[6](out3))
        if verbose == True:
            print(out1.size())
            print(out2.size())
            print(out3.size())
            print(out4.size())
        return out1, out2, out3, out4

    #   Sequential(
    #   (0): Conv2d (3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    #   (1): ReLU(inplace)
    #   (2): Conv2d (64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    #   (3): ReLU(inplace)
    #   (4): MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1))
    #   (5): Conv2d (64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    #   (6): ReLU(inplace)
    #   (7): Conv2d (128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    #   (8): ReLU(inplace)
    #   (9): MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1))
    #   (10): Conv2d (128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    #   (11): ReLU(inplace)
    #   (12): Conv2d (256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    #   (13): ReLU(inplace)
    #   (14): Conv2d (256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    #   (15): ReLU(inplace)
    #   (16): MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1))
    #   (17): Conv2d (256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    #   (18): ReLU(inplace)
    #   (19): Conv2d (512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    #   (20): ReLU(inplace)
    #   (21): Conv2d (512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    #   (22): ReLU(inplace)
    #   (23): MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1))
    #   (24): Conv2d (512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    #   (25): ReLU(inplace)
    #   (26): Conv2d (512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    #   (27): ReLU(inplace)
    #   (28): Conv2d (512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    #   (29): ReLU(inplace)
    #   (30): MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1))
    # )

    def extract_all(self, x, verbose=False):
        # from pooling layers 1,2,3,4,5
        out0 = x
        out1 = self.features[4](F.relu(self.features[2](F.relu(self.features[0](x)))))
        out2 = self.features[9](F.relu(self.features[7](F.relu(self.features[5](out1)))))
        out3 = self.features[16](F.relu(self.features[14](F.relu(self.features[12](F.relu(self.features[10](out2)))))))
        out4 = self.features[23](F.relu(self.features[21](F.relu(self.features[19](F.relu(self.features[17](out3)))))))
        out5 = self.features[30](F.relu(self.features[28](F.relu(self.features[26](F.relu(self.features[24](out4)))))))

        # from the classifier part
        out6 = F.relu(self.classifier[0](out5.view(out5.size(0), -1)))
        out7 = F.relu(self.classifier[3](out6))
        out8 = F.relu(self.classifier[6](out7))
        if verbose == True:
            print(str(out0.shape) + ' ' + str(np.prod(out0.shape[1:])))
            print(str(out1.shape) + ' ' + str(np.prod(out1.shape[1:])))
            print(str(out2.shape) + ' ' + str(np.prod(out2.shape[1:])))
            print(str(out3.shape) + ' ' + str(np.prod(out3.shape[1:])))
            print(str(out4.shape) + ' ' + str(np.prod(out4.shape[1:])))
            print(str(out5.shape) + ' ' + str(np.prod(out5.shape[1:])))
            print(str(out6.shape) + ' ' + str(np.prod(out6.shape[1:])))
            print(str(out7.shape) + ' ' + str(np.prod(out7.shape[1:])))
            print(str(out8.shape) + ' ' + str(np.prod(out8.shape[1:])))

        return [out0, out1, out2, out3, out4, out5, out6, out7, out8]

    def extract_all_vgg19(self, x):
        # Sequential(
        # (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        # (1): ReLU(inplace=True)
        # (2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        # (3): ReLU(inplace=True)
        # (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        # (5): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        # (6): ReLU(inplace=True)
        # (7): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        # (8): ReLU(inplace=True)
        # (9): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        # (10): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        # (11): ReLU(inplace=True)
        # (12): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        # (13): ReLU(inplace=True)
        # (14): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        # (15): ReLU(inplace=True)
        # (16): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        # (17): ReLU(inplace=True)
        # (18): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        # (19): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        # (20): ReLU(inplace=True)
        # (21): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        # (22): ReLU(inplace=True)
        # (23): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        # (24): ReLU(inplace=True)
        # (25): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        # (26): ReLU(inplace=True)
        # (27): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        # (28): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        # (29): ReLU(inplace=True)
        # (30): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        # (31): ReLU(inplace=True)
        # (32): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        # (33): ReLU(inplace=True)
        # (34): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        # (35): ReLU(inplace=True)
        # (36): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        # )

        # from pooling layers 1,2,3,4,5
        out0 = x
        out1 = self.features[4](F.relu(self.features[2](F.relu(self.features[0](x)))))
        out2 = self.features[9](F.relu(self.features[7](F.relu(self.features[5](out1)))))
        out3 = self.features[18](F.relu(self.features[16](
            F.relu(self.features[14](F.relu(self.features[12](F.relu(self.features[10](out2)))))))))
        out4 = self.features[27](F.relu(self.features[25](
            F.relu(self.features[23](F.relu(self.features[21](F.relu(self.features[19](out3)))))))))
        out5 = self.features[36](F.relu(self.features[34](
            F.relu(self.features[32](F.relu(self.features[30](F.relu(self.features[28](out4)))))))))

        # --> mismatch of 7x7x512 output for custom3D and init for cifar10 with only 1x1x512 dim going into classifer
        # --> can't extract activations on fc layers

        return [out0, out1, out2, out3, out4, out5]

    def extract_all_vgg16bn(self, x):
        # Sequential(
        # (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        # (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        # (2): ReLU(inplace=True)
        # (3): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        # (4): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        # (5): ReLU(inplace=True)
        # (6): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        # (7): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        # (8): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        # (9): ReLU(inplace=True)
        # (10): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        # (11): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        # (12): ReLU(inplace=True)
        # (13): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        # (14): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        # (15): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        # (16): ReLU(inplace=True)
        # (17): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        # (18): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        # (19): ReLU(inplace=True)
        # (20): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        # (21): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        # (22): ReLU(inplace=True)
        # (23): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        # (24): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        # (25): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        # (26): ReLU(inplace=True)
        # (27): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        # (28): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        # (29): ReLU(inplace=True)
        # (30): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        # (31): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        # (32): ReLU(inplace=True)
        # (33): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        # (34): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        # (35): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        # (36): ReLU(inplace=True)
        # (37): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        # (38): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        # (39): ReLU(inplace=True)
        # (40): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        # (41): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        # (42): ReLU(inplace=True)
        # (43): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        # )

        # from pooling layers 1,2,3,4,5
        out0 = x
        out1 = self.features[6](F.relu(self.features[4](self.features[3](F.relu(self.features[1](self.features[0](x)))))))
        out2 = self.features[13](F.relu(self.features[11](self.features[10](F.relu(self.features[8](self.features[7](out1)))))))
        out3 = self.features[23](F.relu(self.features[21](self.features[20](
            F.relu(self.features[18](self.features[17](F.relu(self.features[15](self.features[14](out2))))))))))
        out4 = self.features[33](F.relu(self.features[31](self.features[30](
            F.relu(self.features[28](self.features[27](F.relu(self.features[25](self.features[24](out3))))))))))
        out5 = self.features[43](F.relu(self.features[41](self.features[40](
            F.relu(self.features[38](self.features[37](F.relu(self.features[35](self.features[34](out4))))))))))

        # from the classifier part
        out6 = F.relu(self.classifier[0](out5.view(out5.size(0), -1)))
        out7 = F.relu(self.classifier[3](out6))
        out8 = F.relu(self.classifier[6](out7))

        return [out0, out1, out2, out3, out4, out5, out6, out7, out8]


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def vgg11(pretrained=False, **kwargs):
    """VGG 11-layer model (configuration "A")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['A']), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg11']))
    return model


def vgg11_bn(pretrained=False, **kwargs):
    """VGG 11-layer model (configuration "A") with batch normalization
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['A'], batch_norm=True), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg11_bn']))
    return model


def vgg13(pretrained=False, **kwargs):
    """VGG 13-layer model (configuration "B")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['B']), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg13']))
    return model


def vgg13_bn(pretrained=False, **kwargs):
    """VGG 13-layer model (configuration "B") with batch normalization
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['B'], batch_norm=True), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg13_bn']))
    return model


def vgg16(pretrained=False, **kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['D']), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg16']))
    return model


def vgg16_bn(pretrained=False, **kwargs):
    """VGG 16-layer model (configuration "D") with batch normalization
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['D'], batch_norm=True), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg16_bn']))
    return model


def vgg19(pretrained=False, **kwargs):
    """VGG 19-layer model (configuration "E")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['E']), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg19']))
    return model


def vgg19_bn(pretrained=False, **kwargs):
    """VGG 19-layer model (configuration 'E') with batch normalization
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['E'], batch_norm=True), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg19_bn']))
    return model
