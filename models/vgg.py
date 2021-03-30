### adapted from https://github.com/pytorch/vision/tree/master/torchvision

import torch
import torch.nn as nn

#from .utils import load_state_dict_from_url
try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url



__all__ = ['VGG16', 'vgg16']

model_urls = {
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
}


class VGG(nn.Module):

    def __init__(self, features, num_classes=1000, init_weights=True):
        super(VGG, self).__init__()
        self.features = features
        for k,f in enumerate(self.features):
            i = 0
            for p in f.parameters():
                #print (p.size())
                self.register_parameter(name='c%d_p%d'%(k,i), param=p)
                i += 1
        
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.fc1 = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True))
        self.fc2 = nn.Sequential(
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True))
        self.fc3 = nn.Sequential(
            nn.Linear(4096, num_classes))
        
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        fmaps = []
        for f in self.features:
            x = f(x)
            fmaps += [x,]
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        f1 = self.fc1(x)
        f2 = self.fc2(f1)
        f3 = self.fc3(f2)
        return fmaps + [f1[:, :, None, None], f2[:, :, None, None], f3[:, :, None, None]]

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

def make_layers(cfg, in_channels, batch_norm=False):
    layers = []
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
    return in_channels, nn.Sequential(*layers)

def build_vgg16_fmaps(pretrained=False, progress=True, **kwargs):
    if pretrained:
        kwargs['init_weights'] = False
        
    in_channels = 3
    layers = []
    for c in [[64, 64, 'M', 128, 128, 'M', 256], [256], [256, 'M', 512], [512], [512, 'M', 512], [512], [512, 'M']]:
        in_channels, l = make_layers(c, in_channels, batch_norm=False)
        layers += [l,]  
        
    model = VGG(layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['vgg16'],
                                              progress=progress)
        ### Rename dictionary keys to match new breakdown
        state_dict['c0_p0'] = state_dict.pop('features.0.weight')
        state_dict['c0_p1'] = state_dict.pop('features.0.bias')
        state_dict['c0_p2'] = state_dict.pop('features.2.weight')
        state_dict['c0_p3'] = state_dict.pop('features.2.bias')
        state_dict['c0_p4'] = state_dict.pop('features.5.weight')
        state_dict['c0_p5'] = state_dict.pop('features.5.bias')
        state_dict['c0_p6'] = state_dict.pop('features.7.weight')
        state_dict['c0_p7'] = state_dict.pop('features.7.bias')
        state_dict['c0_p8'] = state_dict.pop('features.10.weight')
        state_dict['c0_p9'] = state_dict.pop('features.10.bias')  
        
        state_dict['c1_p0'] = state_dict.pop('features.12.weight')
        state_dict['c1_p1'] = state_dict.pop('features.12.bias')      
        
        state_dict['c2_p0'] = state_dict.pop('features.14.weight')
        state_dict['c2_p1'] = state_dict.pop('features.14.bias')
        state_dict['c2_p2'] = state_dict.pop('features.17.weight')
        state_dict['c2_p3'] = state_dict.pop('features.17.bias')   
        
        state_dict['c3_p0'] = state_dict.pop('features.19.weight')
        state_dict['c3_p1'] = state_dict.pop('features.19.bias')      
        
        state_dict['c4_p0'] = state_dict.pop('features.21.weight')
        state_dict['c4_p1'] = state_dict.pop('features.21.bias')
        state_dict['c4_p2'] = state_dict.pop('features.24.weight')
        state_dict['c4_p3'] = state_dict.pop('features.24.bias')           
        
        state_dict['c5_p0'] = state_dict.pop('features.26.weight')
        state_dict['c5_p1'] = state_dict.pop('features.26.bias') 
        
        state_dict['c6_p0'] = state_dict.pop('features.28.weight')
        state_dict['c6_p1'] = state_dict.pop('features.28.bias') 
        
        state_dict['fc1.0.weight'] = state_dict.pop('classifier.0.weight')
        state_dict['fc1.0.bias']   = state_dict.pop('classifier.0.bias')         
        state_dict['fc2.1.weight'] = state_dict.pop('classifier.3.weight')
        state_dict['fc2.1.bias']   = state_dict.pop('classifier.3.bias')    
        state_dict['fc3.0.weight'] = state_dict.pop('classifier.6.weight')
        state_dict['fc3.0.bias']   = state_dict.pop('classifier.6.bias')    
        model.load_state_dict(state_dict)
    return model


class VGG16_fmaps(nn.Module):
    '''
    image input dtype: float in range [0,1], size: 224, but flexible
    info on the dataloader compliant with the model database
    https://github.com/pytorch/examples/blob/master/imagenet/main.py
    https://github.com/pytorch/vision/blob/master/torchvision/transforms/transforms.py
    '''
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        super(VGG16_fmaps, self).__init__()
        self.mean = nn.Parameter(torch.as_tensor(mean), requires_grad=False)
        self.std = nn.Parameter(torch.as_tensor(std), requires_grad=False)
        self.extractor = build_vgg16_fmaps(pretrained=True)

    def forward(self, _x):
        return self.extractor((_x - self.mean[None, :, None, None])/self.std[None, :, None, None])