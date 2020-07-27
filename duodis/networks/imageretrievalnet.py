import os
import pdb

import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo

import torchvision_

from layers.pooling import MAC, SPoC, GeM, GeMmp, RMAC, Rpool, RMATCH, CroW
from layers.normalization import L2N, PowerLaw
from datasets.genericdataset import ImagesFromList
from utils.general import get_data_root

# for some models, we have imported features (convolutions) from caffe because the image retrieval performance is higher for them
FEATURES = {
    # 'vgg16'         : 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/imagenet/imagenet-caffe-vgg16-features-d369c8e.pth',
    'resnet50'      : 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/imagenet/imagenet-caffe-resnet50-features-ac468af.pth',
    'resnet101'     : 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/imagenet/imagenet-caffe-resnet101-features-10a101d.pth',
    'resnet152'     : 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/imagenet/imagenet-caffe-resnet152-features-1011020.pth',
}

# possible global pooling layers, each on of these can be made regional
POOLING = {
    'mac'   : MAC,
    'spoc'  : SPoC,
    'gem'   : GeM,
    'gemmp' : GeMmp,
    'rmac'  : RMAC,
    'rmatch': RMATCH,
    'crow'  : CroW
}

# TODO: pre-compute for: resnet50-gem-r, resnet50-mac-r, vgg16-mac-r, alexnet-mac-r
# pre-computed regional whitening, for most commonly used architectures and pooling methods
R_WHITENING = {
    'alexnet-gem-r'   : 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/whiten/retrieval-SfM-120k/retrieval-SfM-120k-alexnet-gem-r-rwhiten-c8cf7e2.pth',
    'vgg16-gem-r'     : 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/whiten/retrieval-SfM-120k/retrieval-SfM-120k-vgg16-gem-r-rwhiten-19b204e.pth',
    'resnet101-mac-r' : 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/whiten/retrieval-SfM-120k/retrieval-SfM-120k-resnet101-mac-r-rwhiten-7f1ed8c.pth',
    'resnet101-gem-r' : 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/whiten/retrieval-SfM-120k/retrieval-SfM-120k-resnet101-gem-r-rwhiten-adace84.pth',
}

# TODO: pre-compute for more architectures
# pre-computed final (global) whitening, for most commonly used architectures and pooling methods
WHITENING = {
    'alexnet-gem'            : 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/whiten/retrieval-SfM-120k/retrieval-SfM-120k-alexnet-gem-whiten-454ad53.pth',
    'alexnet-gem-r'          : 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/whiten/retrieval-SfM-120k/retrieval-SfM-120k-alexnet-gem-r-whiten-4c9126b.pth',
    'vgg16-gem'              : 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/whiten/retrieval-SfM-120k/retrieval-SfM-120k-vgg16-gem-whiten-eaa6695.pth',
    'vgg16-gem-r'            : 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/whiten/retrieval-SfM-120k/retrieval-SfM-120k-vgg16-gem-r-whiten-83582df.pth',
    'resnet50-gem'           : 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/whiten/retrieval-SfM-120k/retrieval-SfM-120k-resnet50-gem-whiten-f15da7b.pth',
    'resnet101-mac-r'        : 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/whiten/retrieval-SfM-120k/retrieval-SfM-120k-resnet101-mac-r-whiten-9df41d3.pth',
    'resnet101-gem'          : 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/whiten/retrieval-SfM-120k/retrieval-SfM-120k-resnet101-gem-whiten-22ab0c1.pth',
    'resnet101-gem-r'        : 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/whiten/retrieval-SfM-120k/retrieval-SfM-120k-resnet101-gem-r-whiten-b379c0a.pth',
    'resnet101-gemmp'        : 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/whiten/retrieval-SfM-120k/retrieval-SfM-120k-resnet101-gemmp-whiten-770f53c.pth',
    'resnet152-gem'          : 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/whiten/retrieval-SfM-120k/retrieval-SfM-120k-resnet152-gem-whiten-abe7b93.pth',
    'densenet121-gem'        : 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/whiten/retrieval-SfM-120k/retrieval-SfM-120k-densenet121-gem-whiten-79e3eea.pth',
    'densenet169-gem'        : 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/whiten/retrieval-SfM-120k/retrieval-SfM-120k-densenet169-gem-whiten-6b2a76a.pth',
    'densenet201-gem'        : 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/whiten/retrieval-SfM-120k/retrieval-SfM-120k-densenet201-gem-whiten-22ea45c.pth',
}

# output dimensionality for supported architectures
OUTPUT_DIM = {
    'alexnet'               :  256,
    'vgg11'                 :  512,
    'vgg13'                 :  512,
    'vgg16'                 :  512,
    'vgg19'                 :  512,
    'resnet18'              :  512,
    'resnet34'              :  512,
    'resnet50'              : 2048,
    'resnet101'             : 2048,
    'resnet152'             : 2048,
    'densenet121'           : 1024,
    'densenet169'           : 1664,
    'densenet201'           : 1920,
    'densenet161'           : 2208, # largest densenet
    'squeezenet1_0'         :  512,
    'squeezenet1_1'         :  512,
}

class CamNet(nn.Module):

    def __init__(self, features, pool, num_class):
        super(CamNet, self).__init__()
        self.features = nn.Sequential(*features)
        self.pool = pool
        self.norm = L2N()
        self.fc = nn.Linear(512, num_class, bias=False)
        # self.convs = nn.Conv2d(512, num_class, kernel_size=(1,1),stride=(1,1))

    def forward(self, x):
        with torch.no_grad():
            o = self.norm(self.pool(self.features(x)))
            o = o.view(o.size(0), -1)
        o = self.fc(o)
        return o

class CamGenNet(nn.Module):
    def __init__(self, features, convs):
        super(CamGenNet, self).__init__()
        self.features = nn.Sequential(*features)
        self.convs = convs

    def forward(self,x):
        o = self.convs(self.features(x))
        return o

class ImageRetrievalNet(nn.Module):
    
    def __init__(self, features, lwhiten, pool, whiten, meta):
        super(ImageRetrievalNet, self).__init__()
        self.features = nn.Sequential(*features)
        self.lwhiten = lwhiten
        self.pool = pool
        self.whiten = whiten
        self.norm = L2N()
        self.meta = meta
    
    def forward(self, x):
        # x -> features
        o = self.features(x)

        # TODO: properly test (with pre-l2norm and/or post-l2norm)
        # if lwhiten exist: features -> local whiten
        if self.lwhiten is not None:
            # o = self.norm(o)
            s = o.size()
            o = o.permute(0,2,3,1).contiguous().view(-1, s[1])
            o = self.lwhiten(o)
            o = o.view(s[0],s[2],s[3],self.lwhiten.out_features).permute(0,3,1,2)
            # o = self.norm(o)

        # features -> pool -> norm
        o = self.norm(self.pool(o)).squeeze(-1).squeeze(-1)

        # if whiten exist: pooled features -> whiten -> norm
        if self.whiten is not None:
            o = self.norm(self.whiten(o))

        # permute so that it is Dx1 column vector per image (DxN if many images)
        return o.permute(1,0)

    def __repr__(self):
        tmpstr = super(ImageRetrievalNet, self).__repr__()[:-1]
        tmpstr += self.meta_repr()
        tmpstr = tmpstr + ')'
        return tmpstr

    def meta_repr(self):
        tmpstr = '  (' + 'meta' + '): dict( \n' # + self.meta.__repr__() + '\n'
        tmpstr += '     architecture: {}\n'.format(self.meta['architecture'])
        tmpstr += '     pooling: {}\n'.format(self.meta['pooling'])
        tmpstr += '     whitening: {}\n'.format(self.meta['whitening'])
        tmpstr += '     outputdim: {}\n'.format(self.meta['outputdim'])
        tmpstr += '     mean: {}\n'.format(self.meta['mean'])
        tmpstr += '     std: {}\n'.format(self.meta['std'])
        tmpstr = tmpstr + '  )\n'
        return tmpstr


def init_network(params):

    # parse params with default values
    architecture = params.get('architecture', 'resnet101')
    pooling = params.get('pooling', 'gem')
    whitening = params.get('whitening', False)
    mean = params.get('mean', [0.485, 0.456, 0.406])
    std = params.get('std', [0.229, 0.224, 0.225])
    pretrained = params.get('pretrained', True)

    # get output dimensionality size
    dim = OUTPUT_DIM[architecture]

    # loading network from torchvision
    # initialize with random weights, later on we will fill features with custom pretrained network
    net_in = getattr(torchvision_.models, architecture)(pretrained=False)

    # initialize features
    # take only convolutions for features,
    # always ends with ReLU to make last activations non-negative
    if architecture.startswith('alexnet'):
        features = list(net_in.features.children())[:-1]
    elif architecture.startswith('vgg'):
        features = list(net_in.features.children())[:-1]
    elif architecture.startswith('resnet'):
        features = list(net_in.children())[:-2]
    elif architecture.startswith('densenet'):
        features = list(net_in.features.children())
        features.append(nn.ReLU(inplace=True))
    elif architecture.startswith('squeezenet'):
        features = list(net_in.features.children())
    else:
        raise ValueError('Unsupported or unknown architecture: {}!'.format(architecture))

    # initialize local whitening
    lwhiten = None
    
    # initialize pooling
    pool = POOLING[pooling]()

    # initialize whitening
    whiten = None
    # whiten = nn.Linear(dim, dim, bias=True)

    # create meta information to be stored in the network
    meta = {
        'architecture' : architecture, 
        'pooling' : pooling,  
        'whitening' : whitening, 
        'mean' : mean, 
        'std' : std,
        'outputdim' : dim,
    }

    # create a generic image retrieval network
    net = ImageRetrievalNet(features, lwhiten, pool, whiten, meta)

    return net


def extract_vectors(net, images, image_size, transform, bbxs=None, ms=[1], msp=1, print_freq=1000):
    # moving network to gpu and eval mode
    net.cuda()
    net.eval()

    # creating dataset loader
    loader = torch.utils.data.DataLoader(
        ImagesFromList(root='', images=images, imsize=image_size, bbxs=bbxs, transform=transform),
        batch_size=1, shuffle=False, num_workers=8, pin_memory=True
    )

    # extracting vectors
    with torch.no_grad():
        if net.meta['pooling'] == 'rmatch':
            vecs = []
            for i, input in enumerate(loader):
                input = input.cuda()
                vec = extract_ss(net,input)
                vecs.append(vec)
                if (i+1) % print_freq == 0 or (i+1) == len(images):
                    print('\r>>>> {}/{} done...'.format((i+1), len(images)), end='')
        else:
            vecs = torch.zeros(net.meta['outputdim'], len(images))
            for i, input in enumerate(loader):
                input = input[0].cuda()

                if len(ms) == 1 and ms[0] == 1:
                    vecs[:, i] = extract_ss(net, input)
                else:
                    vecs[:, i] = extract_ms(net, input, ms, msp)

                if (i+1) % print_freq == 0 or (i+1) == len(images):
                    print('\r>>>> {}/{} done...'.format((i+1), len(images)), end=' ')
        print('')

    return vecs

def extract_ss(net, input):
    return net(input).cpu().data.squeeze()

def extract_ms(net, input, ms, msp):
    
    v = torch.zeros(net.meta['outputdim'])
    
    for s in ms: 
        if s == 1:
            input_t = input.clone()
        else:    
            input_t = nn.functional.interpolate(input, scale_factor=s, mode='bilinear', align_corners=False)
        v += net(input_t).pow(msp).cpu().data.squeeze()
        
    v /= len(ms)
    v = v.pow(1./msp)
    v /= v.norm()

    return v


def extract_regional_vectors(net, images, image_size, transform, bbxs=None, ms=[1], msp=1, print_freq=10):
    # moving network to gpu and eval mode
    net.cuda()
    net.eval()

    # creating dataset loader
    loader = torch.utils.data.DataLoader(
        ImagesFromList(root='', images=images, imsize=image_size, bbxs=bbxs, transform=transform),
        batch_size=1, shuffle=False, num_workers=8, pin_memory=True
    )

    # extracting vectors
    with torch.no_grad():
        vecs = []
        for i, input in enumerate(loader):
            input = input.cuda()

            if len(ms) == 1:
                vecs.append(extract_ssr(net, input))
            else:
                # TODO: not implemented yet
                # vecs.append(extract_msr(net, input, ms, msp))
                raise NotImplementedError

            if (i+1) % print_freq == 0 or (i+1) == len(images):
                print('\r>>>> {}/{} done...'.format((i+1), len(images)), end='')
        print('')

    return vecs

def extract_ssr(net, input):
    return net.pool(net.features(input), aggregate=False).squeeze(0).squeeze(-1).squeeze(-1).permute(1,0).cpu().data


def extract_local_vectors(net, images, image_size, transform, bbxs=None, ms=[1], msp=1, print_freq=10):
    # moving network to gpu and eval mode
    net.cuda()
    net.eval()

    # creating dataset loader
    loader = torch.utils.data.DataLoader(
        ImagesFromList(root='', images=images, imsize=image_size, bbxs=bbxs, transform=transform),
        batch_size=1, shuffle=False, num_workers=8, pin_memory=True
    )

    # extracting vectors
    with torch.no_grad():
        vecs = []
        for i, input in enumerate(loader):
            input = input.cuda()

            if len(ms) == 1:
                vecs.append(extract_ssl(net, input))
            else:
                # TODO: not implemented yet
                # vecs.append(extract_msl(net, input, ms, msp))
                raise NotImplementedError

            if (i+1) % print_freq == 0 or (i+1) == len(images):
                print('\r>>>> {}/{} done...'.format((i+1), len(images)), end='')
        print('')

    return vecs

def extract_ssl(net, input):
    return net.norm(net.features(input)).squeeze(0).view(net.meta['outputdim'], -1).cpu().data