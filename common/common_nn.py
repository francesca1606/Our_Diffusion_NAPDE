# -*- coding: utf-8 -*-
#!/usr/bin/env python
u'''Required modules'''
from curses import noecho
import optparse
import re
import warnings
from scipy.stats import kstest
from matplotlib.pyplot import sca
warnings.filterwarnings("ignore")
# COMMON
from common.common_model import AKA
# NUMPY
import copy
import numpy as np
# TORCH GENERIC
#from torch._jit_internal import weak_module, weak_script_method
# NN GENERIC
import torch
from torch import FloatTensor as tFT
from torch import nn
from torch.nn import *
from torch.nn import functional as F
from torch.nn import Sequential as sqn
from torch.nn import Linear as lin
from torch.nn.parallel import data_parallel as pll
from torch.nn import DataParallel
from torch import device as tdev
# LOSSES
from torch.nn.modules.loss import _Loss
from torch.nn import MSELoss as MSE
from torch.nn import NLLLoss as NLL
from torch.nn import BCELoss as BCE
from torch.nn import CrossEntropyLoss as CEL
# OPTIMIZER
from torch.optim import Adam, RMSprop, SGD
from itertools import repeat as ittr
from itertools import chain as ittc
from configuration import app
# AUTOGRAD
from torch.autograd import Variable
from torch.autograd import grad as torch_grad

# COMMON TORCH
from common.common_torch import tfnp,tcat,tavg
from common.common_torch import trnd,ln0c,tnrm,ttns
# NOISE 
from tools.generate_noise import noise_generator
from functools import partial
from pathlib import Path
import os
rndm_args = {'mean': 0, 'std': 1}
eps = 1e-15  # to avoid possible numerical instabilities during backward
b1 = 0.5
b2 = 0.9999

def dir_setup(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)
  
# def get_features(t_img,model,layer,szs):
#     
#      Create a vector of zeros that will hold our feature vector
#     my_embedding = torch.zeros(szs)
#      Define a function that will copy the output of a layer
#     def copy_data(m,i,o):
#         my_embedding.copy_(m.bckward_pass(o).data)
#      Attach that function to our selected layer
#     h = layer.register_forward_hook(copy_data)
#      Run the model on our transformed image
#     model(t_img)
#      Detach our copy function from the layer
#     h.remove()
#     return my_embedding
# 
# class FeatureExtractor(Module):
#     def __init__(self, cnn, feature_layer=11):
#         super(FeatureExtractor, self).__init__()
#         self.features = sqn(*list(cnn.features.children())[:(feature_layer+1)])
#     def forward(self, x):
#         return self.features(x)
# class LayerActivations():    
#     features=[]        
#     def __init__(self,model):        
#         self.features = []        
#         self.hook = model.register_forward_hook(self.hook_fn)        
#         def hook_fn(self,module,input,output):                
#             self.features.extend(output.view(output.size(0),-1).cpu().data)        
#         def remove(self):                
#             self.hook.remove()

class Exp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i.exp()
        ctx.save_for_backward(result)
        return result
    @staticmethod
    def backward(ctx, grad_output):
        breakpoint()
        result, = ctx.saved_tensors
        return grad_output * result


class Feature_extractor(Module):
    def __init__(self,lay):
        super(Feature_extractor,self).__init__()
        self.lay=lay
    def forward(self,X):
        self.feature = self.lay(X)
        return X
  
# Add noise module
class AddNoise(Module):
    def __init__(self,dev=tdev("cpu")):
        super(AddNoise,self).__init__()
        self.dev = dev
    def forward(self,X):
        W,_,_ = noise_generator(X.shape,X.shape,self.dev,rndm_args)
        #if X.is_cuda:
        #    .cuda()
        #else:
        #    return zcat(X,W)
        return zcat(X,W) 
class Swish(Module):
    def __init__(self, train_beta=False):
        super(Swish, self).__init__()
        if train_beta:
            self.weight = Parameter(torch.Tensor([1.]))
        else:
            self.weight = 1.0

    def forward(self, x):
        return x * torch.sigmoid(self.weight * x)
    
# Dropout module
class Dpout(Module):
    def __init__(self,dpc=0.10):
        super(Dpout,self).__init__()
        self.dpc = dpc
    def forward(self,x):
        return F.dropout(x,p=self.dpc,\
                         training=self.training)

class Flatten(Module):
    def __init__(self):
        super(Flatten,self).__init__()
    def forward(self,x):
        # return x.view(-1,1).squeeze(1)
        return torch.flatten(x,1)

class Squeeze(Module):
    def __init__(self, dim=1):
        super(Squeeze,self).__init__()
        self.dim = dim
        
    def forward(self,x):
        return x.squeeze(self.dim)


class UnSqueeze(Module):
    def __init__(self, dim = 1):
        super(UnSqueeze,self).__init__()
        self.dim =  dim
    def forward(self,x):
        return x.unsqueeze(self.dim)

class Shallow(Module):
    def __init__(self, shape):
        super(Shallow,self).__init__()
        self.shape =  shape

    def forward(self, x):
        batch, dimension = self.shape
        return torch.reshape(x,(batch, dimension))
        

class Explode(Module):
    def __init__(self, shape):
        super(Explode,self).__init__()
        self.shape = shape

    def forward(self,x):
        return x.view(-1,*self.shape)


def cnn1d(in_channels,out_channels,act=LeakyReLU(1.0,inplace=True),
            use_weight_regularization=False,bn=True,ker=7,std=4,pad=0, 
            regularization_weight= partial(torch.nn.utils.spectral_norm),\
            normalization=partial(BatchNorm1d),dil=1,grp=1,dpc=None,wn=False,dev=tdev("cpu"),
            bias = False, *args, **kwargs):

    block = [Conv1d(in_channels=in_channels,out_channels=out_channels,\
            kernel_size=ker,stride=std,padding=pad,dilation=dil,groups=grp,bias=bias)]
    
    #weight regularization for the block CNN
    if use_weight_regularization:
        block = [regularization_weight(copy.deepcopy(block[0]))]
    
    # type of normalization for  block CNN
    # It is recommanded to use BatchNorm1d for Generators(Encoder/ Decoder) 
    # to acheive better performance during the Training. 
    # Use InstanceNorm1d, LayerNorm, etc ... in discriminator case only
    if bn:
        if isinstance(normalization,nn.InstanceNorm1d):
            block.append(normalization(out_channels, affine=True))
        else:
            block.append(normalization(out_channels))
    block.append(act)
    if dpc is not None:
        block.append(Dpout(dpc=dpc))
    if wn:
        block.append(AddNoise(dev=dev))
    return block

def cnn1dt(in_channels,out_channels,act=LeakyReLU(1.0),use_weight_regularization=False,
           bn=True,ker=2,std=2,pad=0,opd=0,regularization_weight= partial(torch.nn.utils.spectral_norm),
           normalization=partial(BatchNorm1d),dil=1,grp=1,dpc=None, bias=False, *args, **kwargs):

    block = [ConvTranspose1d(in_channels=in_channels,out_channels=out_channels,\
                kernel_size=ker,stride=std,output_padding=opd,padding=pad,\
                dilation=dil,groups=grp,bias=bias)]
    
    if use_weight_regularization:
        block = [regularization_weight(copy.deepcopy(block[0]))]
    
    if bn:
        if isinstance(normalization,nn.InstanceNorm1d):
            block.append(normalization(out_channels, affine=True))
        else:
            block.append(normalization(out_channels))
    
    block.append(act)
    if dpc is not None:
        block.append(Dpout(dpc=dpc))
    return block

def DenseBlock(in_channels,out_channels,act=[Sigmoid()],dpc=0.1,bias=False):
    block = [Linear(in_channels, out_channels,bias=bias),act,Dpout(dpc=dpc)]
    return block


class ConditionalBatchNorm(nn.Module):
    """ View source : 
    https://github.com/ap229997/Conditional-Batch-Norm/blob/6e237ed5794246e1bbbe95bbda9acf81d0cdeace/model/cbn.py#L9
    """
    def __init__(self, lstm_size, emb_size, out_size, batch_size, channels, width, use_betas=True, use_gammas=True, eps=1.0e-5):
        super(ConditionalBatchNorm, self).__init__()

        self.lstm_size  = lstm_size # size of the lstm emb which is input to MLP
        self.emb_size   = emb_size # size of hidden layer of MLP
        self.out_size   = out_size # output of the MLP - for each channel
        self.use_betas  = use_betas
        self.use_gammas = use_gammas

        self.batch_size = batch_size
        self.channels   = channels
        # self.height     = height
        self.width      = width

        # beta and gamma parameters for each channel - defined as trainable parameters
        self.betas = nn.Parameter(torch.zeros(self.batch_size, self.channels).cuda())
        self.gammas = nn.Parameter(torch.ones(self.batch_size, self.channels).cuda())
        self.eps = eps

        # MLP used to predict betas and gammas
        self.fc_gamma = nn.Sequential(
            nn.Linear(self.lstm_size, self.emb_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.emb_size, self.out_size),
            ).cuda()

        self.fc_beta = nn.Sequential(
            nn.Linear(self.lstm_size, self.emb_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.emb_size, self.out_size),
            ).cuda()

        # initialize weights using Xavier initialization and biases with constant value
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform(m.weight)
                nn.init.constant(m.bias, 0.1)

    '''
    Predicts the value of delta beta and delta gamma for each channel
    Arguments:
        lstm_emb : lstm embedding of the question
    Returns:
        delta_betas, delta_gammas : for each layer
    '''
    def create_cbn_input(self, lstm_emb):

        if self.use_betas:
            delta_betas = self.fc_beta(lstm_emb)
        else:
            delta_betas = torch.zeros(self.batch_size, self.channels).cuda()

        if self.use_gammas:
            delta_gammas = self.fc_gamma(lstm_emb)
        else:
            delta_gammas = torch.zeros(self.batch_size, self.channels).cuda()

        return delta_betas, delta_gammas

    '''
    Computer Normalized feature map with the updated beta and gamma values
    Arguments:
        feature : feature map from the previous layer
        lstm_emb : lstm embedding of the question
    Returns:
        out : beta and gamma normalized feature map
        lstm_emb : lstm embedding of the question (unchanged)
    Note : lstm_emb needs to be returned since CBN is defined within nn.Sequential
           and subsequent CBN layers will also require lstm question embeddings
    '''
    def forward(self, feature, lstm_emb):
        self.batch_size, self.channels, self.width = feature.data.shape
        # get delta values
        delta_betas, delta_gammas = self.create_cbn_input(lstm_emb)

        betas_cloned = self.betas.clone()
        gammas_cloned = self.gammas.clone()

        # update the values of beta and gamma
        betas_cloned += delta_betas
        gammas_cloned += delta_gammas

        # get the mean and variance for the batch norm layer
        batch_mean = torch.mean(feature)
        batch_var = torch.var(feature)

        # extend the betas and gammas of each channel across the height and width of feature map
        # betas_expanded = torch.stack([betas_cloned]*self.height, dim=2)
        betas_expanded = torch.stack([betas_expanded]*self.width, dim=2)

        # gammas_expanded = torch.stack([gammas_cloned]*self.height, dim=2)
        gammas_expanded = torch.stack([gammas_expanded]*self.width, dim=3)

        # normalize the feature map
        feature_normalized = (feature-batch_mean)/torch.sqrt(batch_var+self.eps)

        # get the normalized feature map with the updated beta and gamma values
        out = torch.mul(feature_normalized, gammas_expanded) + betas_expanded
        return out, lstm_emb

class PatchCNNBlock(nn.Module):
    """ ConvBlock specialy design for PatchGAN architcture discriminator
    """
    def __init__(self, in_channels, out_channels, stride):
        super(PatchCNNBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(
                in_channels, out_channels, 4, stride, 1, bias=False, padding_mode="reflect"
            ),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(0.2),
        )
    def forward(self, x):
        return self.conv(x)
    

class ConvBlock(Module):
    def __init__(self, ni, no, ker, std, bias=False,use_weight_regularization=False,
        act = None, bn=True, regularization_weight= partial(torch.nn.utils.spectral_norm),
        normalization = partial(BatchNorm1d), pad=None, dpc=None, dil = 1,*args, **kwargs):
        super(ConvBlock,self).__init__()
        if pad is None: pad = ks//2//stride

        self.ann = [Conv1d(in_channels = ni, out_channels= no, kernel_size=ker, stride = std, 
                    padding=pad, bias=bias, dilation = dil)]

        if use_weight_regularization:
            self.ann = [regularization_weight(copy.deepcopy(self.ann[0]))]

        if bn:
            if isinstance(normalization,InstanceNorm1d):
                self.ann+= [normalization(no, affine=True)]
            else:
                self.ann+= [BatchNorm1d(no)]
                
        if dpc is not None: self.ann += [Dpout(dpc=dpc)]
        if act is not None: self.ann += [act]

        self.ann = sqn(*self.ann)

    def forward(self, x):
        z = self.ann(x)
        return z



class DeconvBlock(Module):
    def __init__(self,ni,no,ks,stride,pad,opd=0,bn=True,act=ReLU(inplace=True),
                 dpc=None):
        super(DeconvBlock,self).__init__()
        self.conv = ConvTranspose1d(ni, no, ks, stride, 
                                    padding=pad, output_padding=opd,
                                    bias=False)
        self.relu = act
        self.bn = BatchNorm1d(no)
        
        self.dpout = Dpout(dpc=dpc)
    def forward(self, x):
        x = self.relu(self.conv(x))
        return self.dpout(self.bn(x)) if self.bn else self.dpout(x)
    
class ResConvBlock(Module):
    def __init__(self, ni, no, ks, stride, bias=False,
                 act = None, bn=True, pad=None, dpc=None):
        super(ResConvBlock,self).__init__()
        
        # 1st block
        if pad is None: pad = ks//2//stride
        self.ann = [Conv1d(ni, no, ks, stride, padding=pad, bias=bias)]
        if bn: self.ann += [BatchNorm1d(no)]
        if dpc is not None: self.ann += [Dpout(dpc=dpc)]
        
        # 2nd block
#         stride = 1
#         ks = ks-1
#         if pad is None: pad = ks//2//stride
        self.ann += [Conv1d(no, no, ks, stride, padding=pad, bias=bias)]
        if bn: self.ann += [BatchNorm1d(no)]
        if dpc is not None: self.ann += [Dpout(dpc=dpc)]
        # activation
        if act is not None: self.ann += [act]

        self.ann = sqn(*self.ann)
        
    def forward(self, x):
        return self.ann(x) + x

class ResidualBlock(Module):
    def __init__(self, in_channels, out_channels, activation='relu'):
        super(ResidualBlock,self).__init__()
        self.in_channels = in_channels,
        self.out_channels = out_channels
        self.activation = activation
        # 1st block
        self.block = nn.Identity()
        self.activate = self.activation_function(self.activation)
        self.shortcut = nn.Identity() 
        
    def forward(self, x):
        # return self.ann(x) + self.ann._modules['0'](x)
        residual  = x
        if self.should_apply_shortcut: residual =  self.shortcut(x)
        x = self.block(x)
        x +=residual
        x = self.activate(x)
        return x


    def activation_function(self, activation):
        return  nn.ModuleDict([
            ['relu', nn.ReLU(inplace=True)],
            ['leaky_relu', nn.LeakyReLU(negative_slope=0.01, inplace=True)],
            ['selu', nn.SELU(inplace=True)],
            ['none', nn.Identity()]
        ])[activation]
    @property
    def should_apply_shortcut(self):
        return self.in_channels != self.out_channels

class ResNetResidualBlock(ResidualBlock):
    def __init__(self,in_channels, out_channels, expansion=1, conv=cnn1d, downsampling=1, *args, **kwargs):
        super().__init__(in_channels,out_channels)
        self.expansion = expansion
        self.downsampling = downsampling
        self._conv = conv
        self.ann = self._conv(in_channels, self.expanded_channels, ker = 1, std = 1, pad= 0)
        # self.shortcut = sqn(*self.ann, nn.BatchNorm1d(self.expanded_channels)) if self.should_apply_shortcut else None

    @property
    def expanded_channels(self):
        return self.out_channels*self.expansion
    @property
    def should_apply_shortcut(self):
        return self.in_channels != self.expanded_channels
    
class ResNetBasicBlock(ResNetResidualBlock):
    def __init__(self, in_channels, out_channels, conv, *args, **kwargs):
        super().__init__(in_channels, out_channels, *args, **kwargs)
        self.ann1 = self.conv_bn(in_channels,out_channels, conv=conv, ker=1, std=1)
        self.ann2 = self.conv_bn(out_channels, self.expanded_channels,conv=conv,ker=1, std=1)
        self.shortcut = self.short_bn(in_channels, out_channels,conv=cnn1d,*args, ** kwargs)
        self.block = sqn(*self.ann1,self.activation_function(self.activation),*self.ann2)

    def conv_bn(self, in_channels, out_channels, conv, *args, **kwargs):
        return sqn(*conv(in_channels, out_channels,*args, **kwargs), nn.BatchNorm1d(out_channels))
    def short_bn(self, in_channels, out_channels, conv, *args, **kwargs):
        #work as an identity if ker = std =1
        ann = conv(in_channels, out_channels, ker=1, std=1, pad=0)
        return  sqn(*ann, nn.BatchNorm1d(self.expanded_channels)) if self.should_apply_shortcut else None

class ResNetLayer(Module):
    def __init__(self,in_channels, out_channels, block=ResNetBasicBlock, conv=cnn1d, n=1, *args, **kwargs):
        super().__init__()
        downsampling = 2 if in_channels !=out_channels else 1
        _block = block(in_channels, out_channels, conv=conv,
            *args, **kwargs, downsampling = downsampling)
        expansion = _block.expansion
        self.Resblock = sqn(
            _block,
            *[block(out_channels*expansion,
                 out_channels, downsampling=1, conv=conv,*args, **kwargs) for _ in range(n - 1)]
            )

    def forward(self, x):
        x = self.Resblock(x)
        return x

u'''[Zeroed gradient for selected optimizers]'''
def zerograd(optz):
    for o in optz: 
        if o is not None:
            o.zero_grad()
    # exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
    # except:
    #     _optim = [o.optimizer for o in optz]
    #     for o in _optim:
    #         if o is not None:
    #             o.zero_grad()

def scheduler_step(scheduler, *args, **kwargs):
    for schedule in scheduler:
        schedule.step(*args, **kwargs)


def patch(y,x = None):
    if torch.isnan(y).any():
        """
        print("your model contain nan value "
            "this signals will be withdrawn from the training "
            "but style be present in the dataset. \n"
            "Then, remember to correct your dataset")
        """
        mask   = [not torch.isnan(torch.max(y[e,:])).tolist() for e in range(len(y))]
        index  = np.array(range(len(y)))
        y.data = y[index[mask]]
        if x is not None:
            x.data = x[index[mask]]
    return y, x 
        
def penalty(loss,params,typ,lam=1.e-5):
    pen = {'L1':1,'L2':2}
    reg = ttns(0.)
    for p in params:
        reg += tnrm(p,pen[typ])
    return loss + lam*reg

def gan_loss(yp,yt,reduction=True):
    if reduction:
        return -tavg(ln0c(yt.cpu()) + ln0c(1.0-yp.cpu()))
    else:
        return -ln0c(yt.cpu()) - ln0c(1.0-yp.cpu())
    
def softplus(_x):
    return torch.log(1.0 + torch.exp(_x))
    
#@weak_module
class GANLoss(_Loss):
    __constants__ = ['reduction']
    __metaclass__ = AKA
    aliases = 'GAN'
    def __init__(self, size_average=None, reduce=None, reduction='mean'):
        super(GANLoss, self).__init__(size_average, reduce, reduction)

    #@weak_script_method
    def forward(self, inp,tar):
        return gan_loss(inp,tar,reduction=self.reduction)
GAN = GANLoss

#@weak_module
class ALILoss(_Loss):
    __metaclass__ = AKA
    aliases = 'ALI'
    def __init__(self, dloss = True):
        super(ALILoss, self).__init__(dloss)
        self.dloss=dloss
    #@weak_script_method
    def forward(self,sample_preds,data_preds):
        if self.dloss:
            # discriminator loss
            return torch.mean(softplus(-data_preds-eps) + softplus(sample_preds+eps))
        else:
            # generator loss
            return torch.mean(softplus(data_preds+eps) + softplus(-sample_preds-eps))

ALI = ALILoss

def zcat(*args):
    return tcat(args,1)

def is_gaussian(distribution):
    _distribution = distribution.cpu().data.numpy()
    stat, pvalue = kstest(_distribution.reshape(-1),'norm')
    return stat, pvalue


# custom weights initialization called on netG and netD¬                 
def set_weights(m):                                 
    classname = m.__class__.__name__               
    if (classname.find('Conv1d') != -1 
            or
        classname.find('ConvTranspose1d') != -1):                   
        try:
            # init.xavier_uniform(m.weight)
            # init.xavier_normal_(m.weight)
            # init.kaiming_uniform(m.weight)
            init.normal_(m.weight, mean=0.0, std=0.02)
        except:
            print("warnings no initialization is made training may not work")
        #m.weight.data.normal_(0.0, 0.02) 
    elif (classname.find('Linear')!=-1): 
        m.weight.data.normal_(0.0, 1.0)
        # m.bias.data.fill_(0)                               
    elif (classname.find('BatchNorm') !=-1):
        m.weight.data.normal_(1.0, 0.02)                          
        m.bias.data.fill_(0)

def tie_weights(m):
    for _,n in m.__dict__['_modules'].items():
        try:
            n.bckward_pass[0].weight = n.forward_pass[0].weight
        except: 
            pass

def generate_latent_variable_3D(batch, nch_zd=4, nch_zf = 4, nzd=128, nzf = 128,std=1):
        zyy  = torch.zeros([batch,nch_zd,nzd]).normal_(mean=0,std=std).to(app.DEVICE, non_blocking = True)
        zxx  = torch.zeros([batch,nch_zd,nzd]).normal_(mean=0,std=std).to(app.DEVICE, non_blocking = True)
        zyx  = torch.zeros([batch,nch_zf,nzf]).normal_(mean=0,std=std).to(app.DEVICE, non_blocking = True)
        zxy  = torch.zeros([batch,nch_zf,nzf]).normal_(mean=0,std=std).to(app.DEVICE, non_blocking = True)
        return zyy, zyx, zxx, zxy

def generate_latent_variable(batch, nch = 1, nzd = 512, std=1.0):
    zyy  = torch.zeros([batch,nzd]).normal_(mean=0.,std=std).to(app.DEVICE, non_blocking = True)
    zxy  = torch.zeros([batch,nzd]).normal_(mean=0.,std=std).to(app.DEVICE, non_blocking = True)
    return zyy, zxy

def get_accuracy(tag, plot_function,encoder, decoder, vld_loader,*args, **kwargs):
    with torch.no_grad():
        def _eval(EG,PG): 
            val = np.sqrt(np.power([10 - eg for eg in EG],2)+\
                np.power([10 - pg for pg in PG],2))
            accuracy = val.mean().tolist()
            return accuracy

        EG_h, PG_h  = plot_function(tag = tag, 
            Qec = encoder, Pdc = decoder, 
            trn_set = vld_loader, 
            *args, **kwargs)

        return _eval(EG_h,PG_h)

def scheduler(scheduler_name,optimizer,*args,**kwargs):
    scale_fn = None
    if scheduler_name == 'CyclicLR':
        scale_fn = torch.optim.lr_scheduler.CyclicLR(optimizer,*args,**kwargs)
    elif scheduler_name == 'MultiStepLR':
        scale_fn = torch.optim.lr_scheduler.MultiStepLR(optimizer,*args, **kwargs)
    elif scheduler_name == 'LinearLR':
        scale_fn = torch.optim.lr_scheduler.LinearLR(optimizer,*args,**kwargs)
    else:
        scale_fn = torch.optim.lr_scheduler.ConstantLR(optimizer,*args,**kwargs)
    return scale_fn

def reset_net(nets,func=set_weights,lr=0.0002,b1=b1,b2=b2,
    weight_decay=None,optim='Adam',scheduler=scheduler,scheduler_name=None, *args, **kwargs):
    p = []
    for n in nets:
        n.apply(func)
        p.append(n.parameters())
    if 'adam' in optim.lower():
        if  weight_decay is None:
            return Adam(ittc(*p),lr=lr,betas=(b1,b2))
        else:
            return Adam(ittc(*p),lr=lr,betas=(b1,b2),weight_decay=weight_decay)
    elif 'rmsprop' in optim.lower():
        return RMSprop(ittc(*p),lr=lr, *args, **kwargs)
    elif 'sgd' in optim.lower():
        return SGD(ittc(*p),lr=lr)

def count_parameters(models):
    from prettytable import PrettyTable
    table = PrettyTable(["Modules", "Parameters","Network Names"])
    table.int_format = '6'
    total_params = 0
    for model in models:
        try:
            param = model.module.number_parameter
            module_name  = model.module.__class__.__name__
            network_name = model.module.model_name
        except:
            param = model.number_parameter
            module_name  = model.__class__.__name__
            network_name = model.model_name
        table.add_row([module_name, param,network_name])
        total_params+=param
    print(table)
    print("Total Trainable Params: {:,}".format(total_params))

def modalite(nets, mode='train'):
    if mode == 'train':
        for n in nets:
            n.train()
    elif mode == 'eval':
        for n in nets:
            n.eval()


def clipweights(netlist,lb=-0.01,ub=0.01):
    for D in netlist:
        for p in D.parameters():
            p.data.clamp_(lb,ub)
    
def dump_conf(m):                                 
    classname = m.__class__.__name__               
    if classname.find('Conv1d') != -1 or classname.find('ConvTranspose1d') != -1:
        print('cuda-check')                       
        print(classname)
        print(m.weight.is_cuda)

#def zero_dpout(m):
#    for name, param in m.named_children():
#        if isinstance(param,Sequential):
#            import pdb
#            pdb.set_trace()

def to_categorical(y,c2type,num_columns=1):
    """Returns one-hot encoded Variable"""
    y_cat = np.zeros((y.shape[0], num_columns))
    y_cat[range(y.shape[0]), y] = 1.
    return Variable(c2type(y_cat))

def get_categorical(labels, n_classes=10):
    cat = np.array(labels.data.tolist())
    cat = np.eye(n_classes)[cat].astype('float32')
    cat = tfnp(cat)
    return Variable(cat)

def runout(funct, world_size):
    mp.spawn(funct,
             args=(world_size,),
             nprocs=world_size,
             join=True)

"""
def hessian_penalty(G, z, k, epsilon):
    # Input G: Function to compute the Hessian Penalty of
    # Input z: Input to G that the Hessian Penalty is taken w.r.t.
    # Input k: Number of Hessian directions to sample
    # Input epsilon: Finite differences hyperparameter
    # Output: Hessian Penalty loss
    G_z = G(z)
    #https://www.geeksforgeeks.org/sympy-stats-rademacher-function-in-python/
    vs = epsilon * random_rademacher(shape=[k, *z.size()])
    finite_diffs = [G(z + v) - 2 * G_z + G(z - v) for v in vs]
    finite_diffs = stack(finite_diffs) / (epsilon ** 2)
    penalty = var(finite_diffs, dim=0).max()
    return penalty
"""

class T(object):
    """docstring for T"""
    def __init__(self):
        super(T, self).__init__()
        pass

    @staticmethod
    def activation_func(activation):
        return {
            'relu': nn.ReLU(inplace=True),
            'leaky_relu': nn.LeakyReLU(negative_slope=1.0, inplace=True),
            'selu': nn.SELU(inplace=True),
            'none': nn.Identity()
            }[activation]

    @staticmethod 
    def activation(name, nly, typo = 'ALICE'):
        acts = {}
        acts['ALICE'] = {
                 'Feu' :[nn.LeakyReLU(0.2,inplace=True) for t in range(1, nly+1)],
                 'FPReLU':[nn.PReLU() for t in range(1, nly+1)],
                 'F2'  :[nn.ReLU(inplace=True) for t in range(1, nly+1)],
                 'Fed' :[nn.LeakyReLU(0.2,inplace=True) for t in range(1, nly)] + [nn.LeakyReLU(1.0,inplace=True)],
                 'Gdd' :[nn.ReLU(inplace=True) for t in range(1, nly)]+[nn.Tanh()],
                 'Gdd_Unic' :[nn.ReLU(inplace=True) for t in range(1, nly+1)],
                 'Gdd_Tanh':[nn.Tanh() for t in range(1, nly+1)],
                 'Gdd_PReLU':[nn.PReLU() for t in range(1, nly)]+[nn.Tanh()],
                 'Gdd_SELU' :[nn.SELU(inplace=True) for t in range(1, nly)]+[nn.LeakyReLU(1.0, inplace=True)],
                 'Gdd_Leaky' :[nn.LeakyReLU(1.0,inplace=True) for t in range(1, nly)]+[nn.LeakyReLU(1.0, inplace=True)],
                 'Gdd2':[nn.ReLU(inplace=True) for t in range(1, nly)]+[nn.LeakyReLU(1.0,inplace=True)],
                 'Fef' :[nn.LeakyReLU(1.0,inplace=True) for t in range(1, nly)]+[nn.LeakyReLU(1.0,inplace=True)],
                 'Gdf' :[nn.ReLU(inplace=True) for t in range(1, nly)]+[nn.Tanh()],
                 'Ghz' :[nn.ReLU(inplace=True) for t in range(1, nly)]+[nn.ReLU(inplace=True)],
                 'Fhz' :[nn.ReLU(inplace=True) for t in range(1, nly)]+[nn.LeakyReLU(1.0,inplace=True)],
                 'Dsx' :[nn.LeakyReLU(1.0,inplace=True) for t in range(1, nly+1)],
                 'Dfx' :[nn.LeakyReLU(0.2,inplace=True) for t in range(1, nly+1)],
                 'Dsz' :[nn.LeakyReLU(1.0,inplace=True) for t in range(1, nly+1)],
                 'Dyz' :[nn.LeakyReLU(0.2,inplace=True) for t in range(1, nly)] + [nn.LeakyReLU(1.0,inplace=True)],
                 'Drx' :[nn.LeakyReLU(1.0,inplace=True) for t in range(1, nly)] + [nn.Sigmoid()],
                 'Dss' :[nn.LeakyReLU(0.2,inplace=True) for t in range(1, nly)] + [nn.Sigmoid()],
                 'Drz' :[nn.LeakyReLU(0.2,inplace=True) for t in range(1, nly+1)] + [nn.Sigmoid()],
                 'Drrz':[nn.LeakyReLU(0.2,inplace=True) for t in range(1, nly)] + [nn.Sigmoid()],
                 'Ddxz':[nn.LeakyReLU(0.2,inplace=True) for t in range(1, nly)] + [nn.Sigmoid()],
                 'Dfxz':[nn.LeakyReLU(0.2,inplace=True) for t in range(1, nly+1)] + [nn.Sigmoid()],
                 'DhXd':[nn.LeakyReLU(0.2,inplace=True) for t in range(1, nly+1)] + [nn.Sigmoid()]
                 }

        acts['WGAN']  = {
                 'Fed' :[nn.LeakyReLU(1.0,inplace=True) for t in range(nly)]+[nn.LeakyReLU(1.0,inplace=True)],
                 'Gdd' :[nn.ReLU(inplace=True) for t in range(nly-1)]+[nn.Tanh()],
                 'Fef' :[nn.LeakyReLU(1.0,inplace=True) for t in range(4)]+[nn.LeakyReLU(1.0,inplace=True)],
                 'Gdf' :[nn.ReLU(inplace=True) for t in range(4)]+[nn.Tanh()],
                 'Ghz' :[nn.ReLU(inplace=True) for t in range(2)]+[nn.LeakyReLU(1.0,inplace=True)],
                 'Dsx' :[nn.LeakyReLU(1.0,inplace=True),nn.LeakyReLU(1.0,inplace=True)],
                 'Dsz' :[nn.LeakyReLU(1.0,inplace=True),nn.LeakyReLU(1.0,inplace=True)],
                 'Drx' :[nn.LeakyReLU(1.0,inplace=True) for t in range(nly+1)],
                 'Drz' :[nn.LeakyReLU(1.0,inplace=True) for t in range(nly+1)],
                 'Ddxz':[nn.LeakyReLU(1.0,inplace=True) for t in range(nly+1)],
                 'DhXd':[nn.LeakyReLU(1.0,inplace=True) for t in range(nly+1)]
                 }
        return acts[typo][name]

    @staticmethod
    def extraction_data(data= 'trn_loader'):
        limit = 100
        NS = []
        EW = []
        UD = []
        if data == 'trn_loader':
            for _,batch in enumerate(trn_loader):
                # Load batch
                # pdb.set_trace()
                xd_data,_,_,_,_,_,_ = batch
                Xd = Variable(xd_data)# BB-signal
                # zd = Variable(zd_data).to(ngpu-1)
                NS.append(Xd[:,0,:])
                EW.append(Xd[:,1,:])
                UD.append(Xd[:,2,:])
            # pdb.set_trace()
            NS = torch.cat(NS).numpy()
            EW = torch.cat(EW).numpy()
            UD = torch.cat(UD).numpy()
            signals  = np.vstack((NS[0:limit,:], EW[0:limit,:], UD[0:limit,:]))
        else:
            try:
                # pdb.set_trace()
                signals = torch.load(data)
                z1 = signals[:,0,:].cpu().detach().numpy()
                z2 = signals[:,1,:].cpu().detach().numpy()
                z3 = signals[:,0,:].cpu().detach().numpy()
                signals = np.vstack((z1,z2,z3))
            except Exception as  e:
                print('data not found !!!')
                raise e

        return signals

    @staticmethod
    def load_broadband_encoder():
        net = Module()
        try:
            net = torch.load("./network/trained/nzd32/Fed.pth")
        except Exception as e:
            raise e
        return net

    @staticmethod 
    def _forward(cnn, x, gang, split = 64):
        ret    = []
        splits = iter(x.split(split, dim = 0))
        s_next = next(splits)
        s_prev = pll(cnn,s_next, gang)

        for s_next in splits:
            # s_prev = self.cnn1(s_prev)
            ret.append(s_prev)
            s_prev = pll(cnn,s_next, gang)

        # s_prev = self.cnn1(s_prev)
        ret.append(s_prev)
        return torch.cat(ret)


    @staticmethod
    def _forward_1G(x, cnn1, split = 12):
        ret    = []
        splits = iter(x.split(split, dim = 0))
        s_next = next(splits)
        s_prev = cnn1(s_next).to(0)

        for s_next in splits:
            # s_prev = self.cnn1(s_prev)
            ret.append(s_prev)
            s_prev = cnn1(s_next).to(0)

        # s_prev = self.cnn1(s_prev)
        ret.append(s_prev)
        return torch.cat(ret).to(0)

    @staticmethod
    def _forward_2G(x, cnn1, cnn2, split = 20):
        # import pdb
        # pdb.set_trace()
        ret    = []
        splits = iter(x.split(split, dim = 0))
        s_next = next(splits)
        s_prev = cnn1(s_next).to(1)
        # import pdb
        # pdb.set_trace()

        for s_next in splits:
             # A. s_prev runs on cuda:1
            s_prev = cnn2(s_prev)
            ret.append(s_prev)

            # B. s_next runs on cuda:0, which can run concurrently with A
            s_prev = cnn1(s_next).to(1)

        s_prev = cnn2(s_prev)
        ret.append(s_prev)

        return torch.cat(ret)
    @staticmethod
    def _forward_3G(x, cnn1, cnn2, cnn3, split = 20):
        ret    = []
        splits = iter(x.split(split, dim = 0))
        s_next = next(splits)
        s_prev = cnn1(s_next).to(1)

        for s_next in splits:
            s_prev = cnn2(s_prev).to(2)
            s_prev = cnn3(s_prev)
            ret.append(s_prev)
            s_prev = cnn1(s_next).to(0)

        s_prev = cnn2(s_prev).to(2)
        s_prev = cnn3(s_prev)
        ret.append(s_prev)
        return torch.cat(ret).to(2)

    @staticmethod
    def _forward_4G(x, cnn1, cnn2, cnn3, cnn4, split = 20):
        ret    = []
        splits = iter(x.split(split, dim = 0))
        s_next = next(splits)
        s_prev = cnn1(s_next).to(1)

        for s_next in splits:
            s_prev = cnn2(s_prev).to(2)
            s_prev = cnn3(s_prev).to(3)
            s_prev = cnn4(s_prev)
            ret.append(s_prev)
            s_prev = cnn1(s_next).to(0)

        s_prev = cnn2(s_prev).to(2)
        s_prev = cnn3(s_prev).to(3)
        s_prev = cnn4(s_prev)
        ret.append(s_prev)
        return torch.cat(ret).to(3)

    @staticmethod
    def load_net(path):
        net = Module()
        try:
            net = torch.load(path)
        except Exception as e:
            raise e
        return net
