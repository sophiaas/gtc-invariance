import torch

from escnn import gspaces
from escnn import nn


class GonR3ConvBlock(torch.nn.Module):
    
    def __init__(self, 
                 n_channels, 
                 kernel_size, 
                 action,
                 N=None,
                 padding=1, 
                 bias=False, 
                 nonlinearity=None,
                 in_type=None,
                 **kwargs):
        super().__init__()
        self.n_channels = n_channels
        self.kernel_size = kernel_size, 
        self.padding = padding
        self.bias = bias
        if N is not None:
            self.g_act = action(N=N)
        else:
            self.g_act = action()
        self.N = N
        
        if in_type is None:
            self.in_type = in_type = nn.FieldType(self.g_act, 
                                                  [self.g_act.trivial_repr])
        else:
            self.in_type = in_type
            
        self.out_type = out_type = nn.FieldType(self.g_act, 
                                                n_channels * [self.g_act.regular_repr])
        
        conv = nn.R3Conv(in_type=in_type, 
                         out_type=out_type, 
                         kernel_size=kernel_size, 
                         padding=padding, 
                         bias=bias)
        
        batch_norm = nn.IIDBatchNorm3d(out_type, affine=True)
        
        sequence = [conv, batch_norm]
        
        if nonlinearity is not None:
            nonlinear_layer = nonlinearity(out_type, inplace=True)
            self.block = nn.SequentialModule(
                conv, 
                batch_norm,
                nonlinear_layer
        )
        else:
            self.block = nn.SequentialModule(
                conv, 
                batch_norm,
        )
            

    def forward(self, x):
        if type(x) != nn.GeometricTensor:
            x = self.in_type(x)
        return self.block(x)
    
    
class GonR2ConvBlock(torch.nn.Module):
    
    def __init__(self, 
                 N, 
                 n_channels, 
                 kernel_size, 
                 action,
                 padding=1, 
                 bias=False, 
                 nonlinearity=None,
                 in_type=None,
                 **kwargs):
        super().__init__()
        self.N = N
        self.n_channels = n_channels
        self.kernel_size = kernel_size, 
        self.padding = padding
        self.bias = bias
        self.g_act = action(N=N)
        
        if in_type is None:
            self.in_type = in_type = nn.FieldType(self.g_act, 
                                                  [self.g_act.trivial_repr])
        else:
            self.in_type = in_type
            
        self.out_type = out_type = nn.FieldType(self.g_act, 
                                                n_channels * [self.g_act.regular_repr])
        
        conv = nn.R2Conv(in_type=in_type, 
                         out_type=out_type, 
                         kernel_size=kernel_size, 
                         padding=padding, 
                         bias=bias)
        
        batch_norm = nn.InnerBatchNorm(out_type)
        
        sequence = [conv, batch_norm]
        
        if nonlinearity is not None:
            nonlinear_layer = nonlinearity(out_type, inplace=True)
            self.block = nn.SequentialModule(
                conv, 
                batch_norm,
                nonlinear_layer
        )
        else:
            self.block = nn.SequentialModule(
                conv, 
                batch_norm,
        )
            

    def forward(self, x):
        if type(x) != nn.GeometricTensor:
            x = self.in_type(x)
        return self.block(x)
    


class SO2onR2ConvBlock(torch.nn.Module):
    
    def __init__(self, 
                 N, 
                 n_channels, 
                 kernel_size, 
                 padding=1, 
                 bias=False, 
                 nonlinearity=nn.ReLU,
                 in_type=None,
                 **kwargs):
        super().__init__()
        self.N = N
        self.n_channels = n_channels
        self.kernel_size = kernel_size, 
        self.padding = padding
        self.bias = bias
        self.g_act = gspaces.rot2dOnR2(N=N)
        if in_type is None:
            self.in_type = in_type = nn.FieldType(self.g_act, 
                                                  [self.g_act.trivial_repr])
        else:
            self.in_type = in_type
        self.out_type = out_type = nn.FieldType(self.g_act, 
                                                n_channels * [self.g_act.regular_repr])
        
        conv = nn.R2Conv(in_type=in_type, 
                         out_type=out_type, 
                         kernel_size=kernel_size, 
                         padding=padding, 
                         bias=bias)
        batch_norm = nn.InnerBatchNorm(out_type)
        nonlinear_layer = nonlinearity(out_type, inplace=True)
        
        self.block = nn.SequentialModule(
            conv,
            batch_norm,
            nonlinear_layer
        )
        
    def forward(self, x):
        if type(x) != nn.GeometricTensor:
            x = self.in_type(x)
        return self.block(x)
    

class FullyConnectedBlock(torch.nn.Module):
    
    def __init__(self, 
                 in_dim,
                 out_dim,
                 nonlinearity=torch.nn.ELU,
                 **kwargs):
        super().__init__()
        self.out_dim = out_dim
        self.nonlinearity = nonlinearity
        self.out_type = None
        linear = torch.nn.Linear(in_dim, self.out_dim)
        batch_norm = torch.nn.BatchNorm1d(self.out_dim)
        nonlinear_layer = self.nonlinearity(inplace=True)
        self.block = torch.nn.Sequential(
            linear,
            batch_norm,
            nonlinear_layer
        )

    def forward(self, x):
        return self.block(x)
    
class BatchNorm1D(torch.nn.Module):
    
    def __init__(self,
                 in_dim,
                 **kwargs):
        super().__init__()
        self.in_dim = in_dim
        self.out_type = torch.Tensor
        self.batch_norm = torch.nn.BatchNorm1d(in_dim)
        
    def forward(self, x):
        return self.batch_norm(x)
    
    
class Linear(torch.nn.Module):
    
    def __init__(self, 
                 in_dim,
                 out_dim, 
                 nonlinearity=torch.nn.ELU,
                 **kwargs):
        super().__init__()
        self.out_dim = out_dim
        self.out_type = None
        self.linear = torch.nn.Linear(in_dim, self.out_dim)

    def forward(self, x):
        return self.linear(x)
    
    
class GTtoT(torch.nn.Module):
    
    def __init__(self, **kwargs):
        super().__init__()
        self.out_type = torch.Tensor
    
    def forward(self, x):
        if type(x) == nn.GeometricTensor:
            x = x.tensor
        return x
    
        
class Ravel(torch.nn.Module):
    
    def __init__(self, **kwargs):
        super().__init__()
        self.out_type = torch.Tensor
    
    def forward(self, x):
        return x.reshape(x.shape[0], -1)
    