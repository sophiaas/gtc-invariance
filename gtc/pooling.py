from escnn.gspaces import *
from escnn.nn.modules.equivariant_module import EquivariantModule
from escnn.nn import FieldType
from escnn.nn import GeometricTensor
from escnn.nn.modules.invariantmaps import GroupPooling

from escnn.nn.modules.equivariant_module import EquivariantModule
from escnn.nn.modules.utils import indexes_from_labels

import torch
from torch import nn

from typing import List, Tuple, Any
from collections import defaultdict
import numpy as np

from gtc.functional import get_cayley_table


class TCGroupPoolingEfficient(GroupPooling):
    
    def __init__(self, in_type, group, idx=None, **kwargs):
        super().__init__(in_type, **kwargs)
        self.idx = idx
        self.group = group()
        self.cayley_table = get_cayley_table(self.group)

    def triple_correlation(self, x):
        b, k, d = x.shape
        x = x.reshape((b * k, d))
        nexts = x[:, self.cayley_table]
        mult = x.unsqueeze(1) * x[:, self.cayley_table.swapaxes(0, 1)]
        TC = torch.bmm(mult, nexts)
        TC = TC.reshape((b, k, d, d))
        return TC
    
    def forward(self, input: GeometricTensor) -> GeometricTensor:
        r"""
        
        Apply Group Pooling to the input feature map.
        
        Args:
            input (GeometricTensor): the input feature map

        Returns:
            the resulting feature map
            
        """
        assert input.type == self.in_type

        coords = input.coords
        input = input.tensor
        b, c = input.shape[:2]
        spatial_shape = input.shape[2:]
        
        for s, contiguous in self._contiguous.items():
            
            in_indices = getattr(self, "in_indices_{}".format(s))
            out_indices = getattr(self, "out_indices_{}".format(s))
            
            if contiguous:
                fm = input[:, in_indices[0]:in_indices[1], ...]
            else:
                fm = input[:, in_indices, ...]
                
            # split the channel dimension in 2 dimensions, separating fields
            fm = fm.view(b, -1, s, *spatial_shape)
            
            output = self.triple_correlation(fm.squeeze())
                
            if self.idx is None:
                idx = torch.triu_indices(output.shape[2], output.shape[3])
            else:
                idx = self.idx

            output = output[:, :, idx[0], idx[1]]
            a, b, c = output.shape
            output = output.reshape((a * b, c))
            output = output / (output.norm(dim=0, keepdim=True) + 1e-5)
            output = output.reshape((a, b, c, 1, 1))
        return output

    def export(self):
        raise NotImplementedError
        
        


class TCGroupPooling(GroupPooling):
    
    def __init__(self, in_type, group_type="cyclic", idx=None, **kwargs):
        """
        group_type should be "cyclic" or "dihedral"
        """
        super().__init__(in_type, **kwargs)
        self.idx = idx
        self.group_type = group_type

    def triple_correlation_vectorized_batch_cyclic(self, x):
        b, k, d = x.shape
        x = x.reshape((b * k, d))
        all_rolls = torch.zeros((b * k, d, d)).to(x.device)
        for i in range(d):
            all_rolls[:, :, i] = torch.roll(x, -i, dims=-1)
        rolls_mult = x.unsqueeze(1) * all_rolls
        TC = torch.bmm(rolls_mult, all_rolls)
        TC = TC.reshape((b, k, d, d))
        return TC
    
    def triple_correlation_vectorized_batch_dihedral(self, x):
        b, k, d = x.shape
        n = d // 2
        x = x.reshape((b * k, d))
        all_rolls = torch.zeros((b * k, d, d)).to(x.device)
        for i in range(d):
            roll0 = torch.roll(x[:, :n], -i, dims=-1)
            roll1 = torch.roll(x[:, n:], -i, dims=-1)
            all_rolls[:, :, i] = torch.hstack([roll0, roll1])
        rolls_mult = x.unsqueeze(1) * all_rolls
        TC = torch.bmm(rolls_mult, all_rolls)
        TC = TC.reshape((b, k, d, d))
        return TC
    
    def triple_correlation_vectorized_batch_r2(self, x):
        b, k, h, w = x.shape
        x = x.reshape((b * k, h * w))
        all_rolls = torch.zeros((b * k, h * w, h * w)).to(x.device)
        for i in range(h * w):
            all_rolls[:, :, i] = torch.roll(x, -i, dims=-1)
        rolls_mult = x.unsqueeze(1) * all_rolls
        TC = torch.bmm(rolls_mult, all_rolls)
        TC = TC.reshape((b, k, h * w, h * w))
        return TC
    
    def forward(self, input: GeometricTensor) -> GeometricTensor:
        r"""
        
        Apply Group Pooling to the input feature map.
        
        Args:
            input (GeometricTensor): the input feature map

        Returns:
            the resulting feature map
            
        """
        
        assert input.type == self.in_type

        coords = input.coords
        input = input.tensor
        b, c = input.shape[:2]
        spatial_shape = input.shape[2:]
        
        for s, contiguous in self._contiguous.items():
            
            in_indices = getattr(self, "in_indices_{}".format(s))
            out_indices = getattr(self, "out_indices_{}".format(s))
            
            if contiguous:
                fm = input[:, in_indices[0]:in_indices[1], ...]
            else:
                fm = input[:, in_indices, ...]
                
            # split the channel dimension in 2 dimensions, separating fields
            fm = fm.view(b, -1, s, *spatial_shape)
            
            if self.group_type == "cyclic":
                output = self.triple_correlation_vectorized_batch_cyclic(fm.squeeze())
                
            elif self.group_type == "dihedral": 
                output = self.triple_correlation_vectorized_batch_dihedral(fm.squeeze())
                
            if self.idx is None:
                idx = torch.triu_indices(output.shape[2], output.shape[3])
            else:
                idx = self.idx

            output = output[:, :, idx[0], idx[1]]
            a, b, c = output.shape
            output = output.reshape((a * b, c))
            output = output / (output.norm(dim=0, keepdim=True) + 1e-5)
            output = output.reshape((a, b, c, 1, 1))
        return output

    def export(self):
        raise NotImplementedError


class TCGroupPoolingR2Spatial(torch.nn.Module):

    def __init__(self, idx=None, **kwargs):
        super().__init__(**kwargs)
        self.idx = idx
        
    def triple_correlation_vectorized_batch(self, x):
        b, k, n, n = x.shape
        d = n * n
        x = x.reshape((b * k, d))
        all_rolls = torch.zeros((b * k, d, d)).to(x.device)
        for i in range(d):
            all_rolls[:, :, i] = torch.roll(x, -i, dims=-1)
        rolls_mult = x.unsqueeze(1) * all_rolls
        TC = torch.bmm(rolls_mult, all_rolls)
        TC = TC.reshape((b, k, d, d))
        return TC
    
    def forward(self, x):
                
        output = self.triple_correlation_vectorized_batch(x.squeeze())
        return output

        if self.idx is None:
            idx = torch.triu_indices(output.shape[2], output.shape[3])
        else:
            idx = self.idx

        output = output[:, :, idx[0], idx[1]]
        a, b, c = output.shape
        output = output.reshape((a * b, c))
        output = output / (output.norm(dim=0, keepdim=True) + 1e-5)
        output = output.reshape((a, b, c, 1, 1))

        return output

    def export(self):
        raise NotImplementedError
