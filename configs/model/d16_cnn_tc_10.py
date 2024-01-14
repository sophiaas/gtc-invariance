from collections import OrderedDict
from escnn import nn
from escnn import gspaces
from torch_tools.config import Config
from gtc.modules import GonR2ConvBlock, FullyConnectedBlock, GTtoT, Ravel, Linear
from gtc.pooling import TCGroupPooling
from gtc.algebra import compute_non_redundant_tc_indices_dihedral

N = 8


"""
CONV 1
"""

conv1 = Config(
    {
        "type": GonR2ConvBlock,
        "params": {"N": N,
                   "action": gspaces.flipRot2dOnR2,
                   "n_channels": 24,
                   "kernel_size": 16,
                   "nonlinearity": None,
                   "padding": 0,
                   "bias": False 
                  },
    }
)


"""
GROUP POOL
"""

gpool = Config(
    {
        "type": TCGroupPooling,
        "params": {
            "idx": None,
            "group_type": "dihedral"
        },
    }
)


"""
RAVEL
"""

ravel = Config(
    {
        "type": Ravel,
        "params": {},
    }
)


"""
FC1
"""

FC1 = Config(
    {
        "type": FullyConnectedBlock,
        "params": {
            "out_dim": 64
        }
    }
)



"""
FC2
"""

FC2 = Config(
    {
        "type": FullyConnectedBlock,
        "params": {
            "out_dim": 64
        }
    }
)


"""
FC3
"""

FC3 = Config(
    {
        "type": FullyConnectedBlock,
        "params": {
            "out_dim": 64
        }
    }
)


"""
LINEAR
"""
linear = Config(
    {
        "type": Linear,
        "params": {
            "out_dim": 10
        }
    }
)


"""
MODEL CONFIG
"""

model_config = OrderedDict(
    {
    "conv1": conv1,
    "gpool": gpool,
    "ravel": ravel,
    "FC1": FC1,
    "FC2": FC2,
    "FC3": FC3,
    "linear": linear
    }
)
