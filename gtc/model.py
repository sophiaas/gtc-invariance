import torch
from collections import OrderedDict


def create_model(block_configs):
    blocks = OrderedDict()
    for i, config in enumerate(blocks):
        if i > 0:
            config.params["in_type"] = self.blocks[-1].out_type
        block = config.build()
        blocks[i] = block
    return torch.nn.Sequential(blocks)
