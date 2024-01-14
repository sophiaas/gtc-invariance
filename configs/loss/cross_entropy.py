import torch
from torch_tools.config import Config

loss_config = Config(
    {
        "type": torch.nn.CrossEntropyLoss,
        "params": {}
    }
)