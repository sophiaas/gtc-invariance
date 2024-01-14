from torch_tools.config import Config
from torch.optim import Adam



"""
OPTIMIZER
"""
optimizer_config = Config({"type": Adam, 
                           "params": {
                               "lr": 0.001
                           }
                          }
                         )