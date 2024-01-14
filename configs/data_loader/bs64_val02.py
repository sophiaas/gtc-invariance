from torch_tools.data import TrainValLoader
from torch_tools.config import Config

"""
DATA_LOADER
"""

data_loader_config = Config(
    {
        "type": TrainValLoader,
        "params": {
            "batch_size": 64,
            "fraction_val": 0.2,
            "num_workers": 1,
        },
    }
)

