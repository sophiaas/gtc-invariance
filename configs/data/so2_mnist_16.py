from transform_datasets.patterns.natural import MNIST
from transform_datasets.transforms import SO2, CenterMean, UnitStd, AddChannelDim, Resize, CircleCrop
from torch_tools.config import Config


"""
DATASET
"""

pattern_config = Config(
    {
        "type": MNIST,
        "params": {"path": "datasets/mnist/mnist_train.csv"},
    }
)


transforms_config = {
    "0": Config(
        {
            "type": SO2,
            "params": {
                "sample_method": "random"
            },
        }
    ),
    "1": Config(
        {
            "type": Resize,
            "params": 
            {
                "new_size": (16, 16)
            }
        }
    ),
    "2": Config(
        {
            "type": CircleCrop,
            "params": 
            {}
        }
    ),
    "3": Config(
        {
            "type": AddChannelDim,
            "params": {}
        }
    )
}


dataset_config = {"pattern": pattern_config, 
                  "transforms": transforms_config,
                  "seed": 0}