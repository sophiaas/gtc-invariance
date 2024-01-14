from transform_datasets.patterns.natural import ModelNet10Voxel
from transform_datasets.transforms import OctahedralRotation, AddChannelDim
from torch_tools.config import Config


"""
DATASET
"""

pattern_config = Config(
    {
        "type": ModelNet10Voxel,
        "params": {"path": "datasets/ModelNet10Voxel10x10x10/",
                   "grid_size": (10, 10, 10)
                  },
    }
)


transforms_config = {
    "0": Config(
        {
            "type": OctahedralRotation,
            "params": {
                "full": True,
                "sample_method": "random"
            },
        }
    ),
    "1": Config(
        {
            "type": AddChannelDim,
            "params": {}
        }
    )
}


dataset_config = {"pattern": pattern_config, 
                  "transforms": transforms_config,
                  "seed": 0}