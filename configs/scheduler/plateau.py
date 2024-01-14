from torch_tools.config import Config
from torch.optim.lr_scheduler import ReduceLROnPlateau


"""
SCHEDULER
"""
scheduler_config = Config({"type": ReduceLROnPlateau, "params": {"factor": 0.5, "patience": 2, "min_lr": 1e-4}})
