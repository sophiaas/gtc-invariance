from torch_tools.config import Config
from torch_tools.logger import WBLogger


"""
LOGGING
"""
logger_config = Config(
    {
        "type": WBLogger,
        "params": {
            "project": "YOUR PROJECT",
            "data_project": "YOUR DATA PROJECT",
            "entity": "YOUR ENTITY",
            "log_interval": 1,
            "watch_interval": 1,
            "plot_interval": 1,
            "end_plotter": None,
            "step_plotter": None,
        },
    }
)
