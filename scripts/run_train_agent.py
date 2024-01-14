import wandb
import argparse
from gtc.utils import run_trainer


parser = argparse.ArgumentParser()
parser.add_argument(
    "--config",
    type=str,
    help="Name of .py config file with no extension."
)
parser.add_argument("--device", type=int, help="device to run on", default=0)
parser.add_argument(
    "--n_examples", type=int, help="number of data examples", default=100000000
)
parser.add_argument("--entity", type=str, help="wandb entity for sweep", default="naturalcomputation")
parser.add_argument("--project", type=str, help="wandb project for sweep", default="tcgpool")
parser.add_argument("-sweep_id", type=str, help="id for sweep")


args = parser.parse_args()

exec("from configs.experiments.{} import master_config, logger_config".format(args.config))


logger_config.params["entity"] = args.entity
logger_config.params["project"] = args.project

def run_wrapper():
    run_trainer(
        master_config=master_config,
        logger_config=logger_config,
        device=args.device,
        n_examples=args.n_examples,
        entity=args.entity,
        project=args.project
    )

if args.sweep_id is not None:
    wandb.agent(
        args.sweep_id, function=run_wrapper, entity=args.entity, project=args.project
    )
else:
    run_wrapper()
