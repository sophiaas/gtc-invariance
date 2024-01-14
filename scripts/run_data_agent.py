import wandb
import argparse
from gtc.utils import create_dataset_run


parser = argparse.ArgumentParser()
parser.add_argument(
    "--config",
    type=str,
    help="Name of .py config file with no extension."
)
parser.add_argument("--entity", type=str, help="wandb entity for sweep", default="naturalcomputation")
parser.add_argument("--project", type=str, help="wandb project for sweep", default="tcgpool-datasets")
parser.add_argument("-sweep_id", type=str, help="id for sweep")


args = parser.parse_args()

exec("from configs.experiments.{} import dataset_config".format(args.config))

def run_wrapper():
    create_dataset_run(
        dataset_config=dataset_config, data_project=args.project, entity=args.entity
    )


if args.sweep_id is not None:
    wandb.agent(
        args.sweep_id, function=run_wrapper, entity=args.entity, project=args.project
    )
else:
    run_wrapper()
