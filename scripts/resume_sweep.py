import argparse
import wandb
import subprocess as sp
from torch_tools.utils import load_checkpoint
from transform_datasets.utils.wandb import load_dataset, load_or_create_dataset
from torch_tools.config import Config


parser = argparse.ArgumentParser()

parser.add_argument(
    "--n_agents", type=int, help="number of parallel agents to run", default=3
)
parser.add_argument(
    "--sweep_id", type=str, help="sweep id."
)
parser.add_argument(
    "--entity", default="naturalcomputation", type=str, help="wandb entity name."
)
parser.add_argument(
    "--project", default="bispectrum", type=str, help="wandb project name."
)
parser.add_argument(
    "--devices", nargs="+", help="list of devices to run on", default=[0, 1, 2]
)
parser.add_argument(
    "--n_examples", type=int, help="number of data examples to resume up to.", default=int(1e8)
)

parser.add_argument(
    "--dataset_config", type=str, help="Optional dataset config if you want to resume on a different dataset.", default=None
)

args = parser.parse_args()


api = wandb.Api()
sweep = api.sweep("{}/{}/{}".format(args.entity, args.project, args.sweep_id))

for run_obj in sweep.runs:
    wandb_run = wandb.init(project=args.project, entity=args.entity, id=run_obj.id, resume="must")
    epoch = wandb.summary.epoch
    loaded = False
    while not loaded:
        if epoch < 0:
            raise Exception("No saved checkpoints.")
        try:
            checkpoint_path = wandb.restore('checkpoints/checkpoint_{}.pt'.format(epoch)).name
            trainer = load_checkpoint(checkpoint_path)
            loaded = True
        except:
            epoch -= 1

    data_loader = Config(trainer.logger.config['data_loader']).build()
    if args.dataset_config is not None:
        exec("from configs.dataset.{} import dataset_config".format(args.dataset_config))
    else:
        dataset_config = trainer.logger.config['dataset']
        
    dataset = load_or_create_dataset(dataset_config, project=trainer.logger.data_project, entity=args.entity)
    data_loader.load(dataset)

    epochs = int(args.n_examples // len(data_loader.train.dataset.data))

    if trainer.epoch >= epochs:
        print("Model has already been trained for {} examples. Skipping to next run.".format(args.n_examples, trainer.n_examples))
    else:
        remaining_epochs = epochs - trainer.epoch

        trainer.resume(data_loader=data_loader,
                       epochs=remaining_epochs)
