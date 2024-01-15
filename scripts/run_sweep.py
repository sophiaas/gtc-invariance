import argparse
import wandb
import subprocess as sp
import copy


parser = argparse.ArgumentParser()

# Sweep Args
parser.add_argument(
    "--sweep_config",
    type=str,
    help="Name of .py sweep config file with no extension.",
)
parser.add_argument(
    "--n_agents", type=int, help="number of parallel agents to run", default=3
)
parser.add_argument(
    "--sweep_id", type=str, help="(Optional) sweep id to continue existing sweep"
)

# Agent Args
parser.add_argument(
    "--config",
    type=str,
    help="Name of .py config file with no extension."
)
parser.add_argument(
    "--devices", nargs="+", help="list of devices to run on", default=[0, 1, 2]
)
parser.add_argument(
    "--n_examples", type=int, help="number of data examples", default=int(1e6)
)
parser.add_argument(
    "--no_data_gen", action='store_true', help="don't sweep through dataset generation"
)

args = parser.parse_args()

exec("from configs.experiments.{} import master_config, logger_config".format(args.config))
exec("from configs.sweeps.{} import sweep_config".format(args.sweep_config))

dataset_in_sweep = sum(["dataset" in x for x in sweep_config["parameters"].keys()]) > 0

if dataset_in_sweep:
    # If sweeping through dataset parameters, then do a dataset sweep
    data_sweep_config = copy.deepcopy(sweep_config)
    data_sweep_config["parameters"] = {}
    for k, v in sweep_config["parameters"].items():
        if k.startswith("dataset"):
            new_k = k[8:]
            data_sweep_config["parameters"][new_k] = v
    if len(data_sweep_config) > 0:    
        data_sweep_id = wandb.sweep(
            data_sweep_config,
            project=logger_config["params"]["data_project"],
            entity=logger_config["params"]["entity"],
        )
        commands = []
        for i in range(args.n_agents):
            command = "python scripts/run_data_agent.py -sweep_id {} --config {} --project {} --entity {}".format(
                data_sweep_id,
                args.config,
                logger_config["params"]["data_project"],
                logger_config["params"]["entity"],
            )
            commands.append(command)

        processes = [sp.Popen(command, shell=True) for command in commands]
        for p in processes:
            p.wait()


# Sweep through training
if args.sweep_id is None:
    args.sweep_id = wandb.sweep(
        sweep_config,
        project=logger_config["params"]["project"],
        entity=logger_config["params"]["entity"],
    )


commands = []
for i in range(args.n_agents):
    idx = i % len(args.devices)
    device = args.devices[idx]
    command = "python scripts/run_train_agent.py -sweep_id {} --device {} --config {} --n_examples {} --project {} --entity {}".format(
        args.sweep_id,
        device,
        args.config,
        args.n_examples,
        logger_config["params"]["project"],
        logger_config["params"]["entity"],
    )
    print(command)
    commands.append(command) 

processes = [sp.Popen(command, shell=True) for command in commands]
for p in processes:
    p.wait()
