from transform_datasets.utils.wandb import load_dataset, create_dataset
from collections import OrderedDict
import numpy as np
import torch
import copy
import wandb
from torch_tools.config import Config
from torch_tools.trainer import Trainer
import gtc


def create_model(block_configs, data_loader):
    blocks = OrderedDict()
    i = 0
    for name, config in block_configs.items():
        if i > 0:
            config.params["in_type"] = block.out_type # Previous block
        if config.type == gtc.modules.FullyConnectedBlock or config.type == gtc.modules.Linear or config.type == gtc.modules.BatchNorm1D:
            x, y = next(iter(data_loader.train))
            with torch.no_grad():
                for k, b in blocks.items():
                    x = b(x)
                out_dim = x.shape[-1]
            config.params["in_dim"] = out_dim
        block = config.build()
        blocks[name] = block
        i += 1
    model = torch.nn.Sequential(blocks)
    return model


def load_checkpoint(logdir, device="cpu"):
    from torch_tools.config import Config
    checkpoint = torch.load(logdir, map_location=device)
    trainer = checkpoint["trainer"]
    data_loader = Config(trainer.logger.config['data_loader']).build()
    dataset_config = trainer.logger.config['dataset']
    dataset = load_dataset(dataset_config, project=trainer.logger.data_project, entity=trainer.logger.entity)
    data_loader.load(dataset)
    if not hasattr(checkpoint, "model"):
        for k, v in trainer.logger.config["model"].items():
            trainer.logger.config["model"][k] = Config(v)
        model = create_model(trainer.logger.config["model"], data_loader)
        trainer.model = model
        trainer.model.load_state_dict(checkpoint['model_state_dict'], strict=False)

    optimizer_config = Config(copy.deepcopy(trainer.logger.config["optimizer"]))
    optimizer_config["params"]["params"] = trainer.model.parameters()
    trainer.optimizer = optimizer_config.build()
    trainer.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint, trainer, data_loader


def load_wandb_checkpoint(entity, project, run_id, epoch=None):
    if epoch is None:
        api = wandb.Api()
        run = api.run("{}/{}/{}".format(entity, project, run_id))
        epoch = run.summary.epoch
        loaded = False
        while not loaded:
            if epoch < 0:
                raise Exception("No saved checkpoints.")
            try:
                checkpoint_path = wandb.restore('checkpoints/checkpoint_{}.pt'.format(epoch), run_path="{}/{}/{}".format(entity, project, run_id)).name
                loaded = True
            except:
                epoch -= 1
    else: 
        checkpoint_path = wandb.restore('checkpoints/checkpoint_{}.pt'.format(epoch), run_path="{}/{}/{}".format(entity, project, run_id)).name
    checkpoint, trainer, data_loader = load_checkpoint(checkpoint_path)
    return checkpoint, trainer, data_loader


def nest_dict(dict1):
    result = {}
    for k, v in dict1.items():

        # for each key call method split_rec which
        # will split keys to form recursively
        # nested dictionary
        split_rec(k, v, result)
    return result


def split_rec(k, v, out, sep="."):

    # splitting keys in dict
    # calling_recursively to break items on '_'
    k, *rest = k.split(sep, 1)
    if rest:
        split_rec(rest[0], v, out.setdefault(k, {}))
    else:
        out[k] = v


def flatten_dict(dd, separator=".", prefix=""):
    return (
        {
            prefix + separator + k if prefix else k: v
            for kk, vv in dd.items()
            for k, v in flatten_dict(vv, separator, kk).items()
        }
        if isinstance(dd, dict)
        else {prefix: dd}
    )


def config_to_hash(config):
    if type(config) != dict:
        config = dict(config)
    flat_config = pd.json_normalize(config).to_dict()
    flat_config = sorted(flat_config.items())
    config_hash = hashlib.md5(jsonpickle.encode(flat_config).encode("utf-8")).digest()
    return config_hash.hex()


def nested_set(dic, keys, value):
    for key in keys[:-1]:
        dic = dic.setdefault(key, {})
    dic[keys[-1]] = value

    
def nested_get(dic, keys):
    for key in keys[:-1]:
        dic = dic.setdefault(key, {})
    return dic[keys[-1]]


def fix_wandb_config(wandb_config, master_config):
    new_config = copy.deepcopy(master_config)
    for k, v in wandb_config.items():
        key_list = k.split(".")
        master_val = nested_get(new_config, key_list)
        if v != "random":
            try:
                exec("import {}".format(v.split(".")[0]))
                v = eval(v)
                imported = True
                nested_set(new_config, key_list, v)
            except:
                imported = False
        else:
            imported = False
        if not imported:
            if type(master_val) == type or callable(master_val):
                continue
            else:
                nested_set(new_config, key_list, v)
    return new_config


def run_trainer(master_config, 
                logger_config,
                device=0, 
                n_examples=1e9,
                entity=None,
                project=None):
        
    flat_config = flatten_dict(master_config)

    with wandb.init(config=flat_config, entity=entity, project=project) as run:
        new_config = fix_wandb_config(wandb.config, master_config)
                
        dataset = load_dataset(new_config["dataset"], 
                               logger_config["params"]["data_project"], 
                               logger_config["params"]["entity"])

        data_loader = new_config["data_loader"].build()
        data_loader.load(dataset)

        trainer = construct_trainer(master_config, logger_config, new_config, data_loader)

        epochs = int(n_examples // len(data_loader.train.dataset.data))
        trainer.model.device = device
        trainer.model = trainer.model.to(device)
        trainer.train(data_loader, epochs=epochs)


def create_dataset_run(dataset_config, data_project, entity):
    flat_config = flatten_dict(dataset_config)

    with wandb.init(config=flat_config, project=data_project, entity=entity) as run:
        new_config = fix_wandb_config(wandb.config, dataset_config)
        dataset = create_dataset(new_config, data_project, entity, run)


def construct_trainer(master_config, logger_config, wandb_config, data_loader):
    """
    master_config has the following format:

    master_config = {
        "dataset": dataset_config,
        "model": model_config,
        "optimizer": optimizer_config,
        "loss": loss_config,
        "data_loader": data_loader_config,
    }

    with optional regularizer, normalizer, and learning rate scheduler
    """
    
    torch.manual_seed(wandb_config["seed"])
    np.random.seed(wandb_config["seed"])
        
    #CURRENTLY, SWEEPS ON MODEL HYPERPARAMS WILL NOT WORK
    model = create_model(master_config["model"], data_loader)
    
    loss = wandb_config["loss"].build()
    
    logger_config["params"]["config"] = wandb_config
    logger = logger_config.build()

    optimizer_config = copy.deepcopy(wandb_config["optimizer"])
    optimizer_config["params"]["params"] = model.parameters()
    optimizer = optimizer_config.build()

    if "trainer" not in master_config:
        trainer_type = Trainer
    else:
        trainer_type = master_config["trainer"]
        
    train_config = Config(
        {
            "type": trainer_type,
            "params": {
                "model": model,
                "loss": loss,
                "logger": logger,
                "optimizer": optimizer,
            },
        }
    )

    if "regularizer" in wandb_config:
        regularizer = wandb_config["regularizer"].build()
        train_config["params"]["regularizer"] = regularizer

    if "normalizer" in wandb_config:
        normalizer = wandb_config["normalizer"].build()
        train_config["params"]["normalizer"] = normalizer
        
    if "scheduler" in wandb_config:
        scheduler_config = copy.deepcopy(wandb_config["scheduler"])
        scheduler_config["params"]["optimizer"] = optimizer
        scheduler = scheduler_config.build()
        train_config["params"]["scheduler"] = scheduler
        
    trainer = train_config.build()
    
    return trainer