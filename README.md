# The $G$-Triple Correlation Layer for Robust $G$-Invariance in $G$-Equivariant Networks

This repository is the official accompaniment to _A General Framework for Robust G-Invariance in G-Equivariant Networks_ (2023) by Sophia Sanborn and Nina Miolane, published in the _Proceedings of the 27th Conference on Neural Information Processing Systems (NeurIPS)._

## Installation

To install the requirements and package, run:

```
pip install -r requirements.txt
python install -e .
```

## Datasets

To download the datasets:

1. Download the zip file [here](https://drive.google.com/file/d/1zXDnPNlzo5uTfYo97RlKIDHstaWVQD3L/view?usp=sharing).
2. Place the file in the top node of this directory, i.e. in `gtc-invariance/`.
3. Run:
    ```
    unzip datasets.zip
    rm -r datasets.zip
    ```

## Training

The full set of hyperparameters and training configurations are specified in the config files in the ```configs/``` folder. To train a model on a particular experiment, you will call the following:

```
scripts/run_data_agent.py --config [name of config]
scripts/run_train_agent.py --config [name of config]
```

The first call will generate the transformed dataset, and the second will train the model on that dataset. The `config` argument should be followed by the name of a particular config file from `configs/experiments`, e.g. `o2mnist_d16_maxpool`. The `.py` extension of the config should be excluded. Each of the configs in the `configs/experiments` folder combines various model, trainer, etc configs also specified in the `configs` folder. The scripts are set up to log the model with [Weights & Biases](https://wandb.ai/). A user's wandb entity and project directories should be specified in `configs/logger`.

## License

This repository is licensed under the MIT License.  
