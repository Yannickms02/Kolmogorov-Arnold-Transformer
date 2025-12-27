# Reproduce Results of Thesis
In order to validate the presented results by reproducing them, the following steps are recommended. These are the exact steps taken in this work. The experiments were conducted on a system with dual RTX 5090 (CUDA 12.8), AMD Ryzen 9 9950X processor and 192 GB DDR5 memory running Ubuntu Server 24.04 LTS. Prior experimentation also included an identical system with an AMD RX 7900 XT GPU and ROCm 7.1 (all presented results however stem from the dual RTX 5090 system).

## Preparation
### Python 3.14
The system came pre-installed with python 3.13.7 and was upgraded to the current stable-release via:

```bash
# PPA
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update

# Python 3.14 installation
sudo apt install python3.14 python3.14-venv python3.14-dev

# verify
python3.14 --version
```

In case you don't have any version of python installed, refer to https://python.org for the current stable release suited for your operating system (at the time of writing, version 3.14).

### Python Virtual Environment
Create a virtual environment to establish a clean framework for running the experiments and avoiding global package conflicts.

```bash
# Virtual Environment
python3.14 -m venv ~/kat-env

# Activate
source ~/kat-env/bin/activate
```
This environment can now be used to install the dependencies for this project as stated in the ``requirements.txt``
### PyTorch 2.10.0 (Nightly)
The current release of PyTorch can be obtained via the PyTorch [Get Started](https://pytorch.org/get-started/locally/) page. For best performance (and due to prior experimentation with PyTorch ROCm 7 which required the nightly 2.10-build), PyTorch version 2.10 was chosen, which can be installed with (for CUDA 12.8, for other configs, please refer to the Get Started page):

```bash
pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128
```

### Clone Repository
Clone this repo to the desired location.

```bash
git clone https://github.com/Yannickms02/Kolmogorov-Arnold-Transformer.git
```

### Dependencies
The KAN-variants are implemented using the [efficient-kan](https://github.com/Blealtan/efficient-kan)-Repository. To use this repo in python, it was installed via pip using:
```bash
pip install git+https://github.com/Blealtan/efficient-kan.git
```

Other dependencies include Weights & Biases (Wandb) for experiment logging and Huggingface Datesets. The full ``requirements.txt`` stems from the wandb-files and encapsulates the exact environment used for training. The dependencies of this file can be installed with:

```bash
pip install -r requirements.txt
```

### Weights & Biases
In order to log the experiments and all meta-data, wandb was used. The training logs and access to the training runs (i. e. via the analysis jupyter notebook) cannot be provided, as they contain senstitive information. To fully reproduce the results, you would need to create your own account on the [Weights & Biases](https://wandb.ai/site/experiment-tracking/) Site (free of charge).

After account creation, you can log in via terminal with:
```bash
wandb login
```
and provide the credentials (API-Key) from your account.


## Project Structure
The resulting project structure should resemble the following to proceed:

```
Kolmogorov-Arnold-Transformer/
|-- models/             # Model Configurations
|   |-- layers/             # Swappable Feed-Forward-Networks
|   |   |-- ffn_mlp.py          # MLP baseline with GELU activation
|   |   |-- ffn_kan.py          # KAN B-Spline using efficient-kan
|   |   |-- ffn_grkan.py        # Group-rational KAN as in KAT (Yang & Wang (2024))
|   |   |-- ffn_mlp_bspline.py  # MLP with learnable B-Spline activations
|   |-- config.py           # Model config (sizes, training hyperparameters, etc.)
|   |-- transformer.py      # Custom Transformer with swappable FFN-Block
|-- experiments/        # Experiments
|   |-- prepare_datasets.py # Prepare and Tokenize datasets (AG News, FineWeb, (WikiText-2))
|   |-- train.py            # Trainer
|-- results/            # Results
|   |-- final/              # Results of thesis (to separate from own results)
|-- checkpoints/        # Trained Best-Model-Checkpoints
|-- data/               # Processed Datasets
|-- runner.py           # Manual Experiment-Runner
|-- run_experiments.py  # Auto-Benchmarking and Dataset-Preparation
|-- requirements.txt
|-- README.md
|-- reproduce.md
|-- analysis.ipynb      # Analysis of experiments via wandb-api (requires account)
|-- model_analysis.ipynb # Analysis of models regarding interpretability
```
The results presented in the thesis are provided within the "final" directory to seperate and retain them while (re-)running the experiments. The checkpoints-directory contains the saved best-model-checkpoints of the "Tiny" model size, due to GitHub's file size constraint "Small/Base" were excluded.

## Run Experiments (Auto)
The experiments can be run with the script ``run_experiments.py``. This script detects the hardware for PyTorch and sets the parameters accordingly. Multi-GPU setups use PyTorch _DataDistributedParallel_ for distributed computation (check batch-size and adjust accordingly). The script automatically fetches the datasets if not already present.

Usage:
```bash
python run_experiments.py
```

The script then loads and tokenizes the datasets. FineWeb is loaded with a limit of 30M-Tokens (as used in the thesis), if you wish to adjust this you need to run the experiments manually (or at the least, load the dataset manually).

The runs follow the configurations of ``models/config.py``. First, all architectures are run, then all seeds, then all sizes (active sizes can be adjusted in ``run_experiments.py``) and lastly all tasks (classification and language_modelling).

## Run Experiments (Manual)
The experiments can also be run manually via the ``runner.py`` script (which the auto-version uses as well).

### Prepare Datasets
The datasets need to be loaded manually as well in this case.

#### AG News
Prepare "AG News" (classification) with this command:

```bash
python experiments/prepare_datasets.py --task ag_news
```

#### FineWeb
Prepare the "FineWeb" (language modelling) dataset with this command (to use the same 30M-Tokens limit):

```bash
python experiments/prepare_datasets.py --task fineweb --tokens 30000000
```

The token count can be adjusted as desired (the FineWeb-Sample has a "full" size of 10B-Tokens). Other (optional) parameters include ``seq_len`` (default: 256) and ``seed``(for random sampling of dataset; default 42).

#### (WikiText-103)
WikiText-103 (103M-Tokens) was used as well during experimentation and is therefore an additional option. The size, however, exceeded the project scope and remains for future work to be evaluated.
This dataset can be prepared via:

```bash
python experiments/prepare_datasets.py --task wikitext
```

### Run Experiments
To run the experiments, you can now resolve to the automatic runner (if desired) or run the experiments as needed manually.

```bash
python runner.py --task <classification|language_modelling> --arch <mlp|kan_bspline|kan_mean|kan_grkan|mlp_bspline> --size <tiny|small|base>
```
Other options are:
- ``--seq_len <int>:`` Sequence Length for language modelling (default: 256),
- ``--seed <int>:`` Seed for experiment (default: 42, thesis used 42, 1337, 2024),
- ``--lm_dataset <wikitext|fineweb>:`` If you wish to run wikitext, you can specify it with this parameter.
- ``--compile:`` Compile the models before running them (can introduce overhead to small models but significantly accelerated epoch times during experimentation with RTX cards; ROCm had issues with running at all with this flag)
- ``--batch_size <int>:`` Override batch-size from config (default: None),
- ``--mini_batch_size <int>:`` Override mini-batch-size from config (default: None)

## Note
- The chosen hyperparameters follow best practices and have been validated with prior experiments (albeit short runs). They are, however, most likely not "perfect" and better accuracy or performance could be achieved by tweaking these. To establish even grounds for drawing conclusions the parameters were chosen to be identical for all architectures. The Learning Rate was set to a conservative level to account for potential instability during training (which was observed for the GR-KAN (NaN errors)).
- If your system differs from the experiment system (mainly gpu), you are most likely required to adjust the config. Batch-Size and Mini-Batch-Size stem from a dual gpu setup, i. e. the Mini-Batch refers to the batch-size served to each gpu (in this work 2). If you have more or less than 2 gpus, you might need to experiment to find the the best settings for these. This also applies in the case of VRAM-constraints (i. e. OOM errors). Choosing the largest possible batch-size generally leads to more stable gradients (which are preferred) and better utilizes modern gpus. Memory (VRAM) constraints may, however, require you to (in spite of using gradient accumulation) lower the (effective) batch-size as well.
- If you have a multi-gpu system, you may want to use _DistributedDataParallel_ for better utilization. In this case substitute the script execution from ``python runner.py --task ...`` to ``torchrun --nproc_per_node <number of gpus> runner.py --task ...``. This only applies to manual execution, the automatic script (``run_experiments.py``) does this by default.
- For analysis of the runs, make sure to adjust the wandb-settings in the jupyter notebook accordingly (if desired). The original wandb-entity-settings have been removed for privacy reasons. 