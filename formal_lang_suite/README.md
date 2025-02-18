## Installation

To set up the environment for running the experiments on the formal language suite.

1. Create a new conda environment:
   ```bash
   conda create -n test
   ```
2. Activate the conda environment:
   ```bash
   conda activate test
   ```
3. Install PyTorch and related packages:
   ```bash
   conda install PyTorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 PyTorch-cuda=11.7 -c pytorch -c Nvidia
   ```
4. Install additional Python packages:
   ```bash
   pip install packaging tqdm transformers matplotlib seaborn
   pip install pandas hydra-core pydantic-settings python-decouple wandb
   ```

Save a `.env` file locally with the following parameters (if Wandb logging is to be enabled):
```
WANDB_API_KEY=__
WANDB_TEAM=__
```

## Running Experiments

```bash
python train_with_ce.py basic.use_wandb=False train.epochs=100 dataset=tomita-1 model.use_reg=False
```
The dataset name can be changed to run different datasets, the model configuration can also be changed to use with or without the regulariser


## Acknowledgements

Code used here has been heavily inspired by the following repository:
* [Recognising Formal Languages - Bhattmishra et al](https://github.com/satwik77/Transformer-Simplicity)