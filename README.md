# Siamese Augmentation Strategies (SiAug)

This repository contains the implementation code for paper: <br>
[Exploring Image Augmentations for Siamese Representation Learning with Chest X-Rays](https://openreview.net/pdf?id=xkmhsBITaCw)

[**Paper**](https://openreview.net/pdf?id=xkmhsBITaCw) | [**OpenReview**](https://openreview.net/forum?id=xkmhsBITaCw) | [**ArXiv**](https://arxiv.org/abs/2301.12636)

## Installation
To contribute to _siaug_, you can install the package in editable mode:

```python
pip install -e .
pip install -r requirements.txt
pre-commit install
pre-commit
```

Make sure to update the `.env` file according to the setup of your cluster and placement of your project folder on disk. Also, run `accelerate config` to generate a config file, and copy it from `~/cache/huggingface/accelerate/default_config.yaml` to the project directory. Finally, create symlinks from the `data/` folder to the datasets you would want to train on.

## Training
Currently, we support two modes of training: pretraining and linear evaluation. 

### Representation learning
To learn a new representation, you can use the `train_repr.py` script.

```python
# Train and log to WandB
accelerate launch siaug/train_repr.py experiment=experiment_name logger=wandb

# Resume from checkpoint
accelerate launch siaug/train_repr.py ... resume_from_ckpt=/path/to/accelerate/ckpt/dir

# Run a fast_dev_run
accelerate launch siaug/train_repr.py ... fast_dev_run=True max_epoch=10 log_every_n_steps=1 ckpt_every_n_epochs=1
```

### Linear evaluation
To train a linear classifier on top of a frozen backbone, use the `train_lcls.py` script.

```python
# Train a linear classifier on top of a frozen backbone
accelerate launch siaug/train_lcls.py experiment=experiment_name model.ckpt_path=/path/to/model/weights

# Train a linear classifier on top of a random initialized backbone
accelerate launch siaug/train_lcls.py model.ckpt_path=None

# Use ImageNet pretrained weights
accelerate launch siaug/train_lcls.py +model.pretrained=True
```

### Zero Shot Evaluation

To evaluate a model on a downstream task without fine-tuning, use the `siaug/eval.py` script.

```python
python siaug/eval.py experiment=eval_chex_resnet +checkpoint_folder=/path/to/model/checkpoints/folder +save_path=/path/to/save/resulting/pickle/files
```



