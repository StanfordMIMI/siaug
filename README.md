# Siamese Augmentation Strategies (SiAug)

[**Paper**](https://openreview.net/pdf?id=xkmhsBITaCw) | [**OpenReview**](https://openreview.net/forum?id=xkmhsBITaCw) | [**ArXiv**](https://arxiv.org/abs/2301.12636)


This repository contains the implementation code for our paper: <br>
[Exploring Image Augmentations for Siamese Representation Learning with Chest X-Rays](https://openreview.net/pdf?id=xkmhsBITaCw)

- Authors: Rogier van der Sluijs*, Nandita Bhaskhar*, Daniel Rubin, Curtis Langlotz, Akshay Chaudhari
- *- co-first authors
- Published at Medical Imaging with Deep Learning (MIDL) 


## tl;dr

Tailored augmentation strategies for image-only Siamese representation learning can outperform supervised baselines with zero-shot learning, linear probing and fine-tuning for chest X-ray classification. We systematically assess the effect of various augmentations on the quality and robustness of the learned representations. We train and evaluate Siamese Networks for abnormality detection on chest X-Rays across three large datasets (MIMIC-CXR, CheXpert and VinDr-CXR). We investigate the efficacy of the learned representations through experiments involving linear probing, fine-tuning, zero-shot transfer, and data efficiency. Finally, we identify a set of augmentations that yield robust representations that generalize well to both out-of-distribution data and diseases, while outperforming supervised baselines using just zero-shot transfer and linear probes by up to 20%.


## Updates
- __[04/29/2023]__ The code is currently being cleaned up for release. Please stay tuned for updates. If you'd like to access our code sooner, reach out to us via [email](#contact)


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


## Contact Us
<a name="contact"></a> 
This repository is being developed at the Stanford's MIMI Lab. Please reach out to `sluijs [at] stanford [dot] edu` and `nanbhas [at] stanford [dot] edu` if you would like to use or contribute to `siaug`. 

## Citation
If you find our paper and/or code useful, please use the following BibTex for citation:
```bib
@article{sluijsnanbhas2023_siaug,
  title={Exploring Image Augmentations for Siamese Representation Learning with Chest X-Rays}, 
  author={Rogier van der Sluijs and Nandita Bhaskhar and Daniel Rubin and Curtis Langlotz and Akshay Chaudhari},
  year={2023},
  journal={Medical Imaging with Deep Learning (MIDL)},
}
```


