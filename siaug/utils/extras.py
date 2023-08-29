import os
import random
from shutil import copyfile
from typing import Dict, List, Union

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from omegaconf import OmegaConf
from tqdm.auto import tqdm

__all__ = ["sanitize_dataloader_kwargs"]


def sanitize_dataloader_kwargs(kwargs):
    """Converts num_workers argument to an int
    NB: this is needed if num_workers is gather from the OS environment.
    """

    if "num_workers" in kwargs:
        kwargs["num_workers"] = int(kwargs["num_workers"])

    return kwargs


def set_seed(seed: int):
    """Seed the RNGs."""

    print(f"=> Setting seed [seed={seed}]")
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.deterministic = True

    print("=> Setting a seed slows down training considerably!")


def move_checkpoints(
    experiments: List[str],
    experiment_type: str,
    dataset: str,
    input_dir: Union[os.PathLike, str] = "/fsx/home-sluijs/siaug/logs/",
    output_dir: Union[os.PathLike, str] = "/fsx/aimi/siaug/checkpoints/",
):
    checkpoints = []
    for experiment in experiments:
        experiment_dir = os.path.join(input_dir, experiment)
        if os.path.isdir(experiment_dir):
            # runs
            run_dir = os.path.join(experiment_dir, "runs")
            if os.path.exists(run_dir):
                timestamps = sorted(os.listdir(run_dir))
                if len(timestamps) > 0:
                    timestamp = timestamps[-1]
                    timestamp_dir = os.path.join(run_dir, timestamp)
                    checkpoint_path = os.path.join(
                        timestamp_dir, "checkpoints", "last.pt", "pytorch_model.bin"
                    )
                    checkpoints.append((checkpoint_path, experiment))

            # multiruns
            multirun_dir = os.path.join(experiment_dir, "multiruns")
            if os.path.exists(multirun_dir):
                timestamps = sorted(os.listdir(multirun_dir))
                if len(timestamps) > 0:
                    timestamp = timestamps[-1]
                    timestamp_dir = os.path.join(multirun_dir, timestamp)
                    numbers = sorted(os.listdir(timestamp_dir))
                    if len(numbers) > 0:
                        number = numbers[-1]
                        number_dir = os.path.join(timestamp_dir, number)
                        checkpoint_path = os.path.join(
                            number_dir, "checkpoints", "last.pt", "pytorch_model.bin"
                        )
                        checkpoints.append((checkpoint_path, experiment))

    # move checkpoints to output directory
    for (checkpoint_path, experiment) in tqdm(checkpoints):
        if os.path.exists(checkpoint_path):
            output_path = os.path.join(
                output_dir, experiment_type, dataset, f"{experiment}" + "-last.pt"
            )
            copyfile(checkpoint_path, output_path)

def move_checkpoints_from_parent_folder(
    # experiments: List[str],
    # experiment_type: str,
    # dataset: str,
    input_dir: Union[os.PathLike, str] = "/fsx/aimi/nanbhas/logs/",
    output_dir: Union[os.PathLike, str] = "/fsx/aimi/siaug/checkpoints/",
    condition: str = None,
):
    experiments = os.listdir(input_dir)

    if condition is not None:
        experiments = [experiment for experiment in experiments if condition in experiment]

    checkpoints = []
    for experiment in experiments:
        experiment_dir = os.path.join(input_dir, experiment)
        if os.path.isdir(experiment_dir):
            # runs
            run_dir = os.path.join(experiment_dir, "runs")
            if os.path.exists(run_dir):
                timestamps = sorted(os.listdir(run_dir))
                if len(timestamps) > 0:
                    timestamp = timestamps[-1]
                    timestamp_dir = os.path.join(run_dir, timestamp)
                    checkpoint_path = os.path.join(
                        timestamp_dir, "checkpoints", "last.pt", "pytorch_model.bin"
                    )
                    if os.path.exists(checkpoint_path):
                            output_path = os.path.join(
                                output_dir, f"{experiment}" + "-last.pt"
                            )
                            checkpoints.append((checkpoint_path, output_path, experiment))

            # multiruns
            multirun_dir = os.path.join(experiment_dir, "multiruns")
            if os.path.exists(multirun_dir):
                timestamps = sorted(os.listdir(multirun_dir))
                if len(timestamps) > 0:
                    timestamp = timestamps[-1]
                    timestamp_dir = os.path.join(multirun_dir, timestamp)
                    numbers = sorted(os.listdir(timestamp_dir))
                    if len(numbers) > 0:
                        number = numbers[-1]
                        number_dir = os.path.join(timestamp_dir, number)
                        checkpoint_path = os.path.join(
                            number_dir, "checkpoints", "last.pt", "pytorch_model.bin"
                        )
                        if os.path.exists(checkpoint_path):
                            output_path = os.path.join(
                                output_dir, f"{experiment}" + "-last.pt"
                            )
                            checkpoints.append((checkpoint_path, output_path, experiment))

    # ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # import pprint
    # pprint.pprint(checkpoints)

    # move checkpoints to output directory
    for (checkpoint_path, output_path, experiment) in tqdm(checkpoints):
        copyfile(checkpoint_path, output_path)


def create_lcls_cfgs(
    experiments: List[str],
    dataset: str,
    input_dir: str = "/fsx/aimi/siaug/checkpoints/repr/",
    output_dir: str = "../configs/experiment",
    overrides: Dict = {},
):
    exp_prefix = "# @package _global_\n\n"
    task_names = []
    for experiment in experiments:
        # remove repr_ prefix and -last.pt postfix
        task_name = f"lcls_{experiment[:-8]}".split("repr_")[-1]
        task_names.append(task_name)

        cfg = OmegaConf.create(
            {
                "defaults": [f"lcls_{dataset}_base.yaml"],
                "task_name": task_name,
                "model": {
                    "backbone": "resnet50",
                    "ckpt_path": os.path.join(input_dir, dataset, experiment),
                },
                **overrides,
            }
        )

        yaml_cfg = OmegaConf.to_yaml(cfg, sort_keys=False)
        with open(os.path.join(output_dir, task_name + ".yaml"), "w") as f:
            f.write(exp_prefix + yaml_cfg)

    return task_names

def create_lcls_cfgs_from_checkpoint_folder(
    dataset: str,
    input_dir: str = "/fsx/aimi/siaug/checkpoints/repr/mimic",
    output_dir: str = "../configs/experiment",
    overrides: Dict = {},
):
    train_dataset = input_dir.split("/")[-1]
    experiments = os.listdir(input_dir)
    exp_prefix = "# @package _global_\n\n"
    task_names = []
    for experiment in experiments:
        # remove -last.pt postfix
        task_name = "lcls_" + dataset + '_' + train_dataset + '_' + experiment.split("-last.pt")[0]
        task_names.append(task_name)

        cfg = OmegaConf.create(
            {
                "defaults": [f"lcls_{dataset}_base.yaml"],
                "task_name": task_name,
                "model": {
                    "backbone": "resnet50",
                    "ckpt_path": os.path.join(input_dir, experiment),
                },
                **overrides,
            }
        )

        yaml_cfg = OmegaConf.to_yaml(cfg, sort_keys=False)

        saveFile = os.path.join(output_dir, task_name + ".yaml")
        print(saveFile)

        with open(saveFile, "w") as f:
            f.write(exp_prefix + yaml_cfg)

    return task_names

# read and compile evaluations from pickle files
def compile_evaluations(folder_path):
    import pandas as pd
    import pickle as pkl

    list_of_files = []
    for ff in os.listdir(folder_path):
        if ff.endswith('_evaluations.pkl'):
            # read pickle file
            with open(os.path.join(folder_path, ff), 'rb') as f:
                x = pkl.load(f)
            list_of_files.append(x)

    dd = pd.DataFrame.from_records(list_of_files)
    dd.to_csv(os.path.join(folder_path, "compiled_evaluations.csv"), index=False)

# concatenate compiled csv files
def concatenate_compiled_evaluations_csvs(list_of_folder_paths: List[str], file_name: str = "compiled_evaluations.csv", output_folder: str = "/fsx/aimi/nanbhas/evalDir/joined_eval_csvs", output_file_name: str = "joined_eval.csv"):

    import pandas as pd

    list_of_files = [os.path.join(folder, file_name) for folder in list_of_folder_paths]
    list_of_dfs = []
    for ff in list_of_files:
        if os.path.exists(ff):
            print(ff)
            df = pd.read_csv(ff)
            list_of_dfs.append(df)
    newDF = pd.concat(list_of_dfs, ignore_index=True)

    newDF.to_csv(os.path.join(output_folder, output_file_name), index=False)

    print('Compiled csvs in {} and saved to {}'.format(" , ".join(list_of_folder_paths), os.path.join(output_folder, output_file_name)))



# find full path for a file
def findFullPath(name, path):
    for root, dirs, files in os.walk(path):
        if name in files:
            return os.path.join(root, name)