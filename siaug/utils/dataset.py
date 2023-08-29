import os
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import polars as pl
import torch
import voxel as vx
import zarr
from torch import Tensor
from torch.utils.data import Dataset
from tqdm.auto import tqdm
from siaug.utils.extras import findFullPath
from skmultilearn.model_selection import iterative_train_test_split


__all__ = ["CHEXPERT_PATHOLOGIES", "load_zmv", "load_txt", "compute_normalization_constants"]


CHEXPERT_PATHOLOGIES = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Enlarged Cardiomediastinum",
    "Fracture",
    "Lung Lesion",
    "Lung Opacity",
    "No Finding",
    "Effusion",
    "Pleural Other",
    "Pneumonia",
    "Pneumothorax",
    "Support Devices",
]

VINDR_PATHOLOGIES = [
    "Aortic enlargement",
    "Atelectasis",
    "Calcification",
    "Cardiomegaly",
    "Clavicle fracture",
    "Consolidation",
    "Edema", 
    "Emphysema", 
    "Enlarged PA", 
    "ILD",
    "Infiltration",
    "Lung Opacity", 
    "Lung cavity",
    "Lung cyst", 
    "Mediastinal shift",
    "Nodule/Mass", 
    "Pleural effusion",
    "Pleural thickening", 
    "Pneumothorax",
    "Pulmonary fibrosis",
    "Rib fracture",
    "Other lesion", 
    "COPD",
    "Lung tumor",
    "Pneumonia",
    "Tuberculosis",
    "Other diseases",
    "No finding"
]

VINDR_SUBSET_PATHOLOGIES = [
    "Cardiomegaly",
    "Pulmonary fibrosis",
    "Pleural thickening",
    "Pleural effusion",
    "Lung Opacity",
    "Tuberculosis",
    "Pneumonia",
    "Nodule/Mass",
    "No finding"
]

MIMIC_SUBSET_PATHOLOGIES = [
    "Atelectasis",
    "Cardiomegaly",
    "Edema",
    "Fracture",
    "Effusion",
    "Pneumonia",
    "Pneumothorax",
    "No Finding",
    ]

def load_zmv(path: os.PathLike):
    """Load a Zarr-based MedicalVolume from a ZipStore."""
    store = zarr.ZipStore(path)
    return vx.MedicalVolume.from_zarr(store, headers_attr="headers", affine=np.eye(4))


def load_txt(path: os.PathLike) -> str:
    """Load a text file."""
    if os.path.exists(path):
        with open(path) as f:
            return f.read()
    return None


def compute_normalization_constants(ds: Dataset, limit: int = None) -> Tuple[float, float]:
    """Compute the normalization constants for a dataset."""
    arr = []
    length = len(ds) if not limit else limit
    for idx in tqdm(range(length), disable=not ds.verbose):
        arr.append(np.ravel(ds[idx][0]))

    arr = np.concatenate(arr)
    return (np.mean(arr), np.std(arr))


def preprocess_chexpert_labels(df: pl.DataFrame, labels: List[str], na_mode: str = None) -> Tensor:
    """Select the labels, apply the na_mode and return a Torch tensor."""
    lbls = torch.from_numpy(df.select(labels).to_numpy())

    if na_mode == "positive":
        lbls = torch.where(
            lbls == -1.0,
            torch.ones_like(lbls),
            torch.where(lbls == 1.0, lbls, torch.zeros_like(lbls)),
        )
    elif na_mode == "negative":
        lbls = (torch.where(lbls == 1.0, lbls, torch.zeros_like(lbls)),)

    return lbls

def preprocess_vindr_filepaths(image_id_list: List[str], root: str, extension: str = ".npy"):
    """Given a list of vindr ids, find their corresponding paths."""

    pathList = []
    for id in tqdm(image_id_list):
        pathList.append(findFullPath(id + extension, root))
    return pathList

def preprocess_vindr_csvs(path: str, image_dir: str, extension: str = ".npy"):
    """
    Given a csv file with vindr labels, create a new csv file after preprocessing to find the corresponding paths for the images.
    Args:
        path: path to the csv file
        image_dir: path to the directory containing the images (default: path to volumes)
        extension: extension of the images (default: .npy)
    """

    new_path = path.split('.csv')[0] + '_ground_truth.csv'
    rawDF = pd.read_csv(path)

    if "train" in str(path):
        print(f"Creating the majority ground truth for train at: {new_path}")
        print("This will take a while ...") 
        newDF = rawDF.groupby('image_id').agg(pd.Series.mode)
        newDF.reset_index(inplace=True)
        newDF = newDF.rename(columns = {'index':'image_id'})
        rawDF = newDF
    
    image_ids = list(rawDF['image_id'])
    img_paths = preprocess_vindr_filepaths(image_id_list = image_ids, root = image_dir, extension = extension)

    rawDF['image_path'] = img_paths

    rawDF.to_csv(new_path, index=False)

    print(f"Done! Saved ground truth csv at: {new_path}")

def preprocess_chexpert_view_slice_csv(path: str, colName: str = "Frontal/Lateral", view: str = "Frontal"):
    """
    Given a csv file with chexpert labels, create a new csv file with a slice of the data containing only the frontal or lateral images.
    Args:
        path: path to the original csv file
        colName: column name containing the view information (default: "Frontal/Lateral")
        view: view to keep (default: "Frontal")
    """

    new_path = path.split('.csv')[0] + '_' + view + '.csv'
    rawDF = pd.read_csv(path)

    newDF = rawDF[rawDF[colName] == view]
    newDF.to_csv(new_path, index=False)

    print(f"Done! Saved ground truth csv at: {new_path}")


def select_vindr_subset_slice(path: str, label_names: List[str] = VINDR_SUBSET_PATHOLOGIES, no_finding_limit: int = 3000, random_state: int = 42):
    """
    Given a csv file with vindr labels, create a new csv file with a slice of the data containing only the rows that have a positive label in label_names.
    Args:
        path: path to the original csv file
        label_names: list of label names to keep (default: VINDR_SUBSET_PATHOLOGIES)
        no_finding_limit: limit on the number of rows with no finding (default: 3000). If 0, no limit is applied.
        random_state: random state for reproducibility (default: 42)
    """

    col_string = ("_" + str(no_finding_limit) + "_subset.csv").replace(" ", "")
    new_path = path.split('.csv')[0] + col_string
    
    rawDF = pd.read_csv(path)

    # separate rows with no finding from rows with at least one positive label
    F = rawDF[rawDF["No finding"] == 0]
    NF = rawDF[rawDF["No finding"] == 1]

    assert no_finding_limit <= len(NF), "no_finding_limit cannot be greater than the number of rows with no finding"

    if no_finding_limit == 0:
        no_finding_limit = len(NF)

    # subselect rows from F which have at least one positive label for the selected pathologies in label_names
    Fsub = F[F[label_names].isin([1]).any(axis=1)]

    # subsample from NF
    NFsub = NF.sample(n=no_finding_limit, random_state=random_state)

    newDF = pd.concat([NFsub, Fsub], axis=0)
    newDF = newDF.sample(frac = 1, random_state=random_state)

    newDF.to_csv(new_path, index=False)

    print(f"Done! Saved subselected ground truth csv at: {new_path}")


def determine_mimic_data_efficiency_splits(csv_path: str, label_names: List[str] = MIMIC_SUBSET_PATHOLOGIES, data_splits_list = [0.1, 0.1], rename_cols_dict: Dict[str, str] = {"Pleural Effusion": "Effusion"}, na_mode: str = "positive"):
    """
    Determine the number of rows to be kept for each data split based on stratified sampling.
    Args:
        csv_path: path to the csv file
        label_names: list of label names in the dataset (default: MIMIC_SUBSET_PATHOLOGIES)
        data_splits_list: list of data split ratios (default: [0.1, 0.1]). Note that the 2nd number is a fraction of the first number. i.e. 0.1, 0.1 means that it will give 10% and 1% splits of the data and the 1% split will be taken from the 10% split.
        rename_cols_dict: dictionary of column names to rename (default: {"Pleural Effusion": "Effusion"})
        na_mode: mode for handling missing values. Options are "positive", "negative", "ignore" (default: "positive")
    """
    # read the csv file as a polars dataframe
    df = pl.read_csv(csv_path, infer_schema_length=1000).rename(rename_cols_dict)

    # get the label set as an numpy array
    y = preprocess_chexpert_labels(df, label_names, na_mode=na_mode)
    y = y.numpy()

    # get list of indices corresponding to rows in the dataset
    X = np.arange(len(df)).reshape(len(df), 1)

    # get the stratified split indices
    X_remaining, y_remaining, X_first, y_first = iterative_train_test_split(X, y, test_size=data_splits_list[0])
    X_temp, y_temp, X_second, y_second = iterative_train_test_split(X_first, y_first, test_size=data_splits_list[1])

    # squeeze the indices
    X_first = X_first.squeeze()
    X_second = X_second.squeeze()

    # reverse the column names
    reverse_rename_cols_dict = {v: k for k, v in rename_cols_dict.items()}

    # df to csv
    df_first = df[X_first].rename(reverse_rename_cols_dict)
    df_second = df[X_second].rename(reverse_rename_cols_dict)

    save_first = csv_path.split('.csv')[0] + '_split_' + str(data_splits_list[0]) + '.csv'
    save_second = save_first.split('.csv')[0] + '_' + str(data_splits_list[1]) + '.csv'

    df_first.write_csv(save_first)
    df_second.write_csv(save_second)

    print(f"Done! Saved csv splits at: {save_first}, {save_second}")


def determine_mimic_data_efficiency_splits_three(csv_path: str, label_names: List[str] = MIMIC_SUBSET_PATHOLOGIES, data_splits_list = [0.5, 0.5, 0.5], rename_cols_dict: Dict[str, str] = {"Pleural Effusion": "Effusion"}, na_mode: str = "positive"):
    """
    Determine the number of rows to be kept for each data split based on stratified sampling.
    Args:
        csv_path: path to the csv file
        label_names: list of label names in the dataset (default: MIMIC_SUBSET_PATHOLOGIES)
        data_splits_list: list of data split ratios (default: [0.1, 0.1]). Note that the 2nd number is a fraction of the first number. i.e. 0.1, 0.1 means that it will give 10% and 1% splits of the data and the 1% split will be taken from the 10% split.
        rename_cols_dict: dictionary of column names to rename (default: {"Pleural Effusion": "Effusion"})
        na_mode: mode for handling missing values. Options are "positive", "negative", "ignore" (default: "positive")
    """
    # read the csv file as a polars dataframe
    df = pl.read_csv(csv_path, infer_schema_length=1000).rename(rename_cols_dict)

    # get the label set as an numpy array
    y = preprocess_chexpert_labels(df, label_names, na_mode=na_mode)
    y = y.numpy()

    # get list of indices corresponding to rows in the dataset
    X = np.arange(len(df)).reshape(len(df), 1)

    # get the stratified split indices
    X_remaining, y_remaining, X_first, y_first = iterative_train_test_split(X, y, test_size=data_splits_list[0])
    X_temp, y_temp, X_second, y_second = iterative_train_test_split(X_first, y_first, test_size=data_splits_list[1])
    X_temp, y_temp, X_third, y_third = iterative_train_test_split(X_second, y_second, test_size=data_splits_list[2])

    # squeeze the indices
    X_first = X_first.squeeze()
    X_second = X_second.squeeze()
    X_third = X_third.squeeze()

    # reverse the column names
    reverse_rename_cols_dict = {v: k for k, v in rename_cols_dict.items()}

    # df to csv
    df_first = df[X_first].rename(reverse_rename_cols_dict)
    df_second = df[X_second].rename(reverse_rename_cols_dict)
    df_third = df[X_third].rename(reverse_rename_cols_dict)

    print(len(df_first), len(df_second), len(df_third))

    save_first = csv_path.split('.csv')[0] + '_split_' + str(data_splits_list[0]) + '.csv'
    save_second = save_first.split('.csv')[0] + '_' + str(data_splits_list[1]) + '.csv'
    save_third = save_second.split('.csv')[0] + '_' + str(data_splits_list[1]) + '.csv'

    df_first.write_csv(save_first)
    df_second.write_csv(save_second)
    df_third.write_csv(save_third)

    print(f"Done! Saved csv splits at: {save_first}, {save_second}, {save_third}")



