import os
import copy
from time import time

import hydra
import pyrootutils
import torch

from hydra.utils import instantiate
from omegaconf import DictConfig
from torch import nn
from torch.utils.data import DataLoader

# TODO: "one - error" is missing
from torchmetrics.functional.classification import multilabel_accuracy, multilabel_auroc, multilabel_average_precision, multilabel_recall, multilabel_precision, multilabel_f1_score

from torchmetrics.functional.classification import multilabel_coverage_error, multilabel_hamming_distance, multilabel_ranking_loss, multilabel_ranking_average_precision, multilabel_exact_match 

from tqdm import tqdm
import pickle as pkl

from siaug.utils.extras import sanitize_dataloader_kwargs, set_seed

######### Helper functions ###########s

def inference(model, dataloader, device):
    """Run inference on a model and a dataloader."""
    
    # set to evaluation mode
    model.eval()

    outputsValues = []
    targetValues = []

    # run inference
    print(f"=> Starting model inference")
    start = time()
    with torch.no_grad():
        for i, batch in tqdm(enumerate(dataloader)):
            images, targets = batch["img"], batch["lbl"]

            images = images.to(device)
            targets = targets.to(device)

            output = model(images)

            outputsValues.append(output.cpu())
            targetValues.append(targets.cpu())

    outputsValues = torch.cat(outputsValues)
    targetValues = torch.cat(targetValues)

    print('Outputs: ', outputsValues.shape)
    print('Target: ', targetValues.shape)

    print(f"=> Finished model inference [time={time() - start:.2f}s]")

    return outputsValues, targetValues

def evaluateMetrics(outputsValues, targetValues, checkpointPath, dataDir, modelBB, num_labels = 8, tt = 0.5, addValues = True):
    """Run evaluation metrics on the outputs and targets."""

    # run evaluation metrics
    print(f"=> Starting evaluation metrics")
    start = time()

    # assign the trained and evaluated datasets
    if 'chex' in checkpointPath:
        trainedOn = 'chexpert'
    elif 'mimic' in checkpointPath:
        trainedOn = 'mimic'
    else:
        trainedOn = 'unknown'
    
    if 'chex' in dataDir:
        evalOn = 'chexpert'
    elif 'mimic' in dataDir:
        evalOn = 'mimic'
    else:
        evalOn = 'unknown'

    evaluations = {'trainedOn': trainedOn, 'evaluatedOn': evalOn, 'checkpoint': checkpointPath, 'dataPath': dataDir, 'modelBackbone': modelBB, 'num_labels': num_labels, 'threshold': tt}
    values = {'trainedOn': trainedOn, 'evaluatedOn': evalOn, 'checkpoint': checkpointPath, 'dataPath': dataDir, 'modelBackbone': modelBB, 'num_labels': num_labels, 'threshold': tt}
    if addValues:
        values['outputs'] = outputsValues
        values['targets'] = targetValues

    # metrics: auroc
    evaluations['aurocs'] = multilabel_auroc(preds = outputsValues, target = targetValues, num_labels=num_labels, average='none').numpy()
    evaluations['macro_auroc'] = multilabel_auroc(preds = outputsValues, target = targetValues, num_labels=num_labels, average='macro').numpy()
    evaluations['micro_auroc'] = multilabel_auroc(preds = outputsValues, target = targetValues, num_labels=num_labels, average='micro').numpy()
    evaluations['weighted_auroc'] = multilabel_auroc(preds = outputsValues, target = targetValues, num_labels=num_labels, average='weighted').numpy()

    # metrics: accuracy
    evaluations['accuracies'] = multilabel_accuracy(preds = outputsValues, target = targetValues, num_labels=num_labels, average='none', threshold= tt).numpy()
    evaluations['macro_acc'] = multilabel_accuracy(preds = outputsValues, target = targetValues, num_labels=num_labels, average='macro', threshold= tt).numpy()
    evaluations['micro_acc'] = multilabel_accuracy(preds = outputsValues, target = targetValues, num_labels=num_labels, average='micro', threshold= tt).numpy()
    evaluations['weighted_acc'] = multilabel_accuracy(preds = outputsValues, target = targetValues, num_labels=num_labels, average='weighted', threshold= tt).numpy()

    # metrics: average precision
    evaluations['av_prec'] = multilabel_average_precision(preds = outputsValues, target = targetValues, num_labels=num_labels, average='none').numpy()
    evaluations['macro_av_prec'] = multilabel_average_precision(preds = outputsValues, target = targetValues, num_labels=num_labels, average='macro').numpy()
    evaluations['micro_av_prec'] = multilabel_average_precision(preds = outputsValues, target = targetValues, num_labels=num_labels, average='micro').numpy()
    evaluations['weighted_av_prec'] = multilabel_average_precision(preds = outputsValues, target = targetValues, num_labels=num_labels, average='weighted').numpy()

    # metrics: recall
    evaluations['recalls'] = multilabel_recall(preds = outputsValues, target = targetValues, num_labels=num_labels, average='none', threshold= tt).numpy()
    evaluations['macro_recall'] = multilabel_recall(preds = outputsValues, target = targetValues, num_labels=num_labels, average='macro', threshold= tt).numpy()
    evaluations['micro_recall'] = multilabel_recall(preds = outputsValues, target = targetValues, num_labels=num_labels, average='micro', threshold= tt).numpy()
    evaluations['weighted_recall'] = multilabel_recall(preds = outputsValues, target = targetValues, num_labels=num_labels, average='weighted', threshold= tt).numpy()

    # metrics: precision
    evaluations['precisions'] = multilabel_precision(preds = outputsValues, target = targetValues, num_labels=num_labels, average='none', threshold= tt).numpy()
    evaluations['macro_precision'] = multilabel_precision(preds = outputsValues, target = targetValues, num_labels=num_labels, average='macro', threshold= tt).numpy()
    evaluations['micro_precision'] = multilabel_precision(preds = outputsValues, target = targetValues, num_labels=num_labels, average='micro', threshold= tt).numpy()
    evaluations['weighted_precision'] = multilabel_precision(preds = outputsValues, target = targetValues, num_labels=num_labels, average='weighted', threshold= tt).numpy()

    # metrics: f1 score
    evaluations['f1_scores'] = multilabel_f1_score(preds = outputsValues, target = targetValues, num_labels=num_labels, average='none', threshold= tt).numpy()
    evaluations['macro_f1_score'] = multilabel_f1_score(preds = outputsValues, target = targetValues, num_labels=num_labels, average='macro', threshold= tt).numpy()
    evaluations['micro_f1_score'] = multilabel_f1_score(preds = outputsValues, target = targetValues, num_labels=num_labels, average='micro', threshold= tt).numpy()
    evaluations['weighted_f1_score'] = multilabel_f1_score(preds = outputsValues, target = targetValues, num_labels=num_labels, average='weighted', threshold= tt).numpy()

    # metrics: hamming loss
    evaluations['hamming_loss'] = multilabel_hamming_distance(preds = outputsValues, target = targetValues, num_labels=num_labels, average='none', threshold= tt).numpy()
    evaluations['macro_hamming_loss'] = multilabel_hamming_distance(preds = outputsValues, target = targetValues, num_labels=num_labels, average='macro', threshold= tt).numpy()
    evaluations['micro_hamming_loss'] = multilabel_hamming_distance(preds = outputsValues, target = targetValues, num_labels=num_labels, average='micro', threshold= tt).numpy()
    evaluations['weighted_hamming_loss'] = multilabel_hamming_distance(preds = outputsValues, target = targetValues, num_labels=num_labels, average='weighted', threshold= tt).numpy()

    # metrics: coverage error, ranking loss, ranking average precision, exact match
    evaluations['coverage_error'] = multilabel_coverage_error(preds = outputsValues, target = targetValues, num_labels=num_labels).numpy()
    evaluations['ranking_loss'] = multilabel_ranking_loss(preds = outputsValues, target = targetValues, num_labels=num_labels).numpy()
    evaluations['ranking_av_prec'] = multilabel_ranking_average_precision(preds = outputsValues, target = targetValues, num_labels=num_labels).numpy()
    evaluations['exact_match'] = multilabel_exact_match(preds = outputsValues, target = targetValues, num_labels=num_labels, threshold= tt).numpy()
   
    print(f"=> Finished evaluation metrics [time={time() - start:.2f}s]")

    return evaluations, values

######################################

# set the project root
root = pyrootutils.setup_root(__file__, dotenv=True, pythonpath=True)
config_dir = os.path.join(root, "configs")


@hydra.main(version_base="1.2", config_path=config_dir, config_name="train.yaml")
def evaluate(cfg: DictConfig):
    """Run inference and evaluate a pretrained representation / fine-tuned classifier based on a Hydra configuration."""

    cfg_copy = copy.deepcopy(cfg)
    backbone = cfg_copy["model"]["backbone"]
    datapath = cfg_copy["dataloader"]["valid"]["dataset"]["data_dir"]
    num_labels = cfg_copy["model"]["num_classes"]

    print(f"=> Starting [experiment={cfg['task_name']}]")
    print("=> Initializing Hydra configuration")
    cfg = instantiate(cfg)


    seed = cfg.get("seed", None)
    if seed is not None:
        set_seed(seed)

    # set up device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(f"=> Instantiating valid dataloader [device={device}]")
    valid_dataloader = DataLoader(**sanitize_dataloader_kwargs(cfg["dataloader"]["valid"]))

    # create the model
    print(f"=> Creating model [device={device}]")
    model = cfg["model"].to(device)
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

    assert cfg["save_path"] is not None, "Must specify a save path for the outputs" #(given from command line)
    savePath = cfg["save_path"]

    assert cfg["checkpoint_folder"] is not None, "Must specify a checkpoint folder" #(given from command line)

    # load checkpoint
    if cfg["checkpoint_folder"] is not None:
        print(cfg["checkpoint_folder"])
        checkpoints = os.listdir(cfg["checkpoint_folder"])
        for checkpoint in tqdm(checkpoints):
            print(checkpoint)
            checkpoint_path = os.path.join(cfg["checkpoint_folder"], checkpoint)
            print("  Loading checkpoint from " + checkpoint_path)
            state_dict = torch.load(checkpoint_path)
            model.load_state_dict(state_dict)

            outputsValues, targetValues = inference(model, valid_dataloader, device)
            evaluations, values = evaluateMetrics(outputsValues, targetValues, checkpointPath = checkpoint_path, dataDir= datapath, modelBB= backbone, num_labels = num_labels, tt = 0.5, addValues = True)

            print('Writing outputs to pickle files...')

            # pickle dump the outputs
            with open(os.path.join(savePath, checkpoint + '_modelBB_' + backbone + '_dataset_' + datapath.split('/')[-1] + '_evaluations.pkl'), 'wb') as f:
                pkl.dump(evaluations, f)
            with open(os.path.join(savePath, checkpoint + '_modelBB_' + backbone + '_dataset_' + datapath.split('/')[-1] + '_values.pkl'), 'wb') as f:
                pkl.dump(values, f)


    elif cfg["resume_from_ckpt"] is not None:
        print('Loading default checkpoint specified in config')
        checkpoint_path = cfg["resume_from_ckpt"]
        print("  Loading checkpoint from " + checkpoint_path)
        state_dict = torch.load(checkpoint_path)
        model.load_state_dict(state_dict)

        outputsValues, targetValues = inference(model, valid_dataloader, device)
        evaluations, values = evaluateMetrics(outputsValues, targetValues, checkpointPath = checkpoint_path, dataDir= datapath, modelBB= backbone, num_labels = num_labels, tt = 0.5, addValues = True)

        print('Writing outputs to pickle files...')

        # pickle dump the outputs
        with open(os.path.join(savePath, checkpoint_path + '_modelBB_' + backbone + '_dataset_' + datapath.split('/')[-1] + '_evaluations.pkl'), 'wb') as f:
            pkl.dump(evaluations, f)
        with open(os.path.join(savePath, checkpoint_path + '_modelBB_' + backbone + '_dataset_' + datapath.split('/')[-1] + '_values.pkl'), 'wb') as f:
            pkl.dump(values, f)

    else:
        print("No checkpoint found")

    print(f"=> Finished [experiment={cfg['task_name']}]")

if __name__ == "__main__":
    evaluate()
