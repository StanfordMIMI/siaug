import os
from typing import List, Optional

import timm
import torch

__all__ = ["create_lincls", "create_model_for_inference"]


def create_lincls(
    backbone: str,
    num_classes: int,
    num_channels: int,
    freeze: bool = True,
    ckpt_path: Optional[os.PathLike] = None,
    prefix: Optional[str] = "encoder.",
    model_keys: Optional[List[str]] = None,
    reset_head: bool = True,
    **kwargs,
):
    """Models generated by timm with everything frozen by default except for the fc layer."""

    # create model
    kwargs = {
        **kwargs,
        "model_name": backbone,
        "in_chans": num_channels,
        "num_classes": num_classes,
    }

    model = timm.create_model(**kwargs)
    new_keys = list(model.state_dict().keys())

    if ckpt_path is not None:
        print(f"=> Checkpoint loaded [path={ckpt_path}]")
        state_dict = torch.load(ckpt_path, map_location="cpu")

        # find the state_dict for nested models
        if model_keys is not None:
            for key in model_keys:
                state_dict = state_dict[key]

        if isinstance(prefix, str):
            for k in list(state_dict.keys()):
                if k.startswith(prefix):
                    state_dict[k[len(prefix) :]] = state_dict[k]  # noqa: E203

                del state_dict[k]

        # load state dict
        if reset_head:
            missing_keys, _ = model.load_state_dict(state_dict, strict=False)

            if "fc.weight" in new_keys and "fc.bias" in new_keys:
                assert missing_keys == ["fc.weight", "fc.bias"]

            if "head.weight" in new_keys and "head.bias" in new_keys:
                assert missing_keys == ["head.weight", "head.bias"]
        else:
            model.load_state_dict(state_dict, strict=True)

    else:
        print("=> Warning: no checkpoint was loaded!")

    # freeze everything but the fc layer
    if freeze:
        for name, param in model.named_parameters():
            if name not in ["fc.weight", "fc.bias", "head.weight", "head.bias"]:
                param.requires_grad = False

    # reinitialize the fc layer
    if reset_head:
        if "fc.weight" in new_keys:
            model.fc.weight.data.normal_(mean=0.0, std=0.01)

        if "fc.bias" in new_keys:
            model.fc.bias.data.zero_()

        if "head.weight" in new_keys:
            model.head.weight.data.normal_(mean=0.0, std=0.01)

        if "head.bias" in new_keys:
            model.head.bias.data.zero_()

    return model


def create_model_for_inference(
    backbone: str,
    num_classes: int,
    num_channels: int,
    freeze: bool = True,
    ckpt_path: Optional[os.PathLike] = None,
    prefix: str = "encoder.",
    model_keys: Optional[List[str]] = None,
    **kwargs,
):
    """Models generated by timm with everything frozen by default ready for inference"""

    # create model
    kwargs = {
        **kwargs,
        "model_name": backbone,
        "in_chans": num_channels,
        "num_classes": num_classes,
    }

    model = timm.create_model(**kwargs)

    if ckpt_path is not None:
        print(f"=> Checkpoint loaded [path={ckpt_path}]")
        state_dict = torch.load(ckpt_path, map_location="cpu")

        # find the state_dict for nested models
        if model_keys is not None:
            for key in model_keys:
                state_dict = state_dict[key]

        for k in list(state_dict.keys()):
            if k.startswith(prefix):
                state_dict[k[len(prefix) :]] = state_dict[k]  # noqa: E203

            del state_dict[k]

        # load state dict
        missing_keys, _ = model.load_state_dict(state_dict, strict=False)
        
    else:
        print("=> Warning: no checkpoint was loaded!")

    # freeze everything 
    if freeze:
        for name, param in model.named_parameters():
            param.requires_grad = False

    return model