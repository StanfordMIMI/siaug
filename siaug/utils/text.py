from typing import Tuple

from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    CLIPTextConfig,
    CLIPTextModel,
    CLIPTokenizer,
)

__all__ = ["get_config_tokenizer_model_classes"]


def get_config_tokenizer_model_classes(model_name: str) -> Tuple:
    """Retrieve class constructors for a HuggingFace model."""

    config_class = AutoConfig
    tokenizer_class = AutoTokenizer
    model_class = AutoModel

    if model_name == "openai/clip-vit-large-patch14":
        config_class = CLIPTextConfig
        tokenizer_class = CLIPTokenizer
        model_class = CLIPTextModel

    return config_class, tokenizer_class, model_class
