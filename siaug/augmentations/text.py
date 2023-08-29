from typing import Dict, List, Union

import numpy as np
from torch import Tensor

from siaug.utils.text import get_config_tokenizer_model_classes

__all__ = ["Tokenizer", "SanitizeText", "RandomSentenceSampler"]


class Tokenizer:
    """Generic tokenization class for HuggingFace models."""

    def __init__(
        self, model_name: str, enforce_model_max_length: int = None, keep_dims: bool = True
    ):
        self.keep_dims = keep_dims

        (config_class, tokenizer_class, _) = get_config_tokenizer_model_classes(model_name)
        config = config_class.from_pretrained(model_name, trust_remote_code=True)

        model_max_length = None
        if enforce_model_max_length is not None:
            model_max_length = enforce_model_max_length
        elif hasattr(config, "max_position_embeddings"):
            model_max_length = config.max_position_embeddings

        self.tokenizer = tokenizer_class.from_pretrained(
            model_name,
            model_max_length=model_max_length,
            trust_remote_code=True,
        )

    def __call__(
        self,
        txt: Union[str, List[str], List[List[str]]] = None,
    ) -> Dict[str, Tensor]:
        output = self.tokenizer(txt, padding="max_length", truncation=True, return_tensors="pt")
        if self.keep_dims:
            # TODO: check if there's a better way to remove the batch dimension, if needed
            return {k: v.squeeze() for (k, v) in output.items()}

        return output


class SanitizeText:
    def __init__(self, sep: str = ". "):
        self.sep = sep

    def __call__(self, txt: str) -> str:
        sentences = txt.replace("\n", "").split(self.sep)
        return self.sep.join([s.strip() for s in sentences if len(s.strip()) > 0])


class RandomSentenceSampler:
    """Split a text into sentences and return a random one.

    Args:
        empty_txt (str, optional): Token to return if no sentences are found. Defaults to "[PAD]".
        min_length (int, optional): Minimum length of a sentence to be considered. Defaults to 1.
    """

    def __init__(self, sep: str = ". ", empty_txt: str = "[PAD]", min_words: int = 1):
        self.sep, self.empty_txt, self.min_words = sep, empty_txt, min_words

    def __call__(self, txt: str) -> str:
        sentences = [s for s in txt.split(self.sep) if len(s.split(" ")) >= self.min_words]
        if len(sentences) == 0:
            return self.empty_txt

        return sentences[np.random.randint(0, len(sentences))]
