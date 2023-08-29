from typing import Dict

import torch
from torch import nn

from siaug.utils.text import get_config_tokenizer_model_classes


class TextEncoder(nn.Module):
    def __init__(self, model_name: str, embedding_method: str = "last_hidden_state_cls"):
        super().__init__()

        self.model_name = model_name
        self.embedding_method = embedding_method

        config_class, _, model_class = get_config_tokenizer_model_classes(model_name)

        self.config = config_class.from_pretrained(model_name, trust_remote_code=True)

        if self.embedding_method[:-1] == "hidden_state_numbered_from_the_end_":
            assert self.embedding_method[-1].isnumeric()
            self.config.output_hidden_states = True

        self.model = model_class.from_pretrained(
            model_name, config=self.config, trust_remote_code=True
        )

        # TODO: remove this hack, and find a better solution
        if not self.embedding_method == "pooler_output":
            for param in self.model.pooler.parameters():
                param.requires_grad = False

    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        if self.embedding_method == "last_hidden_state_mean":
            # TODO does not take into account the masking
            return self.model(**inputs).last_hidden_state[:, 1:-1, :].mean(axis=1)

        elif self.embedding_method == "last_hidden_state_cls":
            return self.model(**inputs).last_hidden_state[:, 0, :]

        elif self.embedding_method == "last_hidden_state":
            return self.model(**inputs).last_hidden_state[:, :, :]

        elif self.embedding_method[:-1] == "hidden_state_numbered_from_the_end_":
            return self.model(**inputs).hidden_states[-1 * int(self.embedding_method[-1])]

        elif self.embedding_method == "pooler_output":
            return self.model(**inputs).pooler_output

        elif self.embedding_method == "raw_output":
            return self.model(**inputs)

        elif self.embedding_method == "get_projected_text_embeddings":
            return self.model.get_projected_text_embeddings(**inputs)

        elif self.embedding_method == "get_text_features":
            return self.model.get_text_features(**inputs)

        else:
            raise ValueError("Embedding type not handled.")
