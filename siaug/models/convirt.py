from typing import Dict, Tuple, Union

import timm
from torch import Tensor, nn

from siaug.models.components import Projector, ProjectorConfig, TextEncoder


class ConVIRT(nn.Module):
    """Implementation of ConVIRT: Zhang, Y. Contrastive Learning of Medical Visual Representations
    from Paired Images and Text. https://arxiv.org/pdf/2010.00747.pdf.

    Args:
        img_backbone (str): Name of the image backbone, forwarded to timm
        txt_backbone (str): HuggingFace repository of the text backbone
        num_channels (int): Number of channels in the input image
        output_dim (int): Number of features in the output of the projector
        embedding_method (str, optional): Embedding method used to propagate to the projector
    """

    def __init__(
        self,
        img_backbone: str,
        txt_backbone: str,
        num_channels: int,
        output_dim: int = 512,
        embedding_method: str = "last_hidden_state_cls",
        freeze_n_bert_layers: int = 6,
        **kwargs,
    ):
        super().__init__()

        # image encoder
        kwargs = {
            **kwargs,
            "model_name": img_backbone,
            "in_chans": num_channels,
            "num_classes": 0,
        }
        self.img_encoder = timm.create_model(**kwargs)

        # image projector
        self.img_projector = Projector(
            ProjectorConfig(
                input_dim=self.img_encoder.num_features,
                output_dim=output_dim,
                num_linear_layers=2,
                batch_norm=False,
                bias=True,
            )
        )

        # text encoder
        # NB: the default for ConVIRT is "emilyalsentzer/Bio_ClinicalBERT"
        self.txt_encoder = TextEncoder(txt_backbone, embedding_method)

        # ConVIRT typically freezes the fist 6 layers of the text encoder
        for i in range(freeze_n_bert_layers):
            for param in self.txt_encoder.model.encoder.layer[i].parameters():
                param.requires_grad = False

        # text projector
        self.txt_projector = Projector(
            ProjectorConfig(
                input_dim=self.txt_encoder.config.hidden_size,
                output_dim=output_dim,
                num_linear_layers=2,
                batch_norm=False,
                bias=True,
            )
        )

    def forward(
        self,
        inputs: Dict[str, Union[Tensor, Dict[str, Tensor]]],
    ) -> Tuple[Tensor, ...]:
        # compute features for each view
        img, txt = inputs["img"], inputs["txt"]

        z1 = self.img_projector(self.img_encoder(img))
        z2 = self.txt_projector(self.txt_encoder(txt))

        return {"z1": z1, "z2": z2}
