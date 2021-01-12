from typing import List, Tuple

import torch
import torch.nn as nn
from fastai.layers import Embedding
from fastai.torch_core import Module

from .autoencoder import SymmetricalAutoEncoder

__all__ = [
    "EmbeddingLayer",
    "AutoEmbedder",
]


class EmbeddingLayer(Module):
    """
    Embedding layer.
    Automatically splits input Tensors based on embedding sizes;
    then, embeds each feature separately and concatenates the output
    back into a single outpuut Tensor.
    """

    def __init__(self, emb_szs: List[Tuple[int, int]]):
        self.embeddings = nn.ModuleList(
            [Embedding(in_sz, out_sz) for in_sz, out_sz in emb_szs]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = [emb(x[..., i]) for i, emb in enumerate(self.embeddings)]
        x = torch.cat(x, dim=-1)
        return x


class AutoEmbedder(Module):
    """
    Module used to learn unsupervised embeddings for categorical variables.
    Combines neural embedding layers and an autoencoder for unsupervised training.

    During the training phase, the autoencoder is used to tune embeddings in an
    unsupervised fashion; in the evaluation phase, the autoencoder is dropped and
    the output of the embedding layer is returned.
    """

    def __init__(
        self,
        in_sz: int,
        out_sz: int,
        emb_szs: List[Tuple[int, int]],
        layer_szs: List[int],
    ):
        self.embeddings = EmbeddingLayer(emb_szs)
        self.autoencoder = SymmetricalAutoEncoder([in_sz] + layer_szs)
        self.out = nn.Linear(in_sz, out_sz)
        self.last_target = None

    def forward(self, x_cat: torch.Tensor, x_cont: torch.Tensor) -> torch.Tensor:
        # Pass each categorical feature into its corresponding embedding module
        if x_cat.size(0) > 0:
            x = self.embeddings(x_cat)
            # x = self.emb_drop(x)
        # Concatenate embedding outputs and continuous variables
        if self.training and x_cont.size(0) > 0:
            x = torch.cat([x, x_cont], -1) if x_cat.size(0) > 0 else x_cont
        self.last_target = x.clone().detach()
        # If in evaluation mode, simply return the embedding output
        if not self.training:
            return x
        # Otherwise, if in training, save the embedding output for loss computation
        # and pass the embedding output to the autoencoder
        x = self.autoencoder(x)
        # x = self.out(x)
        return x
