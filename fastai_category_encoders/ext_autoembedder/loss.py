"""
Loss functions that work with the EmbeddingModel.
"""

from typing import Callable

from fastai.metrics import mse

__all__ = [
    "EmbeddingLoss",
]


class EmbeddingLoss(Callable):
    """
    Custom loss function wrapper for the EmbeddingModel.
    Retrieves output of the embedding layer and
    computes given loss against it.
    """

    def __init__(self, model, loss_fn: Callable = mse):
        self.model = model
        self.loss_fn = loss_fn

    def __call__(self, pred, targ):
        return self.loss_fn(pred, self.model.last_target)
