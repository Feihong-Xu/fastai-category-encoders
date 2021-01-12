from typing import Callable, List, Optional

import numpy as np
import torch
import torch.nn as nn

from ..functional import consecutive_pairs

__all__ = [
    "AutoEncoder",
    "SymmetricalAutoEncoder",
    "linear_relu",
]


class AutoEncoder(nn.Module):
    """
    Base autoencoder class.
    Holds reference to encoder / decoder / bottleneck (optional) members
    and defines a basic `forward` method.
    """

    def __init__(self, enc: nn.Module, dec: nn.Module, btn: Optional[nn.Module] = None):
        super().__init__()
        self.enc = enc
        self.btn = btn
        self.dec = dec

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.enc(x)
        if self.btn is not None:
            x = self.btn(x)
        x = self.dec(x)
        return x


def linear_relu(in_sz: int, out_sz: int) -> List[nn.Module]:
    """Linear + ReLU utility."""
    linear = nn.Linear(in_sz, out_sz)
    relu = nn.ReLU()
    return [linear, relu]


class SymmetricalAutoEncoder(AutoEncoder):
    """
    AutoEncoder for linear data.
    Calls `block_fn` with each size in `layer_sizes` for both
    encoder and decoder blocks.

    Parameters
    ----------
    layer_sizes: List[int]
        The list of encoder layer sizes.
    block_fn: Callable[[int, int], List[nn.Module]]
        The block function to be used. Accepts in/out sizes and returns a list of nn.Modules.
    """

    def __init__(
        self,
        layer_sizes: List[int],
        block_fn: Callable[[int, int], List[nn.Module]] = linear_relu,
    ) -> None:
        encoder_blocks = np.array(
            [
                block_fn(in_sz, out_sz)
                for in_sz, out_sz in consecutive_pairs(layer_sizes)
            ]
        )
        decoder_blocks = np.array(
            [
                block_fn(in_sz, out_sz)
                for in_sz, out_sz in consecutive_pairs(reversed(layer_sizes))
            ]
        )
        enc = nn.Sequential(*(encoder_blocks.flatten()))
        dec = nn.Sequential(*(decoder_blocks.flatten()[:-1]))
        super().__init__(enc, dec)
