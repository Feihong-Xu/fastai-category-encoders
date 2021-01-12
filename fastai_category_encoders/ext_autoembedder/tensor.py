from typing import Callable

import torch

__all__ = [
    "expanded",
]


def expanded(a: torch.Tensor, b: torch.Tensor, fn: Callable) -> torch.Tensor:
    """
    Makes `a` and `b` shapes compatible, then calls `fn(a, b)`. Uses device of tensor a.

    Parameters
    ----------
    a : torch.Tensor
        The first tensor.
    b : torch.Tensor
        The second tensor.
    fn : Callable
        The function to be applied on expanded `a` and `b`.

    Examples
    --------
    >>> a = torch.randn(5, 2)
    >>> b = torch.randn(12, 2)
    >>> c = expanded(a, b, lambda a, b: a + b)
    >>> c.shape
    torch.Size([5, 12, 2])
    """
    N, M = a.shape[0], b.shape[0]

    # Allocate device space to store results
    res = torch.zeros(N, M).to(device=a.device)

    _a = a.view(N, 1, -1).to(device=a.device)
    _b = b.expand(N, -1, -1).to(device=a.device)

    # Invoke the function over the two tensors
    res = fn(_a, _b)

    # Cleanup device space
    del _a
    del _b
    if a.is_cuda:
        torch.cuda.empty_cache()
    return res
