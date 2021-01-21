from typing import Callable, Generator, Iterable, Optional, Tuple, TypeVar

T = TypeVar("T")


__all__ = [
    "consecutive_pairs",
    "find",
]


def consecutive_pairs(iterable: Iterable[T]) -> Generator[Tuple[T, T], None, None]:
    """
    Returns a generator of pairs of consecutive elements in `iterable`.
    """
    prev = None
    for item in iterable:
        if prev is not None:
            yield prev, item
        prev = item


def find(
    func: Callable[[T], bool], iterable: Iterable[T], last: bool = False
) -> Optional[T]:
    """
    Returns the first (or last, if so specified) element in `iterable` for which `func` returns `True`.

    Parameters
    ----------
    func : Callable[[T], bool]
        The lookup function.
    iterable : Iterable[T]
        The iterable to use for lookup.
    last : bool, default = False
        Whether to return the last matching element instead of the first.

    Returns
    -------
        Optional[T]: the first element in `iterable` that matches `func`, or None if missing.
    """
    it = reversed(iterable) if last else iterable
    for item in it:
        if func(item):
            return item
    return None
