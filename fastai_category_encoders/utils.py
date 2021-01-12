import pandas as pd

__all__ = ["calc_embedding_size"]


def calc_embedding_size(df: pd.DataFrame) -> int:
    """
    Calculates the appropriate FastText vector size for categoricals in `df`.
    https://developers.googleblog.com/2017/11/introducing-tensorflow-feature-columns.html

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe
    """
    n_categories = sum([len(df[col].unique()) for col in df.columns])
    return int(n_categories ** 0.25)
