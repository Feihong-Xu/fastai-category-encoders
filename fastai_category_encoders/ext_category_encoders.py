"""
This module wraps category encoders from the sklearn contrib
library `category_encoders` into a custom subclass.
"""
from typing import List, Optional

import category_encoders as ce
import pandas as pd

from .base import CategoryEncoder

__all__ = [
    "LibraryCategoryEncoder",
    "LIBRARY_ENCODER_STRATEGIES",
    "LIBRARY_ENCODER_STRATEGY_NAMES",
]


# Configuration
LIBRARY_ENCODER_STRATEGIES = {
    "supervised": {
        "cat-boost": ce.CatBoostEncoder,
        "glmm": ce.GLMMEncoder,
        "james-stein": ce.JamesSteinEncoder,
        "leave-one-out": ce.LeaveOneOutEncoder,
        "m-estimator": ce.MEstimateEncoder,
        "target": ce.TargetEncoder,
        "weight-of-evidence": ce.WOEEncoder,
    },
    "unsupervised": {
        "backward-difference": ce.BackwardDifferenceEncoder,
        "base-n": ce.BaseNEncoder,
        "binary": ce.BinaryEncoder,
        "count": ce.CountEncoder,
        "hashing": ce.HashingEncoder,
        "helmert": ce.HelmertEncoder,
        "ohe": ce.OneHotEncoder,
        "ordinal": ce.OrdinalEncoder,
        "sum": ce.SumEncoder,
        "polynomial": ce.PolynomialEncoder,
    },
}

# Utility
LIBRARY_ENCODER_STRATEGY_NAMES = list(
    LIBRARY_ENCODER_STRATEGIES["supervised"].keys()
) + list(LIBRARY_ENCODER_STRATEGIES["unsupervised"].keys())


class LibraryCategoryEncoder(CategoryEncoder):
    """Wrapper class for the `category-encoders` library."""

    def __init__(self, cat_names: List[str], cont_names: List[str], strategy: str):
        super().__init__(cat_names, cont_names)
        if strategy in LIBRARY_ENCODER_STRATEGIES["supervised"]:
            enc_t = LIBRARY_ENCODER_STRATEGIES["supervised"][strategy]
            self._is_supervised = True
        elif strategy in LIBRARY_ENCODER_STRATEGIES["unsupervised"]:
            enc_t = LIBRARY_ENCODER_STRATEGIES["unsupervised"][strategy]
            self._is_supervised = False
        else:
            raise ValueError(f"Unsupported strategy: {strategy}")
        self.enc = enc_t(cols=cat_names)
        self.placeholder = None
        self.strategy = strategy

    def transform(self, X: pd.DataFrame):
        """Encodes `X` using the given strategy."""
        return self.enc.transform(X)

    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.DataFrame] = None):
        """Fits the encoder with the given strategy over `X` and (optionally) `y`."""
        if self._is_supervised and y is None:
            raise ValueError(
                f"Given strategy {self.strategy} is supervised, but `y` argument is None."
            )
        return self.enc.fit_transform(X, y=y)

    def inverse_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Performs the inverse transform from continuous to categorical."""
        if hasattr(self.enc, "inverse_transform"):
            return self.enc.inverse_transform(X)
        else:
            raise AttributeError(
                'Invoked `inverse_transform` on a non-invertible encoding method. Invertible methods are "base-n", "binary", "ohe" and "ordinal".'
            )

    def get_feature_names(self) -> List[str]:
        """Returns the names of all transformed / added columns."""
        return self.enc.get_feature_names()
