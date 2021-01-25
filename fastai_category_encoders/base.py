from typing import List, Optional, Type

import pandas as pd

__all__ = [
    "CategoryEncoder",
    "CategoryEncoderPreprocessor",
    "CustomCategoryEncoder",
]


class CategoryEncoder:
    """
    Base interface for a class that processes categorical features,
    turning them into continuous ones.
    Possible subclasses include PCA strategies, embedding generation,
    one-hot encoding and more.

    Parameters
    ----------
    cat_names: List[str]
        List of categorical feature names to convert.

    cont_names: List[str]
        List of continuous feature names in the dataset.
    """

    _is_supervised: bool

    def __init__(self, cat_names: List[str], cont_names: List[str]):
        self.cat_names = cat_names
        self.cont_names = cont_names

    def fit(self, X: pd.DataFrame, y: Optional[pd.DataFrame] = None):
        # we just use fit_transform and drop the output
        # see https://github.com/scikit-learn-contrib/category_encoders/blob/master/README.md#examples
        self.fit_transform(X, y=y)

    def transform(self, X: pd.DataFrame):
        raise NotImplementedError

    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.DataFrame] = None):
        raise NotImplementedError

    def inverse_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError

    def get_feature_names(self) -> List[str]:
        raise NotImplementedError


class CategoryEncoderPreprocessor:
    """
    Base interface for a category encoder preprocessor.
    The main responsibility of a preprocessor is to perform
    data transformations from a Pandas DataFrame into the format
    used by the encoder (which could use a ML model, and thus
    require some form of data loading.)
    """

    def __init__(self, cat_names: List[str], cont_names: List[str], **kwargs):
        self.cat_names = cat_names
        self.cont_names = cont_names
        self.stored_kwargs = kwargs

    def process(self, df: pd.DataFrame, target: Optional[pd.DataFrame] = None, first: bool = False) -> any:
        """
        Processes `df` (and optionally `target`) into
        a data form usable by the category encoder.

        Parameters
        ----------
        df: pd.DataFrame
            The Pandas DataFrame containing the categorical features (and
            optionally continuous, depending on the encoding method)

        target: Optional[pd.DataFrame], default = None
            The optional target variable(s)

        first: bool, default = False
            If `True`, it means that the encoder is being trained with the processed data.
            For some preprocessors, this could require different operations.
        """
        raise NotImplementedError


class CustomCategoryEncoder(CategoryEncoder):
    """
    Subclass of `CategoryEncoder` that uses a `CategoryEncoderPreprocessor`
    instance to feed data to a trainable model.

    Extra arguments will be forwarded to the preprocessor.
    """

    # Placeholder preprocessor type for subclasses
    _preprocessor_cls: Type[CategoryEncoderPreprocessor] = None

    def __init__(self, cat_names: List[str], cont_names: List[str], *args, **kwargs):
        super().__init__(cat_names, cont_names)
        self._preprocessor = self._preprocessor_cls(cat_names, cont_names, *args, **kwargs)

    def encode(self, X) -> pd.DataFrame:
        """Encodes categoricals in `X` into continuous values."""
        raise NotImplementedError

    def decode(self, X: pd.DataFrame) -> pd.DataFrame:
        """Decodes `X` into categorical values."""
        raise NotImplementedError

    def before_fit(self, df: pd.DataFrame):
        """Performs setup steps using the DataFrame before training the model."""
        pass

    def fit(self, X):
        """Trains the model used to encode categoricals."""
        raise NotImplementedError

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Performs preprocessing steps and encodes categoricals."""
        X = self._preprocessor.process(X)
        return self.encode(X)

    def inverse_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Performs the inverse transform from continuous to categorical."""
        return self.decode(X)

    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Trains the model and encodes categoricals right away."""
        self.before_fit(X)
        X = self._preprocessor.process(X, first=True)
        self.fit(X)
        return self.encode(X)
