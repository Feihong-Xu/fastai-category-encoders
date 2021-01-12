"""
This module implements the base `CustomCategoryEncoder` with
the FastText model from `gensim` to generate unsupervised
embeddings from categorical features.
"""
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from .base import CategoryEncoderPreprocessor, CustomCategoryEncoder
from .functional import find
from .utils import calc_embedding_size

__all__ = [
    "FastTextPreprocessor",
    "FastTextCategoryEncoder",
]


class FastTextPreprocessor(CategoryEncoderPreprocessor):
    def __init__(
        self,
        cat_names: List[str],
        cont_names: List[str],
        groupby: Optional[List[str]] = None,
    ):
        super().__init__(cat_names, cont_names)
        self.groupby = groupby

    def process(self, df: pd.DataFrame, first: bool = False) -> List[List[str]]:
        if df is None:
            raise RuntimeError("DataFrame is missing")
        df = df[self.cat_names].astype(str)
        sentences = self.build_horizontal_context(df)
        if first and self.groupby is not None:
            sentences += self.build_vertical_context(df, self.groupby)
        return sentences

    def build_horizontal_context(self, df: pd.DataFrame) -> List[List[str]]:
        """
        Builds sentences using values of this processor's columns in the DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame to be used when building sentences.
        """
        df = df[self.cat_names]
        return [
            [f"{col}__{val}" for col, val in zip(df.columns, df.values[i])]
            for i in range(df.values.shape[0])
        ]

    def build_vertical_context(self, df: pd.DataFrame, groupby_cols: List[str]) -> List[List[str]]:
        """
        Builds sentences by joining this processor's columns
        on multiple rows by grouping with `groupby_cols`.

        Parameters
        ----------
        df : pd.DataFrame
            The source DataFrame.
        groupby_cols : List[str]
            The list of columns to be used for the group operation.
        """
        sentences = []
        if not all(map(lambda c: c in df.columns, groupby_cols)):
            return sentences
        for gb_col in groupby_cols:
            for _, group in df.groupby(gb_col):
                if len(group) > 2:
                    sentences.append(self.build_horizontal_context(group)[0])
        return sentences


class FastTextCategoryEncoder(CustomCategoryEncoder):
    _preprocessor_cls = FastTextPreprocessor
    model = None
    emb_sz: int = None

    def before_fit(self, X: pd.DataFrame):
        """Performs setup steps before `fit`."""
        self.emb_sz = calc_embedding_size(X[self.cat_names])

    def encode(self, X) -> pd.DataFrame:
        """Encodes all elements in `data`."""
        ret = np.array([[self.model.wv[word] for word in sentence] for sentence in X])
        return pd.DataFrame(ret.reshape(len(X), -1), columns=self.get_feature_names())

    def fit(self, X):
        """Trains a FastText model for categorical encoding into embeddings."""
        try:
            from gensim.models import FastText
        except Exception:
            raise Exception(
                f"In order to use {self.__class__.__name__} you need to install the `gensim` library."
            )
        self.model = FastText(
            size=self.emb_sz, batch_words=1_000, min_count=1, sample=0, workers=10,
        )
        self.model.window = len(X[0])
        self.model.build_vocab(sentences=X)
        self.model.train(sentences=X, total_examples=len(X), epochs=5)

    def decode(self, X: pd.DataFrame) -> pd.DataFrame:
        """Decodes `X` into categorical values."""
        TOP_N = 1000
        encoded_names = self.get_feature_names()
        x_cont = X[encoded_names].values
        x_cont = np.split(x_cont, x_cont.shape[-1] // self.emb_sz, axis=-1)
        cols = []
        for colname, feature_emb in zip(self.cat_names, x_cont):
            # Map each embedding vector into a list of best-matching words
            topn_words = [map(lambda o: o[0], self.model.wv.similar_by_vector(embedding, topn=TOP_N)) for embedding in feature_emb]
            # Look for words that contain the feature name
            matches = map(lambda words: str(find(lambda w: w.split("__")[0] == colname, words)), topn_words)
            # Remove column name from words
            matches = map(lambda w: w.split("__")[1], matches)
            # Add found words in columns
            cols.append(list(matches))
        return pd.DataFrame(np.array(cols).transpose(), columns=self.cat_names)

    def get_feature_names(self) -> List[str]:
        """
        Returns a list of encoded feature names.
        For embeddings, this is a list of original categorical names followed by embedding index,
        e.g. [feature_a_0, feature_a_1, feature_b_0, feature_b_1].
        """
        mapped_names = map(lambda col: [f"{col}_{count}" for count in range(self.emb_sz)], self.cat_names)
        return np.array(list(mapped_names)).flatten().tolist()

    def get_emb_szs(self):
        """Returns a dictionary of embedding sizes for each categorical feature."""
        return {col: self.emb_sz for col in self.cat_names}
