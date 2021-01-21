"""
This module implements the base `CustomCategoryEncoder` with
the AutoEmbedder model from `gensim` to generate unsupervised
embeddings from categorical features.
"""
from typing import Dict, Iterable, List, Type, Union

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from fastai.learner import Learner
from fastai.tabular.core import TabDataLoader
from fastai.tabular.data import (Categorify, FillMissing, Normalize,
                                 TabularDataLoaders)
from fastai.tabular.model import get_emb_sz

from ..base import CategoryEncoderPreprocessor, CustomCategoryEncoder
from .autoembedder import AutoEmbedder
from .loss import EmbeddingLoss
from .tensor import expanded

__all__ = [
    "AutoEmbedderPreprocessor",
    "AutoEmbedderCategoryEncoder",
]


class AutoEmbedderPreprocessor(CategoryEncoderPreprocessor):
    """Uses an `AutoEmbedder` model to perform encoding of categorical features."""

    def process(self, df: pd.DataFrame, first: bool = False) -> TabularDataLoaders:
        if df is None:
            raise RuntimeError("DataFrame is missing")
        # Setup feature names + processes
        procs = [FillMissing, Categorify, Normalize]
        if first:
            self.data = TabularDataLoaders.from_df(
                df,
                cat_names=self.cat_names,
                cont_names=self.cont_names,
                procs=procs,
                valid_idx=[],
            )
            return self.data
        else:
            return self.data.test_dl(df)


class AutoEmbedderCategoryEncoder(CustomCategoryEncoder):
    _preprocessor_cls: Type[CategoryEncoderPreprocessor] = AutoEmbedderPreprocessor
    learn: Learner = None
    emb_szs: Dict[str, int] = None

    def encode(self, X: TabDataLoader):
        """Encodes all elements in `data`."""
        data = X if isinstance(X, TabDataLoader) else X.train
        preds = self.learn.get_preds(dl=data, reorder=False)[0].cpu().numpy()
        return pd.DataFrame(preds, columns=self.get_feature_names())

    def fit(self, X: TabularDataLoaders):
        """Creates the learner and trains it."""
        emb_szs = get_emb_sz(X.train_ds, {})
        self.emb_szs = {col: sz for col, sz in zip(self.cat_names, emb_szs)}
        n_conts = len(X.cont_names)
        n_cats = sum(list(map(lambda e: e[1], emb_szs)))
        in_sz = n_conts + n_cats
        out_sz = n_conts + len(X.cat_names)
        # Create the embedding model
        model = AutoEmbedder(in_sz, out_sz, emb_szs, [2000, 1000])
        self.learn = Learner(X, model, loss_func=EmbeddingLoss(model), wd=1.0)
        # TODO hide training progress?
        with self.learn.no_bar():
            self.learn.fit_one_cycle(20, lr_max=3e-3)

    def decode(self, X: pd.DataFrame) -> pd.DataFrame:
        """Decodes multiple items for one feature embedding."""
        column_idx = 0
        df = pd.DataFrame()
        data = torch.tensor(X[self.get_feature_names()].values)
        embeddings = self.learn.model.embeddings.embeddings
        # Split data into chunks depending on embedding sizes
        data = torch.split(
            data, list(map(lambda o: o[1], self.emb_szs.values())), dim=-1
        )
        # Iterate over features, decoding each one for all rows
        for (embedding_vectors,
             embedding_layer,
             (colname, (n_unique_values, embedding_size))) in zip(data, embeddings, self.emb_szs.items()):
            # Calculate the embedding output for each category value
            cat_embeddings = embedding_layer(torch.tensor(range(n_unique_values)).to(device=embedding_layer.device))
            # Compute cosine similarity over embeddings
            most_similar = expanded(embedding_vectors, cat_embeddings, lambda a, b: F.cosine_similarity(a, b, dim=-1))
            # Map values to their most similar category
            most_similar = most_similar.argmax(dim=-1)
            # Save data into decoded column
            df[colname] = most_similar.cpu().numpy()
            # move forward the column index
            column_idx += embedding_size
        return df

    def get_feature_names(self) -> List[str]:
        """
        Returns a list of encoded feature names.
        For embeddings, this is a list of original categorical names followed by embedding index,
        e.g. [feature_a_0, feature_a_1, feature_b_0, feature_b_1].
        """
        return [f"{column}_{feature_num}" for column in self.cat_names for feature_num in range(self.emb_szs[column][1])]

    def get_emb_szs(self):
        """Returns a dict of embedding sizes for each categorical feature."""
        return self.emb_szs
