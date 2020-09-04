"""
This module implements the base `CustomCategoryEncoder` with
the AutoEmbedder model from `gensim` to generate unsupervised
embeddings from categorical features.
"""
import torch
import numpy as np
import pandas as pd

from typing import Iterable, Type, Union, List, Dict
from fastai.tabular import (
    TabularList,
    TabularDataBunch,
    Categorify,
    FillMissing,
    Normalize,
    DatasetType,
)
from fastai.basic_train import Learner

from ..base import CustomCategoryEncoder, CategoryEncoderPreprocessor
from .autoembedder import AutoEmbedder
from .loss import EmbeddingLoss


__all__ = [
    "AutoEmbedderPreprocessor",
    "AutoEmbedderCategoryEncoder",
]


class AutoEmbedderPreprocessor(CategoryEncoderPreprocessor):
    """Uses an `AutoEmbedder` model to perform encoding of categorical features."""

    def process(self, df: pd.DataFrame, first: bool = False) -> TabularDataBunch:
        if df is None:
            raise RuntimeError("DataFrame is missing")
        # Setup feature names + processes
        procs = [FillMissing, Categorify, Normalize]
        if first:
            self.data = (
                TabularList.from_df(
                    df,
                    cat_names=self.cat_names,
                    cont_names=self.cont_names,
                    procs=procs,
                )
                .split_none()
                .label_empty()
                .databunch()
            )
            return self.data
        else:
            self.data.test_dl = None
            self.data.add_test(
                TabularList.from_df(
                    df, cat_names=self.cat_names, cont_names=self.cont_names
                )
            )
            return self.data.test_dl


class AutoEmbedderCategoryEncoder(CustomCategoryEncoder):
    _preprocessor_cls: Type[CategoryEncoderPreprocessor] = AutoEmbedderPreprocessor
    learn: Learner = None
    emb_szs: Dict[str, int] = None

    def encode(self, X: Union[TabularDataBunch, torch.utils.data.DataLoader]):
        """Encodes all elements in `data`."""
        ds_type = (
            DatasetType.Train if isinstance(X, TabularDataBunch) else DatasetType.Test
        )
        print(f"Encoding {ds_type}")
        preds = self.learn.get_preds(ds_type=ds_type)[0].cpu().numpy()
        print(preds.shape)
        return pd.DataFrame(preds, columns=self.get_feature_names())

    def fit(self, X: TabularDataBunch):
        """Creates the learner and trains it."""
        emb_szs = X.get_emb_szs({})
        self.emb_szs = {col: sz[1] for col, sz in zip(self.cat_names, emb_szs)}
        n_conts = len(X.cont_names)
        n_cats = sum(list(map(lambda e: e[1], emb_szs)))
        in_sz = n_conts + n_cats
        out_sz = n_conts + len(X.cat_names)
        # Create the embedding model
        model = AutoEmbedder(in_sz, out_sz, emb_szs, [2000, 1000])
        self.learn = Learner(X, model, loss_func=EmbeddingLoss(model), wd=1.0)
        # TODO hide training progress?
        self.learn.fit_one_cycle(1, max_lr=3e-3)

    def decode(self, X: pd.DataFrame) -> pd.DataFrame:
        """Decodes multiple items for one feature embedding."""
        start_emb = 0
        df = pd.DataFrame()
        data = torch.tensor(X.values)
        embeddings = self.learn.model.embeddings.embeddings
        emb_szs = list(map(lambda e: e.embedding_dim, embeddings))
        cat_szs = list(map(lambda e: e.num_embeddings, embeddings))
        for emb, emb_sz, n_classes, name in zip(
            embeddings, emb_szs, cat_szs, self.cat_names
        ):
            cat_embeddings = [emb(torch.tensor([c]).cuda()) for c in range(n_classes)]
            item_feature = data[start_emb : start_emb + emb_sz]
            most_similar = torch.nn.functional.cosine_similarity(
                item_feature.unsqueeze(0), torch.cat(cat_embeddings, dim=0)
            )
            most_similar = most_similar.argmax()
            print((name, len(most_similar)))
            df[name] = [most_similar.cpu().numpy()]
            start_emb += emb_sz
            # TODO: map back into strings?
        return df

    def get_feature_names(self) -> List[str]:
        """TODO document"""
        return [
            f"{column}_{feature_num}"
            for column in self.cat_names
            for feature_num in range(self.emb_szs[column])
        ]
