import torch
import numpy as np
import pandas as pd
from .kess import NeuroDataEncoder
from .access import load_locked


class NeuroFrame:
    """
    Drop-in pandas-like interface backed by a trained neural model.
    The raw data never needs to be present — only the .kess file.
    """

    def __init__(self, path: str, password: str = None):
        if password:
            checkpoint = load_locked(path, password)
        else:
            checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        self._num_cols   = checkpoint['num_cols']
        self._cat_cols   = checkpoint['cat_cols']
        self._col_order  = checkpoint['col_order']
        self._cat_dec    = checkpoint['cat_decoders']
        self._num_means  = checkpoint['num_means']
        self._num_stds   = checkpoint['num_stds']
        self._n_rows     = checkpoint['n_rows']

        cat_sizes = [len(checkpoint['cat_encoders'][c]) for c in self._cat_cols]
        self._model = NeuroDataEncoder(
            n_rows=self._n_rows,
            n_num_cols=len(self._num_cols),
            cat_col_sizes=cat_sizes,
        )
        self._model.load_state_dict(checkpoint['state_dict'])
        self._model.eval()

        self._cache = None  # reconstructed dataframe cache

    # ── reconstruction ───────────────────────────────────────────────────────

    def _reconstruct(self) -> pd.DataFrame:
        if self._cache is not None:
            return self._cache

        with torch.no_grad():
            row_ids = torch.arange(self._n_rows)
            out = self._model(row_ids)

        data = {}

        if self._num_cols:
            num = out['numerical'].numpy()
            for i, col in enumerate(self._num_cols):
                data[col] = num[:, i] * self._num_stds[col] + self._num_means[col]

        for i, col in enumerate(self._cat_cols):
            preds = out['categorical'][i].argmax(dim=1).numpy()
            data[col] = [self._cat_dec[col][p] for p in preds]

        self._cache = pd.DataFrame(data)[self._col_order]
        return self._cache

    # ── pandas-compatible interface ──────────────────────────────────────────

    @property
    def columns(self):
        return self._col_order

    @property
    def shape(self):
        return (self._n_rows, len(self._col_order))

    def __len__(self):
        return self._n_rows

    def __repr__(self):
        return f"NeuroFrame  rows={self._n_rows}  cols={len(self._col_order)}\n{self._reconstruct().__repr__()}"

    def __getitem__(self, key):
        return self._reconstruct()[key]

    def head(self, n=5):
        return self._reconstruct().head(n)

    def tail(self, n=5):
        return self._reconstruct().tail(n)

    def describe(self):
        return self._reconstruct().describe()

    def groupby(self, by):
        return self._reconstruct().groupby(by)

    def value_counts(self, col):
        return self._reconstruct()[col].value_counts()

    def mean(self):
        return self._reconstruct()[self._num_cols].mean()

    def std(self):
        return self._reconstruct()[self._num_cols].std()

    def corr(self):
        return self._reconstruct()[self._num_cols].corr()

    def to_csv(self, *args, **kwargs):
        return self._reconstruct().to_csv(*args, **kwargs)

    def to_dataframe(self) -> pd.DataFrame:
        return self._reconstruct().copy()
