import torch
import torch.nn as nn
import numpy as np


class NeuroDataEncoder(nn.Module):
    """Maps row index → all column values."""

    def __init__(self, n_rows, n_num_cols, cat_col_sizes):
        super().__init__()
        self.n_rows = n_rows
        self.n_num_cols = n_num_cols
        self.cat_col_sizes = cat_col_sizes  # list of vocab sizes for categorical cols

        hidden = 512
        self.row_embed = nn.Embedding(n_rows, 128)

        self.trunk = nn.Sequential(
            nn.Linear(128, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )

        # numerical head
        self.num_head = nn.Linear(hidden, n_num_cols) if n_num_cols > 0 else None

        # one head per categorical column
        self.cat_heads = nn.ModuleList([
            nn.Linear(hidden, vocab) for vocab in cat_col_sizes
        ])

    def forward(self, row_ids):
        x = self.row_embed(row_ids)
        x = self.trunk(x)
        outputs = {}
        if self.num_head is not None:
            outputs['numerical'] = self.num_head(x)
        outputs['categorical'] = [head(x) for head in self.cat_heads]
        return outputs
