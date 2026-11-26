import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from .model import NeuroDataEncoder
from .privacy import DPOptimizer
from .access import save_locked


class NeuroDataTrainer:
    def __init__(self, df: pd.DataFrame, epochs=500, lr=1e-3, device=None,
                 epsilon=None, delta=1e-5, max_grad_norm=1.0):
        self.df = df.reset_index(drop=True)
        self.epochs = epochs
        self.lr = lr
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.epsilon = epsilon
        self.delta = delta
        self.max_grad_norm = max_grad_norm

        self._analyse_columns()
        self._build_tensors()
        self._build_model()

    def _analyse_columns(self):
        self.num_cols = []
        self.cat_cols = []
        self.col_order = list(self.df.columns)

        for col in self.df.columns:
            if pd.api.types.is_numeric_dtype(self.df[col]):
                self.num_cols.append(col)
            else:
                self.cat_cols.append(col)

        # encoders for categorical columns
        self.cat_encoders = {}
        self.cat_decoders = {}
        for col in self.cat_cols:
            vals = self.df[col].astype(str).unique().tolist()
            self.cat_encoders[col] = {v: i for i, v in enumerate(vals)}
            self.cat_decoders[col] = {i: v for i, v in enumerate(vals)}

        # fill NaN with column mean before training
        for col in self.num_cols:
            self.df[col] = self.df[col].fillna(self.df[col].mean())

        # normalisation stats for numerical columns
        self.num_means = {}
        self.num_stds = {}
        for col in self.num_cols:
            self.num_means[col] = float(self.df[col].mean())
            self.num_stds[col] = float(self.df[col].std()) or 1.0

    def _build_tensors(self):
        n = len(self.df)
        self.row_ids = torch.arange(n, device=self.device)

        if self.num_cols:
            num_data = np.zeros((n, len(self.num_cols)), dtype=np.float32)
            for i, col in enumerate(self.num_cols):
                num_data[:, i] = (self.df[col].values - self.num_means[col]) / self.num_stds[col]
            self.num_target = torch.tensor(num_data, device=self.device)

        self.cat_targets = []
        for col in self.cat_cols:
            encoded = self.df[col].astype(str).map(self.cat_encoders[col]).values
            self.cat_targets.append(torch.tensor(encoded, dtype=torch.long, device=self.device))

    def _build_model(self):
        cat_sizes = [len(self.cat_encoders[c]) for c in self.cat_cols]
        self.model = NeuroDataEncoder(
            n_rows=len(self.df),
            n_num_cols=len(self.num_cols),
            cat_col_sizes=cat_sizes,
        ).to(self.device)

    def train(self, verbose=True):
        base_opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        if self.epsilon is not None:
            opt = DPOptimizer(base_opt, n_samples=len(self.df),
                              epochs=self.epochs, epsilon=self.epsilon,
                              delta=self.delta, max_grad_norm=self.max_grad_norm)
            self.privacy_spent = opt.privacy_spent
        else:
            opt = base_opt
            self.privacy_spent = None

        mse = nn.MSELoss()
        ce = nn.CrossEntropyLoss()

        for epoch in range(1, self.epochs + 1):
            self.model.train()
            opt.zero_grad()
            out = self.model(self.row_ids)

            loss = torch.tensor(0.0, device=self.device)
            if self.num_cols:
                loss = loss + mse(out['numerical'], self.num_target)
            for i, cat_out in enumerate(out['categorical']):
                loss = loss + ce(cat_out, self.cat_targets[i])

            loss.backward()
            if self.epsilon is not None:
                opt.step(self.model.parameters())
            else:
                opt.step()

            if verbose and epoch % 100 == 0:
                print(f"Epoch {epoch}/{self.epochs}  loss={loss.item():.4f}")

        self.model.eval()
        return self

    def save(self, path: str, password: str = None):
        checkpoint = {
            'state_dict': self.model.state_dict(),
            'num_cols': self.num_cols,
            'cat_cols': self.cat_cols,
            'col_order': self.col_order,
            'cat_encoders': self.cat_encoders,
            'cat_decoders': self.cat_decoders,
            'num_means': self.num_means,
            'num_stds': self.num_stds,
            'n_rows': len(self.df),
            'privacy_spent': self.privacy_spent,
        }
        if password:
            save_locked(checkpoint, path, password)
        else:
            torch.save(checkpoint, path)
            print(f"Model saved → {path}")
