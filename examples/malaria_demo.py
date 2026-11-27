"""
Demo: encode a malaria-like health dataset into a .kess file,
then work with it exactly like a CSV — no raw data needed.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import neurodata

# ── 1. Simulate a small health dataset (stand-in for real Kintampo data) ──────
np.random.seed(42)
n = 200
df = pd.DataFrame({
    'age':        np.random.randint(1, 80, n),
    'temperature':np.round(np.random.normal(38.2, 1.1, n), 1),
    'parasitemia':np.random.randint(0, 50000, n),
    'region':     np.random.choice(['Ashanti', 'Brong-Ahafo', 'Northern', 'Upper East'], n),
    'outcome':    np.random.choice(['Recovered', 'Admitted', 'Referred'], n),
})
print("=== Original data (first 5 rows) ===")
print(df.head())

# ── 2. Train the model on the data — this is what Kintampo runs locally ───────
OUT = Path(__file__).parent
neurodata.fit(df, path=str(OUT / 'malaria.kess'), epochs=600, verbose=True)

# ── 3. Load the model — this is what the researcher receives ──────────────────
print("\n=== Loading model (no CSV, no raw data) ===")
data = neurodata.load(str(OUT / 'malaria.kess'))

# ── 4. Use it exactly like a dataframe ────────────────────────────────────────
print("\ndata.shape:", data.shape)
print("\ndata.head():")
print(data.head())

print("\ndata.describe():")
print(data.describe())

print("\ndata['region'].value_counts():")
print(data.value_counts('region'))

print("\ndata.groupby('outcome')['age'].mean():")
print(data.groupby('outcome')['age'].mean())

print("\ndata.corr():")
print(data.corr())

print("\n✓ Everything works. The researcher never touched the original CSV.")
