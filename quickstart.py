"""
NeuroData — Quick Start
========================
Runs in ~30 seconds. Shows the full pipeline:
  1. Train a model on a health dataset
  2. Load it like a DataFrame — no raw data needed
  3. Test encryption (password protection)
  4. Check fidelity vs the original data

Usage:
  python quickstart.py
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import numpy as np
import neurodata

ROOT = Path(__file__).parent

# ── 1. Sample health dataset (or swap in your own CSV) ────────────────────────
np.random.seed(42)
n = 200
df = pd.DataFrame({
    'age':         np.random.randint(1, 80, n),
    'temperature': np.round(np.random.normal(38.2, 1.1, n), 1),
    'parasitemia': np.random.randint(0, 50000, n),
    'region':      np.random.choice(['Ashanti', 'Brong-Ahafo', 'Northern', 'Upper East'], n),
    'outcome':     np.random.choice(['Recovered', 'Admitted', 'Referred'], n),
})

print(f"Original data:  {df.shape[0]} rows × {df.shape[1]} columns")
print(df.head(3).to_string(), "\n")

# ── 2. Train and save as a .kess file ────────────────────────────────────────
print("Training NeuroData model (200 epochs)...")
neurodata.fit(df, path=str(ROOT / 'quickstart.kess'), epochs=200)
print()

# ── 3. Load and query — no CSV needed ─────────────────────────────────────────
data = neurodata.load(str(ROOT / 'quickstart.kess'))

print("data.head():")
print(data.head(3).to_string())
print()

print("data.groupby('outcome')['age'].mean():")
print(data.groupby('outcome')['age'].mean())
print()

print("data.corr():")
print(data.corr().round(3))
print()

# ── 4. Encrypted model (password-protected) ───────────────────────────────────
print("Training encrypted model (password protected)...")
neurodata.fit(df, path=str(ROOT / 'quickstart_locked.kess'),
              epochs=200, password='secret123')
print()

locked = neurodata.load(str(ROOT / 'quickstart_locked.kess'), password='secret123')
print("Loaded encrypted model — data.describe():")
print(locked.describe().round(2))
print()

try:
    neurodata.load(str(ROOT / 'quickstart_locked.kess'), password='wrongpass')
except ValueError as e:
    print(f"Wrong password correctly rejected: {e}")
print()

# ── 5. Fidelity check ─────────────────────────────────────────────────────────
print("Fidelity vs original data:")
neurodata.fidelity(df, data)
