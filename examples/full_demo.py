"""
Full demo: DP training + encryption + fidelity report
Simulates Kintampo sharing a malaria dataset via a locked .model file.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
OUT = Path(__file__).parent

import pandas as pd
import numpy as np
import neurodata

np.random.seed(42)
n = 200
df = pd.DataFrame({
    'age':         np.random.randint(1, 80, n),
    'temperature': np.round(np.random.normal(38.2, 1.1, n), 1),
    'parasitemia': np.random.randint(0, 50000, n),
    'region':      np.random.choice(['Ashanti', 'Brong-Ahafo', 'Northern', 'Upper East'], n),
    'outcome':     np.random.choice(['Recovered', 'Admitted', 'Referred'], n),
})

# ── KINTAMPO SIDE: train with DP, encrypt, share the file ────────────────────
print("=== Kintampo: training with differential privacy (ε=2.0) ===")
neurodata.fit(
    df,
    path=str(OUT / 'malaria_locked.model'),
    epochs=600,
    epsilon=2.0,
    password='kintampo2026',
    verbose=True,
)

# ── RESEARCHER SIDE: load with password, work normally ───────────────────────
print("\n=== Researcher: loading locked model ===")
data = neurodata.load(str(OUT / 'malaria_locked.model'), password='kintampo2026')

print("\ndata.head():")
print(data.head())

print("\ndata.describe():")
print(data.describe())

print("\ndata.groupby('outcome')['age'].mean():")
print(data.groupby('outcome')['age'].mean())

# ── Wrong password test ───────────────────────────────────────────────────────
print("\n=== Wrong password attempt ===")
try:
    bad = neurodata.load(str(OUT / 'malaria_locked.model'), password='wrongpass')
except ValueError as e:
    print(f"Blocked: {e}")

# ── Fidelity report ───────────────────────────────────────────────────────────
print("\n=== Fidelity report (DP model vs original data) ===")
neurodata.fidelity(df, data)
