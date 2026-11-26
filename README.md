# NeuroData

**NeuroData** encodes a tabular dataset as a compact neural network. Instead of sharing a CSV file, you share a `.model` file. Recipients query it like a DataFrame — no raw data ever leaves the data owner.

```python
import neurodata

# Data owner: encode the dataset once
neurodata.fit(df, 'health_data.model', password='secret')

# Researcher: query without ever seeing the raw data
data = neurodata.load('health_data.model', password='secret')
print(data.head())
print(data.groupby('country')['life_expectancy'].mean())
print(data.corr())
```

---

## Why NeuroData

Health research centers in Africa have valuable data — clinical trials, disease surveillance, patient outcomes — that cannot be shared because of privacy regulations and the absence of safe data infrastructure.

Existing solutions (federated learning, synthetic GANs) require cloud connectivity, large compute budgets, and specialist data engineers. NeuroData requires none of these:

- **Encode once** on the data owner's local machine
- **Share a file** — a 1–5 MB `.model` instead of a CSV
- **Query anywhere** — the recipient uses a standard pandas-like API
- **No raw data moves** — the model reconstructs records on demand

NeuroData was built and validated on real African health surveillance data.

---

## Results

Tested on 575 records × 9 columns from 25 African countries (World Bank + Our World in Data, 2000–2022):

| Method | Fidelity | Privacy |
|---|---|---|
| Raw CSV | 100.0 / 100 | None |
| Gaussian noise | 94.5 / 100 | Weak |
| Aggregates only | 90.0 / 100 | Moderate |
| **NeuroData (no DP)** | **98.6 / 100** | Encrypted |
| NeuroData (DP ε=5) | 49.4 / 100 | Formal (ε,δ)-DP |

NeuroData without DP outperforms all privacy-preserving baselines while remaining within 1.4 points of the raw data ceiling.

---

## Privacy Options

### Without Differential Privacy
Fidelity: **98.6 / 100**. The model memorises the dataset with near-perfect reconstruction.
Access control via AES-256-GCM encryption — only someone with the password can load the file.

```python
neurodata.fit(df, 'data.model', password='secret')
data = neurodata.load('data.model', password='secret')
```

### With Differential Privacy
Fidelity: **~50 / 100** at ε=5. Formal (ε, δ)-DP guarantee via DP-SGD training.
Use when a mathematical privacy proof is required (e.g. ethics board submission, GDPR compliance).

```python
neurodata.fit(df, 'data.model', epsilon=5.0, delta=1e-5, epochs=1500)
```

---

## Neural Architecture

```
Row Index (integer)
       │
  Embedding(n_rows, 128)
       │
  Linear(128 → 512) + ReLU
  Linear(512 → 512) + ReLU
  Linear(512 → 512) + ReLU
       │
   ┌───┴──────────────────┐
   │                      │
Linear(512, n_num)   Linear(512, vocab_i)  × n_cat_cols
   │                      │
Numerical values     Categorical logits
```

Training objective: MSE on z-score normalised numerical columns + cross-entropy on categorical columns.

DP-SGD (when `epsilon` is set): per-sample gradient clipping → Gaussian noise injection → noise multiplier `σ = √(2 ln(1.25/δ)) / ε`.

---

## Model File Format

The `.model` file contains neural network weights, column metadata, and dataset shape.

With a password:
- Key derivation: PBKDF2-HMAC-SHA256, 200,000 iterations, random 16-byte salt
- Encryption: AES-256-GCM with random 12-byte nonce
- File layout: `[salt (16 B)] [nonce (12 B)] [ciphertext]`

---

## NeuroFrame API

```python
data = neurodata.load('health.model')

data.head(10)                              # first N rows as DataFrame
data.describe()                            # summary statistics
data.groupby('country')['gdp'].mean()      # GroupBy → mean, sum, count
data.corr()                                # Pearson correlation matrix
data.value_counts('region')               # categorical frequency
data.to_csv('export.csv')                  # export reconstructed data
data.to_dataframe()                        # full DataFrame
data.shape                                 # (n_rows, n_cols)
```

---

## Installation

```bash
git clone https://github.com/richmagkessie/neurodata.git
cd neurodata
pip install -e .
```

**Requirements** (auto-installed):

```
torch>=1.12  pandas>=1.3  numpy>=1.21  scipy>=1.7
scikit-learn>=1.0  matplotlib>=3.4  cryptography>=38.0
```

---

## Quick Start

```bash
python3 quickstart.py
```

This runs in ~30 seconds and demonstrates training, loading, encryption, password rejection, and fidelity reporting.

---

## Examples

| Script | What it shows |
|---|---|
| `quickstart.py` | Full pipeline in 30 seconds |
| `examples/malaria_demo.py` | Basic encode → query demo |
| `examples/full_demo.py` | DP training + AES-256-GCM encryption |
| `examples/africa_health_demo.py` | Real 25-country African surveillance dataset |
| `examples/tradeoff_experiment.py` | Privacy-accuracy tradeoff across ε values |
| `examples/baseline_comparison.py` | NeuroData vs Gaussian noise, aggregates |

---

## Project Structure

```
neurodata/
├── neurodata/
│   ├── __init__.py        # Public API: fit(), load(), fidelity()
│   ├── model.py           # NeuroDataEncoder neural architecture
│   ├── trainer.py         # Training loop with optional DP-SGD
│   ├── privacy.py         # DPOptimizer (DP-SGD implementation)
│   ├── access.py          # AES-GCM encryption / decryption
│   ├── fidelity.py        # Statistical fidelity metrics (KS, chi-square)
│   └── frame.py           # NeuroFrame pandas-compatible interface
├── data/
│   └── africa_health.csv  # Real African health surveillance dataset
├── examples/              # Runnable experiment scripts
├── figures/               # Publication-quality figures (300 DPI)
├── paper/                 # IEEE manuscript (under review)
├── quickstart.py          # Entry point — runs in 30 seconds
├── setup.py
└── requirements.txt
```

---

## Citation

```bibtex
@article{kessie2025neurodata,
  title   = {NeuroData: Deep Neural Tabular Encoding for Privacy-Preserving
             Health Surveillance Data Sharing with Formal Differential Privacy},
  author  = {Kessie, Richmond and Nimbe, Peter and Akoto-Adjepong, Vivian},
  journal = {IEEE Transactions on Computational Biology and Bioinformatics},
  year    = {2025},
  note    = {Under review}
}
```

---

## Authors

**Richmond Kessie** · University of Energy and Natural Resources, Ghana  
**Peter Nimbe** · University of Energy and Natural Resources, Ghana  
**Vivian Akoto-Adjepong** · University of Energy and Natural Resources, Ghana

---

## License

MIT License. See [LICENSE](LICENSE) for details.
