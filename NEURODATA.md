# NeuroData: Neural Dataset Encoding for Privacy-Preserving Data Sharing

**NeuroData** encodes a tabular dataset as a compact neural network. Instead of sharing a CSV file, you share a `.model` file. Recipients query the model directly using a pandas-compatible interface — no raw data ever leaves the data owner.

---

## The Core Idea

Traditional data sharing: `data.csv` → recipient reads records directly.

NeuroData: `data.model` → recipient queries a function that reconstructs records on demand.

The model is a neural network trained to map **row indices → column values**. It memorizes the dataset. The model file is what gets shared — not the data.

```python
import neurodata

# Data owner: encode the dataset
neurodata.fit(df, path='health_data.model', epochs=800)

# Data recipient: query the model like a DataFrame
data = neurodata.load('health_data.model')
print(data.head())
print(data.groupby('country')['life_expectancy'].mean())
print(data.corr())
```

---

## Privacy Options

### Without Differential Privacy (epsilon = infinity)
- Fidelity: **98.6/100** on real African health data
- Access control: AES-256-GCM encryption with PBKDF2 key
- Use when: analytical fidelity is the priority

```python
neurodata.fit(df, path='data.model', password='secret123')
data = neurodata.load('data.model', password='secret123')
```

### With Differential Privacy (formal guarantee)
- Fidelity: **49.4/100** at epsilon=5, delta=1e-5
- Guarantee: (epsilon, delta)-DP via DP-SGD training
- Use when: formal privacy proof is required

```python
neurodata.fit(df, path='data.model', epsilon=5.0, delta=1e-5, epochs=1500)
```

---

## Neural Architecture

```
Row Index (integer)
       |
  Embedding(n_rows, 128)
       |
  Linear(128 -> 512) + ReLU
  Linear(512 -> 512) + ReLU
  Linear(512 -> 512) + ReLU
       |
   ┌───┴──────────────┐
   |                  |
Linear(512, n_num)   Linear(512, vocab_i)  x  n_cat_cols
   |                  |
Numerical values   Categorical logits
(denormalized)     (argmax -> label)
```

Training objective:
- Numerical columns: MSE loss on z-score normalized values
- Categorical columns: Cross-entropy loss

### Differentially Private Training (DP-SGD)
When `epsilon` is set:
1. Compute per-sample gradients
2. Clip each gradient to max norm C
3. Add Gaussian noise: N(0, sigma^2 * C^2 * I)
4. Apply optimizer step

Noise multiplier: `sigma = sqrt(2 * ln(1.25 / delta)) / epsilon`

---

## Model File Format

The `.model` file contains:
- Neural network weights (PyTorch state_dict)
- Column metadata: names, types, numerical means/stds, categorical encodings
- Dataset shape: n_rows
- Privacy budget spent (epsilon, delta if DP was used)

With a password, the file is encrypted:
- Key derivation: PBKDF2-HMAC-SHA256, 200,000 iterations, random 16-byte salt
- Encryption: AES-256-GCM with random 12-byte nonce
- File layout: `[salt (16 bytes)] [nonce (12 bytes)] [ciphertext]`

---

## NeuroFrame Interface

```python
data = neurodata.load('health.model')

data.head(10)              # first 10 rows as DataFrame
data.describe()            # summary statistics
data.groupby('country')    # GroupBy supporting .mean(), .sum(), .count()
data.corr()                # Pearson correlation matrix
data.value_counts('iso3')  # categorical frequency
data.to_csv('export.csv')  # export reconstructed data
data.to_dataframe()        # full reconstructed DataFrame
```

---

## Real Dataset: African Health Surveillance

**Sources:**
- World Bank Open Data API: 5 health indicators
- Our World in Data: malaria incidence

**Coverage:**
- 25 African countries: Ghana, Kenya, Nigeria, South Africa, Ethiopia, Tanzania, Uganda, Cameroon, Senegal, Mali, Burkina Faso, Guinea, Cote d'Ivoire, Mozambique, Zambia, Zimbabwe, Rwanda, Togo, Benin, Niger, Malawi, Lesotho, Sierra Leone, Liberia, Gambia
- Years: 2000 to 2022
- Records: 575 rows × 9 columns

**Columns:**
| Column | Type | Description |
|---|---|---|
| country | categorical | Country name |
| iso3 | categorical | ISO 3166-1 alpha-3 code |
| year | numerical | Year |
| maternal_mortality_per100k | numerical | Maternal deaths per 100,000 live births |
| infant_mortality_per1000 | numerical | Infant deaths per 1,000 live births |
| tb_incidence_per100k | numerical | TB cases per 100,000 population |
| hiv_incidence_per1000 | numerical | HIV new cases per 1,000 population |
| life_expectancy | numerical | Life expectancy at birth (years) |
| malaria_incidence_per1000 | numerical | Malaria cases per 1,000 population at risk |

---

## Experimental Results

### Statistical Fidelity (No-DP model)
Overall: **98.6/100**

The NeuroData model nearly perfectly memorizes the dataset. Life expectancy reconstruction achieves MAE of 0.00-0.01 years across six countries tested (Ghana, Kenya, Nigeria, South Africa, Ethiopia, Rwanda).

### Privacy-Accuracy Tradeoff
| Privacy Budget (epsilon) | Overall Fidelity |
|---|---|
| 0.5 (strongest) | 47.9/100 |
| 1.0 | 51.6/100 |
| 2.0 | 49.7/100 |
| 5.0 | 51.3/100 |
| 10.0 | 54.2/100 |
| No DP (infinity) | 98.6/100 |

### Baseline Comparison
| Method | Fidelity | Privacy |
|---|---|---|
| Raw CSV | 100.0/100 | None |
| Gaussian Noise | 94.5/100 | Weak |
| Aggregates Only | 90.0/100 | Moderate |
| NeuroData (no DP) | 98.6/100 | Partial (encrypted) |
| NeuroData (DP, eps=5) | 49.4/100 | Formal (DP) |

NeuroData without DP outperforms all privacy-preserving baselines while remaining within 1.4 points of the raw data ceiling.

---

## Project Structure

```
neurodata/
├── neurodata/
│   ├── __init__.py        # Public API: fit(), load(), fidelity()
│   ├── model.py           # NeuroDataEncoder neural architecture
│   ├── trainer.py         # Training loop with optional DP-SGD
│   ├── privacy.py         # DPOptimizer (DP-SGD implementation)
│   ├── access.py          # AES-GCM encryption/decryption
│   ├── fidelity.py        # Statistical fidelity metrics (KS, chi-square)
│   └── frame.py           # NeuroFrame pandas-compatible interface
├── data/
│   └── africa_health.csv  # Real African health surveillance dataset
├── examples/
│   ├── malaria_demo.py          # Basic usage demo
│   ├── full_demo.py             # DP + encryption + fidelity demo
│   ├── tradeoff_experiment.py   # Privacy-accuracy tradeoff
│   ├── build_africa_dataset.py  # Fetches World Bank + OWID data
│   ├── africa_health_demo.py    # Full pipeline on real data
│   └── baseline_comparison.py  # Comparison against 4 baselines
├── figures/
│   ├── fig1_architecture.png    # System architecture diagram
│   ├── fig2_tradeoff.png        # Privacy-accuracy tradeoff plot
│   ├── fig3_baselines.png       # Baseline comparison chart
│   └── fig4_reconstruction.png  # Multi-country reconstruction quality
├── paper/
│   └── NeuroData_JBHI_IEEE.docx # IEEE JBHI submission
└── NEURODATA.md           # This file
```

---

## Installation

```bash
pip install torch pandas numpy matplotlib scipy scikit-learn
```

No additional packages required. The project uses only standard scientific Python libraries.

---

## Quick Start

```python
import pandas as pd
import neurodata

# Load your dataset
df = pd.read_csv('your_data.csv')

# Encode as a model (no DP, fast)
neurodata.fit(df, 'my_data.model', epochs=500)

# Query the model
data = neurodata.load('my_data.model')
print(data.describe())

# Check fidelity
score = neurodata.fidelity(df, data)
print(f"Fidelity: {score['overall_score']:.1f}/100")
```

---

## Citation

```
R. Kessie, P. Nimbe, and V. Akoto-Adjepong, "NeuroData: A Neural Dataset
Encoding Framework for Privacy-Preserving Health Surveillance Data Sharing,"
IEEE Journal of Biomedical and Health Informatics, 2025 (under review).
```
