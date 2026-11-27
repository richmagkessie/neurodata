# neurodata (R package)

R interface to [NeuroData](https://github.com/richmagkessie/neurodata) — a neural dataset encoding
framework for privacy-preserving health data sharing.

Wraps the Python package via [reticulate](https://rstudio.github.io/reticulate/), exposing the same
three-step workflow in idiomatic R.

---

## What it does

Instead of sharing a raw CSV, you train a neural network that **memorises** the dataset as model
weights. Recipients query the model and get back a data frame — statistically faithful, never the
original rows. Optional differential privacy (DP-SGD) and AES-256-GCM encryption are supported.

---

## Installation

### 1. Install the Python backend

```bash
# clone and install the Python package
git clone https://github.com/richmagkessie/neurodata.git
cd neurodata
pip install -e .
```

### 2. Install the R package

```r
# from source (inside the cloned repo)
install.packages("R/neurodata", repos = NULL, type = "source")

# or with devtools
devtools::install("R/neurodata")
```

### 3. Point reticulate at the right Python

```r
library(reticulate)
use_python(Sys.which("python3"), required = TRUE)   # or use_virtualenv / use_condaenv
```

---

## Usage

```r
library(neurodata)

# ── 1. Load your data ────────────────────────────────────────────────────────
df <- read.csv("data/malaria_rct_synthetic.csv")

# ── 2. Train (encode) the model ──────────────────────────────────────────────
neurodata_fit(df, "trial.kess", epochs = 800L)

# with differential privacy (ε = 5)
neurodata_fit(df, "trial_dp.kess", epochs = 1500L, epsilon = 5.0)

# with AES-256-GCM encryption
neurodata_fit(df, "trial_locked.kess", password = "s3cret!")

# ── 3. Reconstruct (share and use the .kess file, not the CSV) ──────────────
recon <- neurodata_load("trial.kess")
head(recon)
summary(recon)

# ── 4. Measure statistical fidelity ─────────────────────────────────────────
nd_summary(df, recon)
# --- NeuroData Fidelity Report ---
# Overall Score:  94.2 / 100
#
#   age_months                   score =  96.3
#   treatment_arm                score =  98.1
#   trial_outcome                score =  91.7
#   ...
# ---------------------------------
```

---

## Functions

| Function | Description |
|---|---|
| `neurodata_fit(df, path, ...)` | Train and save a `.kess` file |
| `neurodata_load(path, ...)` | Load a model; returns a `data.frame` |
| `neurodata_fidelity(original, recon)` | Returns fidelity report as a named list |
| `nd_summary(original, recon)` | Prints per-column fidelity scores to console |

### `neurodata_fit` parameters

| Parameter | Default | Description |
|---|---|---|
| `df` | — | `data.frame` or tibble to encode |
| `path` | — | Output path for the `.kess` file |
| `epochs` | `500L` | Training epochs (800+ for higher fidelity) |
| `epsilon` | `NULL` | DP budget — smaller = stronger privacy; `NULL` = no DP |
| `delta` | `1e-5` | DP failure probability |
| `password` | `NULL` | AES-256-GCM encryption password |
| `verbose` | `TRUE` | Print training progress |

---

## Downstream analysis

The returned data frame is a plain `data.frame`, compatible with base R and tidyverse:

```r
library(dplyr)

recon |>
  group_by(treatment_arm) |>
  summarise(
    n            = n(),
    success_rate = mean(trial_outcome == "Treatment Success"),
    ae_rate      = mean(adverse_event != "No_AE")
  )
```

---

## Privacy options

| Mode | How to invoke | Guarantee |
|---|---|---|
| No privacy | `epsilon = NULL` (default) | Highest fidelity |
| Differential privacy | `epsilon = 5.0` (or lower) | Formal (ε, δ)-DP via DP-SGD |
| Encryption | `password = "..."` | AES-256-GCM; model unusable without password |
| Both | `epsilon = 5.0, password = "..."` | DP + encryption combined |

---

## Citation

```bibtex
@software{kessie2025neurodata,
  author  = {Kessie, Richmond},
  title   = {NeuroData: Neural Dataset Encoding for Privacy-Preserving Health Data Sharing},
  year    = {2025},
  url     = {https://github.com/richmagkessie/neurodata}
}
```

---

## License

MIT © Richmond Kessie
