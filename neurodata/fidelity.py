"""
Fidelity metrics: quantify how faithfully the NeuroFrame reproduces
the statistical properties of the original dataset.
"""
import pandas as pd
import numpy as np
from scipy import stats


def evaluate(original: pd.DataFrame, reconstructed: pd.DataFrame) -> dict:
    """
    Returns a fidelity report comparing original to reconstructed data.

    Numerical columns : KS test p-value + mean/std absolute error
    Categorical columns: chi-square p-value + distribution overlap
    Overall fidelity score: 0–100 (higher = more faithful)
    """
    report = {'columns': {}, 'overall_score': None}
    scores = []

    num_cols = original.select_dtypes(include='number').columns.tolist()
    cat_cols = original.select_dtypes(exclude='number').columns.tolist()

    for col in num_cols:
        orig = original[col].dropna().values.astype(float)
        recon = reconstructed[col].dropna().values.astype(float)

        ks_stat, ks_p = stats.ks_2samp(orig, recon)
        mean_err = abs(orig.mean() - recon.mean())
        std_err  = abs(orig.std()  - recon.std())

        col_score = max(0.0, (1 - ks_stat) * 100)
        scores.append(col_score)

        report['columns'][col] = {
            'type': 'numerical',
            'ks_statistic': round(ks_stat, 4),
            'ks_p_value':   round(ks_p, 4),
            'mean_abs_error': round(mean_err, 4),
            'std_abs_error':  round(std_err, 4),
            'col_score': round(col_score, 2),
        }

    for col in cat_cols:
        orig_counts  = original[col].astype(str).value_counts()
        recon_counts = reconstructed[col].astype(str).value_counts()

        all_cats = set(orig_counts.index) | set(recon_counts.index)
        orig_vec  = np.array([orig_counts.get(c, 0)  for c in all_cats], dtype=float)
        recon_vec = np.array([recon_counts.get(c, 0) for c in all_cats], dtype=float)

        # distribution overlap (1 - total variation distance)
        orig_p  = orig_vec  / orig_vec.sum()
        recon_p = recon_vec / recon_vec.sum()
        overlap = 1 - 0.5 * np.abs(orig_p - recon_p).sum()

        chi2, chi2_p = stats.chisquare(recon_vec + 1e-9, f_exp=orig_vec + 1e-9)
        col_score = overlap * 100
        scores.append(col_score)

        report['columns'][col] = {
            'type': 'categorical',
            'distribution_overlap': round(overlap, 4),
            'chi2_p_value': round(chi2_p, 4),
            'col_score': round(col_score, 2),
        }

    report['overall_score'] = round(np.mean(scores), 2)
    return report


def print_report(report: dict):
    print("\n╔══════════════════════════════════════════════╗")
    print(f"║  NeuroData Fidelity Report                   ║")
    print(f"║  Overall Score: {report['overall_score']:>5.1f} / 100                  ║")
    print("╠══════════════════════════════════════════════╣")
    for col, stats_dict in report['columns'].items():
        print(f"║  {col:<20}  score={stats_dict['col_score']:>5.1f}         ║")
        if stats_dict['type'] == 'numerical':
            print(f"║    KS stat={stats_dict['ks_statistic']:.4f}  mean_err={stats_dict['mean_abs_error']:.4f}  ║")
        else:
            print(f"║    overlap={stats_dict['distribution_overlap']:.4f}  χ² p={stats_dict['chi2_p_value']:.4f}       ║")
    print("╚══════════════════════════════════════════════╝")
