"""
NeuroData demo on real African health surveillance data.
25 countries, 2000-2022, 6 WHO/World Bank health indicators.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
ROOT = Path(__file__).parent.parent
OUT  = Path(__file__).parent

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import neurodata
from neurodata.fidelity import evaluate, print_report

df = pd.read_csv(ROOT / 'data' / 'africa_health.csv')
print(f"Real dataset: {df.shape[0]} rows x {df.shape[1]} columns")
print(f"Countries: {df['country'].nunique()} | Years: {df['year'].min()}-{df['year'].max()}")
print(df.head(5).to_string())

# ── 1. Train without DP — full fidelity ceiling ───────────────────────────────
print("\n=== Training (no DP) ===")
neurodata.fit(df, path=str(OUT / 'africa_nodp.model'), epochs=800, verbose=False)
data_nodp = neurodata.load(str(OUT / 'africa_nodp.model'))

print("\ndata.head():")
print(data_nodp.head())

print("\ndata.describe():")
print(data_nodp.describe())

print("\ndata.groupby('country')['life_expectancy'].mean().sort_values().tail(10):")
print(data_nodp.groupby('country')['life_expectancy'].mean().sort_values().tail(10))

# ── 2. Fidelity report ────────────────────────────────────────────────────────
print("\n=== Fidelity Report (no DP) ===")
report_nodp = neurodata.fidelity(df, data_nodp)

# ── 3. DP tradeoff on real data ───────────────────────────────────────────────
print("\n=== Privacy-Accuracy Tradeoff on Real Data ===")
eps_vals = [1.0, 2.0, 5.0, 10.0, None]
eps_labels = ['1.0', '2.0', '5.0', '10.0', '∞']
results = []

for eps, label in zip(eps_vals, eps_labels):
    neurodata.fit(df, path=str(OUT / f'africa_eps{label}.model'),
                  epochs=800, epsilon=eps, verbose=False)
    d = neurodata.load(str(OUT / f'africa_eps{label}.model'))
    r = evaluate(df, d.to_dataframe())
    results.append({'label': label, 'score': r['overall_score']})
    print(f"  ε={label:>4}  fidelity={r['overall_score']:.1f}/100")

# ── 4. Plot ───────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle('NeuroData on Real African Health Surveillance Data\n'
             '25 Countries · 575 Records · WHO/World Bank Indicators', fontsize=13, fontweight='bold')

# Left: tradeoff bar chart
ax = axes[0]
x = list(range(len(results)))
scores = [r['score'] for r in results]
colors = ['#c0392b']*4 + ['#27ae60']
bars = ax.bar(x, scores, color=colors, edgecolor='white', width=0.6)
ax.axhline(scores[-1], color='#27ae60', linestyle='--', linewidth=1.2, label='No-DP ceiling')
ax.set_xticks(x)
ax.set_xticklabels([r['label'] for r in results], fontsize=11)
ax.set_xlabel('Privacy Budget ε  (smaller = stronger privacy)', fontsize=11)
ax.set_ylabel('Fidelity Score (0–100)', fontsize=11)
ax.set_title('Privacy–Accuracy Tradeoff', fontsize=12)
ax.set_ylim(0, 110)
ax.legend(fontsize=10)
for bar, s in zip(bars, scores):
    ax.text(bar.get_x()+bar.get_width()/2, s+1, f'{s:.1f}', ha='center', fontsize=10)

# Right: Ghana life expectancy — original vs reconstructed
ax2 = axes[1]
gh_orig  = df[df['country']=='Ghana'].sort_values('year')
gh_recon = data_nodp.to_dataframe()
gh_recon = gh_recon[gh_recon['country']=='Ghana'].sort_values('year')

ax2.plot(gh_orig['year'],  gh_orig['life_expectancy'],  'o-', color='#2980b9', label='Original', linewidth=2)
ax2.plot(gh_recon['year'], gh_recon['life_expectancy'], 's--', color='#e74c3c', label='NeuroData', linewidth=2)
ax2.set_xlabel('Year', fontsize=11)
ax2.set_ylabel('Life Expectancy', fontsize=11)
ax2.set_title('Ghana: Life Expectancy\nOriginal vs NeuroData Reconstruction', fontsize=12)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
out = str(OUT / 'africa_results.png')
plt.savefig(out, dpi=150, bbox_inches='tight')
print(f"\nFigure saved → {out}")
