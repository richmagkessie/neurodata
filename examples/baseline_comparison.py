"""
Baseline comparison for the paper.
Compares four data-sharing approaches on the real African health dataset:
  1. Raw CSV          — full fidelity, zero privacy (the status quo)
  2. Gaussian noise   — add noise to mask values (naive anonymisation)
  3. Aggregates only  — share only mean/std per country (common practice)
  4. NeuroData no DP  — our approach, no privacy guarantee
  5. NeuroData DP ε=5 — our approach, formal (ε,δ)-DP guarantee
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
from neurodata.fidelity import evaluate

np.random.seed(42)
df = pd.read_csv(ROOT / 'data' / 'africa_health.csv')
num_cols = df.select_dtypes(include='number').columns.tolist()
cat_cols  = df.select_dtypes(exclude='number').columns.tolist()

# ── Baseline 1: Raw CSV (perfect ceiling) ─────────────────────────────────────
raw_report = evaluate(df, df)
print(f"Raw CSV:              {raw_report['overall_score']:.1f}/100  (privacy: none)")

# ── Baseline 2: Gaussian noise anonymisation ──────────────────────────────────
df_noisy = df.copy()
for col in num_cols:
    sigma = df[col].std() * 0.3   # 30% noise — typical naive approach
    df_noisy[col] = df[col] + np.random.normal(0, sigma, len(df))
noisy_report = evaluate(df, df_noisy)
print(f"Gaussian noise:       {noisy_report['overall_score']:.1f}/100  (privacy: weak)")

# ── Baseline 3: Aggregates only (mean per country per decade) ─────────────────
df_temp = df.copy()
df['decade'] = (df['year'] // 10) * 10
agg = df.groupby(['country','iso3','decade'])[num_cols].mean().reset_index()
# expand: for each country-decade row, repeat the aggregate for each year in that decade
expanded = []
for _, row in agg.iterrows():
    decade = int(row['decade'])
    for yr in range(decade, decade + 10):
        r = {col: row[col] for col in num_cols}
        r['country'] = row['country']
        r['iso3'] = row['iso3']
        r['year'] = yr
        expanded.append(r)
df_agg = pd.DataFrame(expanded)[df_temp.columns]
df_agg = df_agg[df_agg['year'].isin(df['year'].unique())].head(len(df_temp)).reset_index(drop=True)
df_temp = df.drop(columns='decade', errors='ignore')
agg_report = evaluate(df_temp, df_agg)
print(f"Aggregates only:      {agg_report['overall_score']:.1f}/100  (privacy: moderate)")

# ── Our approach: NeuroData no DP ─────────────────────────────────────────────
print("Training NeuroData (no DP)...")
neurodata.fit(df_temp, path=str(OUT / 'africa_nodp.kess'), epochs=800, verbose=False)
data_nodp = neurodata.load(str(OUT / 'africa_nodp.kess'))
nodp_report = evaluate(df_temp, data_nodp.to_dataframe())
print(f"NeuroData (no DP):    {nodp_report['overall_score']:.1f}/100  (privacy: partial)")

# ── Our approach: NeuroData DP ε=5 ───────────────────────────────────────────
print("Training NeuroData DP ε=5 (more epochs)...")
neurodata.fit(df_temp, path=str(OUT / 'africa_dp5_long.kess'),
              epochs=1500, epsilon=5.0, verbose=False)
data_dp5 = neurodata.load(str(OUT / 'africa_dp5_long.kess'))
dp5_report = evaluate(df_temp, data_dp5.to_dataframe())
print(f"NeuroData (DP ε=5):   {dp5_report['overall_score']:.1f}/100  (privacy: formal DP)")

# ── Summary table ─────────────────────────────────────────────────────────────
methods = [
    ('Raw CSV',             raw_report['overall_score'],   'None',       '#c0392b'),
    ('Gaussian noise',      noisy_report['overall_score'], 'Weak',       '#e67e22'),
    ('Aggregates only',     agg_report['overall_score'],   'Moderate',   '#f1c40f'),
    ('NeuroData (no DP)',   nodp_report['overall_score'],  'Partial',    '#2980b9'),
    ('NeuroData (DP ε=5)', dp5_report['overall_score'],   'Formal (DP)','#27ae60'),
]

print("\n=== Baseline Comparison Summary ===")
print(f"{'Method':<25} {'Fidelity':>10} {'Privacy':>12}")
print("-" * 50)
for name, score, privacy, _ in methods:
    print(f"{name:<25} {score:>9.1f}%  {privacy:>12}")

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('NeuroData vs Baselines: Fidelity–Privacy Tradeoff\n'
             'Real African Health Surveillance Data (25 Countries, 2000–2022)',
             fontsize=13, fontweight='bold')

# Left: fidelity bar chart
ax = axes[0]
names   = [m[0] for m in methods]
scores  = [m[1] for m in methods]
colors  = [m[3] for m in methods]
x = range(len(methods))
bars = ax.bar(x, scores, color=colors, edgecolor='white', width=0.6)
ax.set_xticks(x)
ax.set_xticklabels(names, rotation=18, ha='right', fontsize=10)
ax.set_ylabel('Fidelity Score (0–100)', fontsize=11)
ax.set_title('Statistical Fidelity by Method', fontsize=12)
ax.set_ylim(0, 115)
for bar, s in zip(bars, scores):
    ax.text(bar.get_x()+bar.get_width()/2, s+1, f'{s:.1f}', ha='center', fontsize=10, fontweight='bold')

# Right: scatter — fidelity vs privacy strength
ax2 = axes[1]
privacy_rank = {'None': 0, 'Weak': 1, 'Moderate': 2, 'Partial': 3, 'Formal (DP)': 4}
xlabels = ['None', 'Weak', 'Moderate', 'Partial', 'Formal\n(DP)']

for name, score, privacy, color in methods:
    px = privacy_rank[privacy]
    ax2.scatter(px, score, s=220, color=color, zorder=5, edgecolors='white', linewidth=1.5)
    ax2.annotate(name, (px, score), textcoords='offset points',
                 xytext=(8, 0), fontsize=9, color=color, fontweight='bold')

ax2.set_xticks(list(privacy_rank.values()))
ax2.set_xticklabels(xlabels, fontsize=10)
ax2.set_xlabel('Privacy Strength', fontsize=11)
ax2.set_ylabel('Fidelity Score (0–100)', fontsize=11)
ax2.set_title('Fidelity vs Privacy: Operating Points', fontsize=12)
ax2.set_xlim(-0.5, 4.8)
ax2.set_ylim(30, 115)
ax2.grid(True, alpha=0.3)
# draw Pareto frontier arrow
ax2.annotate('', xy=(3.9, dp5_report['overall_score']+3),
             xytext=(0.1, raw_report['overall_score']-3),
             arrowprops=dict(arrowstyle='->', color='grey', lw=1.5, linestyle='dashed'))
ax2.text(2.0, 85, 'Ideal\nfrontier', color='grey', fontsize=9, ha='center')

plt.tight_layout()
out = str(OUT / 'baseline_comparison.png')
plt.savefig(out, dpi=150, bbox_inches='tight')
print(f"\nFigure saved → {out}")
