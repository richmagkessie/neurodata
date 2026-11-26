"""
Privacy-accuracy tradeoff experiment.
Trains NeuroData at multiple epsilon values and plots fidelity score vs epsilon.
This becomes Figure 1 of the paper.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
OUT = Path(__file__).parent

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import neurodata
from neurodata.fidelity import evaluate

# ── Dataset ───────────────────────────────────────────────────────────────────
np.random.seed(42)
n = 200
df = pd.DataFrame({
    'age':         np.random.randint(1, 80, n),
    'temperature': np.round(np.random.normal(38.2, 1.1, n), 1),
    'parasitemia': np.random.randint(0, 50000, n),
    'region':      np.random.choice(['Ashanti', 'Brong-Ahafo', 'Northern', 'Upper East'], n),
    'outcome':     np.random.choice(['Recovered', 'Admitted', 'Referred'], n),
})

# ── Epsilon values to test ─────────────────────────────────────────────────────
# None = no DP (infinite epsilon, full accuracy ceiling)
epsilon_values = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, None]
labels         = ['0.1', '0.5', '1.0', '2.0', '5.0', '10.0', '∞ (no DP)']

results = []

for eps, label in zip(epsilon_values, labels):
    print(f"\n{'='*50}")
    print(f"Training  ε = {label}")
    print('='*50)

    path = str(OUT / f'model_eps_{label.replace(" ", "_").replace("(", "").replace(")", "")}.pt')

    neurodata.fit(
        df, path=path,
        epochs=600,
        epsilon=eps,
        verbose=False,
    )

    data = neurodata.load(path)
    report = evaluate(df, data.to_dataframe())
    score = report['overall_score']

    # per-column scores
    col_scores = {col: report['columns'][col]['col_score'] for col in report['columns']}

    results.append({
        'epsilon': eps if eps is not None else float('inf'),
        'label': label,
        'overall': score,
        **col_scores,
    })
    print(f"  → Fidelity score: {score:.1f}/100")

# ── Print summary table ───────────────────────────────────────────────────────
print("\n\n=== Privacy-Accuracy Tradeoff Results ===")
print(f"{'ε':<12} {'Overall':>8} {'age':>8} {'temp':>8} {'parasit':>8} {'region':>8} {'outcome':>8}")
print("-" * 65)
for r in results:
    print(f"{r['label']:<12} {r['overall']:>8.1f} {r.get('age',0):>8.1f} "
          f"{r.get('temperature',0):>8.1f} {r.get('parasitemia',0):>8.1f} "
          f"{r.get('region',0):>8.1f} {r.get('outcome',0):>8.1f}")

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('NeuroData: Privacy–Accuracy Tradeoff', fontsize=14, fontweight='bold')

x     = list(range(len(results)))
xlabs = [r['label'] for r in results]

# Left: overall fidelity score
ax1 = axes[0]
scores = [r['overall'] for r in results]
bars = ax1.bar(x, scores, color=['#c0392b']*6 + ['#27ae60'], edgecolor='white', width=0.6)
ax1.axhline(scores[-1], color='#27ae60', linestyle='--', linewidth=1.2, label='No-DP ceiling')
ax1.set_xticks(x)
ax1.set_xticklabels(xlabs, fontsize=10)
ax1.set_xlabel('Privacy Budget  ε  (smaller = stronger privacy)', fontsize=11)
ax1.set_ylabel('Fidelity Score  (0–100)', fontsize=11)
ax1.set_title('Overall Fidelity vs Privacy Budget', fontsize=12)
ax1.set_ylim(0, 105)
ax1.legend(fontsize=9)
for bar, score in zip(bars, scores):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
             f'{score:.1f}', ha='center', va='bottom', fontsize=9)

# Right: per-column breakdown
ax2 = axes[1]
cols_to_plot = ['age', 'temperature', 'parasitemia', 'region', 'outcome']
colors = ['#2980b9', '#8e44ad', '#e67e22', '#16a085', '#c0392b']

width = 0.15
for i, (col, color) in enumerate(zip(cols_to_plot, colors)):
    col_scores = [r.get(col, 0) for r in results]
    offsets = [xi + (i - 2) * width for xi in x]
    ax2.bar(offsets, col_scores, width=width, label=col, color=color, alpha=0.85)

ax2.set_xticks(x)
ax2.set_xticklabels(xlabs, fontsize=10)
ax2.set_xlabel('Privacy Budget  ε', fontsize=11)
ax2.set_ylabel('Column Fidelity Score  (0–100)', fontsize=11)
ax2.set_title('Per-Column Fidelity vs Privacy Budget', fontsize=12)
ax2.set_ylim(0, 115)
ax2.legend(fontsize=9, ncol=2)

plt.tight_layout()
out_path = str(OUT / 'tradeoff_figure.png')
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"\nFigure saved → {out_path}")
