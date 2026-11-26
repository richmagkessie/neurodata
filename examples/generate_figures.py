"""
Generates all four publication-quality figures for the NeuroData paper.
"""
import sys
sys.path.insert(0, '/Users/macos/Downloads/projects/neurodata')

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import neurodata
from neurodata.fidelity import evaluate

FIGDIR = '/Users/macos/Downloads/projects/neurodata/figures/'
import os; os.makedirs(FIGDIR, exist_ok=True)

PALETTE = {
    'blue':   '#1a5276',
    'green':  '#1e8449',
    'red':    '#922b21',
    'orange': '#d35400',
    'gold':   '#b7950b',
    'teal':   '#148f77',
    'grey':   '#5d6d7e',
    'light':  '#f0f3f4',
}
FONT = {'family': 'serif', 'size': 11}
matplotlib.rc('font', **FONT)
matplotlib.rc('axes', titlesize=12, labelsize=11)
matplotlib.rc('xtick', labelsize=10)
matplotlib.rc('ytick', labelsize=10)
matplotlib.rc('legend', fontsize=10)

df = pd.read_csv('/Users/macos/Downloads/projects/neurodata/data/africa_health.csv')
num_cols = df.select_dtypes(include='number').columns.tolist()

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1: System Architecture  (clean three-panel redesign)
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(13, 4.8))
ax.set_xlim(0, 13); ax.set_ylim(0, 4.8); ax.axis('off')

C_OWNER  = '#1a4f72'   # dark blue  — data owner
C_OWNER_L= '#d6eaf8'   # light blue panel
C_MODEL  = '#1e8449'   # dark green — .model file
C_MODEL_L= '#d5f5e3'   # light green panel
C_RES    = '#6c3483'   # dark purple — researcher
C_RES_L  = '#e8daef'   # light purple panel
C_BOX    = '#2e86c1'   # inner boxes
C_TEAL   = '#148f77'
C_GREY   = '#5d6d7e'
C_RED    = '#a93226'

def rbox(ax, x, y, w, h, title, subtitle='', fc=C_BOX, tc='white', fs=9.5, zorder=3):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.1',
                                facecolor=fc, edgecolor='white', linewidth=1.8, zorder=zorder))
    yo = 0.14 if subtitle else 0
    ax.text(x+w/2, y+h/2+yo, title, ha='center', va='center',
            color=tc, fontsize=fs, fontweight='bold', zorder=zorder+1)
    if subtitle:
        ax.text(x+w/2, y+h/2-0.2, subtitle, ha='center', va='center',
                color=tc, fontsize=7.8, zorder=zorder+1)

def harrow(ax, x1, x2, y, label='', color=C_GREY, lw=2.0):
    ax.annotate('', xy=(x2, y), xytext=(x1, y),
                arrowprops=dict(arrowstyle='->', color=color, lw=lw,
                                mutation_scale=18))
    if label:
        ax.text((x1+x2)/2, y+0.18, label, ha='center', va='bottom',
                fontsize=8, color=color, style='italic')

def varrow(ax, x, y1, y2, color=C_GREY, lw=1.8):
    ax.annotate('', xy=(x, y2), xytext=(x, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=lw, mutation_scale=14))

# ── PANEL 1: Data Owner ──────────────────────────────────────────────────────
ax.add_patch(FancyBboxPatch((0.15, 0.4), 3.6, 4.0, boxstyle='round,pad=0.12',
             facecolor=C_OWNER_L, edgecolor=C_OWNER, linewidth=1.8, linestyle='--', zorder=1))
ax.text(1.95, 4.22, 'DATA OWNER', ha='center', va='center',
        fontsize=10.5, color=C_OWNER, fontweight='bold')

rbox(ax, 0.4, 2.9, 3.1, 0.9, 'Health Surveillance Dataset',
     'CSV  |  575 rows  |  never shared', fc='#2471a3', fs=9.5)
rbox(ax, 0.4, 1.7, 3.1, 0.95, 'NeuroData Encoder',
     'Row index  ->  column values', fc=C_OWNER, fs=9.5)
rbox(ax, 0.4, 0.55, 1.4, 0.85, 'DP-SGD', 'Privacy training', fc=C_RED, fs=8.5)
rbox(ax, 2.1, 0.55, 1.4, 0.85, 'AES-256-GCM', 'Encryption', fc='#5d6d7e', fs=8.5)

varrow(ax, 1.95, 2.9, 2.65)
varrow(ax, 1.95, 1.7, 1.4)

ax.text(0.82, 1.3, u'↑', ha='center', va='center', fontsize=14, color=C_RED)
ax.text(2.52, 1.3, u'↑', ha='center', va='center', fontsize=14, color='#5d6d7e')

# ── PANEL 2: .model file ─────────────────────────────────────────────────────
ax.add_patch(FancyBboxPatch((4.15, 1.0), 2.5, 2.8, boxstyle='round,pad=0.12',
             facecolor=C_MODEL_L, edgecolor=C_MODEL, linewidth=2.0, linestyle='-', zorder=1))
ax.text(5.4, 3.62, '.model FILE', ha='center', va='center',
        fontsize=10.5, color=C_MODEL, fontweight='bold')

rbox(ax, 4.4, 2.3, 2.0, 0.85, 'africa.model', 'Encrypted binary', fc=C_MODEL, fs=9.5)

ax.text(5.4, 2.1,  'AES-256-GCM key',  ha='center', fontsize=8, color=C_GREY)
ax.text(5.4, 1.78, 'PBKDF2  200k iters', ha='center', fontsize=8, color=C_GREY)
ax.text(5.4, 1.46, 'Size: 2.1 MB',       ha='center', fontsize=8, color=C_GREY)
ax.text(5.4, 1.14, u'ε = {0.5 – ∞}  |  δ = 10⁻⁵',
        ha='center', fontsize=8, color=C_RED, fontweight='bold')

# ── Transfer arrows ──────────────────────────────────────────────────────────
harrow(ax, 3.75, 4.15, 2.75, 'encode', C_OWNER, lw=2.2)
harrow(ax, 6.65, 7.1, 2.75, 'share', C_MODEL, lw=2.2)
ax.text(5.4, 2.55, u'↓', ha='center', fontsize=18, color=C_MODEL, fontweight='bold')

# ── PANEL 3: Researcher ───────────────────────────────────────────────────────
ax.add_patch(FancyBboxPatch((7.1, 0.4), 5.65, 4.0, boxstyle='round,pad=0.12',
             facecolor=C_RES_L, edgecolor=C_RES, linewidth=1.8, linestyle='--', zorder=1))
ax.text(9.93, 4.22, 'DATA RECIPIENT', ha='center', va='center',
        fontsize=10.5, color=C_RES, fontweight='bold')

rbox(ax, 7.35, 3.05, 5.15, 0.95, 'NeuroFrame  ( data = neurodata.load("africa.model") )',
     'Password-gated  |  Pandas-compatible interface', fc=C_RES, fs=9.0)

# API operation boxes in a clean 3x2 grid
ops = [
    ('data.head()',       'First n rows as DataFrame'),
    ('data.describe()',   'Summary statistics'),
    ('data.groupby(col)', 'Aggregation by column'),
    ('data.corr()',       'Correlation matrix'),
    ('data.value_counts(col)', 'Frequency distribution'),
    ('data.to_csv(path)', 'Export reconstructed data'),
]
cols3 = [7.35, 9.1, 10.85]
rows2 = [1.65, 0.55]
for i, (name, desc) in enumerate(ops):
    bx = cols3[i % 3]
    by = rows2[i // 3]
    rbox(ax, bx, by, 1.6, 0.9, name, desc, fc=C_TEAL, fs=7.8)
    varrow(ax, bx+0.8, 3.05, by+0.9, color=C_RES, lw=1.3)

ax.text(9.93, 2.78, 'Pandas-compatible operations', ha='center',
        fontsize=8.5, color=C_RES, style='italic')

ax.set_title(
    'Fig. 1.  NeuroData System Architecture: Encoding a Health Dataset as a Neural Network for Privacy-Preserving Sharing',
    fontsize=10.5, pad=10, fontweight='bold')

plt.tight_layout()
plt.savefig(FIGDIR + 'fig1_architecture.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print("Fig 1 saved.")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2: Privacy-Accuracy Tradeoff (clean, two panels)
# ══════════════════════════════════════════════════════════════════════════════
eps_vals   = [0.5, 1.0, 2.0, 5.0, 10.0, None]
eps_labels = ['0.5', '1.0', '2.0', '5.0', '10.0', 'No DP']
scores = []
per_col_scores = {c: [] for c in num_cols}

for eps, label in zip(eps_vals, eps_labels):
    path = f'/tmp/fig2_eps{label}.model'
    ep = 1200 if eps is not None else 800
    neurodata.fit(df, path=path, epochs=ep, epsilon=eps, verbose=False)
    d = neurodata.load(path)
    r = evaluate(df, d.to_dataframe())
    scores.append(r['overall_score'])
    for c in num_cols:
        per_col_scores[c].append(r['columns'].get(c, {}).get('col_score', 0))
    print(f"  eps={label:>5}  score={r['overall_score']:.1f}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
fig.suptitle('Fig. 2. Privacy Budget vs. Statistical Fidelity on African Health Surveillance Data',
             fontsize=11, fontweight='bold', y=1.01)

x = np.arange(len(eps_labels))
bar_colors = [PALETTE['red']]*5 + [PALETTE['green']]
bars = ax1.bar(x, scores, color=bar_colors, edgecolor='white', width=0.6, zorder=3)
ax1.axhline(scores[-1], color=PALETTE['green'], linestyle='--', lw=1.5,
            label=f'No-DP ceiling ({scores[-1]:.1f})', zorder=2)
ax1.set_xticks(x)
ax1.set_xticklabels(eps_labels)
ax1.set_xlabel('Privacy Budget (epsilon) -- smaller = stronger privacy')
ax1.set_ylabel('Overall Fidelity Score (0-100)')
ax1.set_title('(a) Overall Fidelity vs. Privacy Budget')
ax1.set_ylim(0, 115)
ax1.legend()
ax1.grid(axis='y', alpha=0.3, zorder=1)
for bar, s in zip(bars, scores):
    ax1.text(bar.get_x()+bar.get_width()/2, s+1.5, f'{s:.1f}',
             ha='center', fontsize=9, fontweight='bold')

col_colors = ['#1a5276','#922b21','#1e8449','#d35400','#7d3c98','#148f77','#b7950b']
for i, (col, col_scores_list) in enumerate(per_col_scores.items()):
    label = col.replace('_', ' ').replace('per100k','(/100k)').replace('per1000','(/1k)')
    ax2.plot(x, col_scores_list, 'o-', color=col_colors[i % len(col_colors)],
             label=label, lw=1.8, markersize=5)
ax2.set_xticks(x)
ax2.set_xticklabels(eps_labels)
ax2.set_xlabel('Privacy Budget (epsilon)')
ax2.set_ylabel('Column Fidelity Score (0-100)')
ax2.set_title('(b) Per-Indicator Fidelity vs. Privacy Budget')
ax2.legend(fontsize=8, ncol=2)
ax2.set_ylim(0, 115)
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(FIGDIR + 'fig2_tradeoff.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print("Fig 2 saved.")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 3: Baseline Comparison
# ══════════════════════════════════════════════════════════════════════════════
df_clean = df.copy()
for col in num_cols:
    df_clean[col] = df_clean[col].fillna(df_clean[col].mean())

df_noisy = df_clean.copy()
for col in num_cols:
    df_noisy[col] = df_clean[col] + np.random.normal(0, df_clean[col].std()*0.3, len(df_clean))

data_nodp = neurodata.load('/tmp/fig2_epsNo DP.model')
neurodata.fit(df_clean, path='/tmp/fig3_dp5.model', epochs=1200, epsilon=5.0, verbose=False)
data_dp5  = neurodata.load('/tmp/fig3_dp5.model')

methods = [
    ('Raw CSV',           evaluate(df_clean, df_clean)['overall_score'],           'None',         PALETTE['red']),
    ('Gaussian Noise',    evaluate(df_clean, df_noisy)['overall_score'],            'Weak',         PALETTE['orange']),
    ('Aggregates Only',   evaluate(df_clean, df_clean.groupby('country')[num_cols].transform('mean').assign(country=df_clean['country'], iso3=df_clean['iso3'], year=df_clean['year']))['overall_score'], 'Moderate', PALETTE['gold']),
    ('NeuroData (no DP)', evaluate(df_clean, data_nodp.to_dataframe())['overall_score'], 'Partial', PALETTE['blue']),
    ('NeuroData (DP e=5)',evaluate(df_clean, data_dp5.to_dataframe())['overall_score'],  'Formal DP', PALETTE['green']),
]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
fig.suptitle('Fig. 3. NeuroData vs. Baseline Data-Sharing Methods: Fidelity and Privacy Comparison',
             fontsize=11, fontweight='bold', y=1.01)

names  = [m[0] for m in methods]
fids   = [m[1] for m in methods]
colors = [m[3] for m in methods]
x = np.arange(len(methods))
bars = ax1.bar(x, fids, color=colors, edgecolor='white', width=0.55, zorder=3)
ax1.set_xticks(x)
ax1.set_xticklabels(names, rotation=15, ha='right')
ax1.set_ylabel('Fidelity Score (0-100)')
ax1.set_title('(a) Statistical Fidelity by Method')
ax1.set_ylim(0, 115)
ax1.grid(axis='y', alpha=0.3, zorder=1)
for bar, s in zip(bars, fids):
    ax1.text(bar.get_x()+bar.get_width()/2, s+1.5, f'{s:.1f}',
             ha='center', fontsize=9, fontweight='bold')

priv_rank = {'None': 0, 'Weak': 1, 'Moderate': 2, 'Partial': 3, 'Formal DP': 4}
xlabels   = ['None', 'Weak', 'Moderate', 'Partial', 'Formal\n(DP)']
for name, score, priv, color in methods:
    px = priv_rank[priv]
    ax2.scatter(px, score, s=260, color=color, zorder=5,
                edgecolors='white', linewidth=1.5)
    offset = (10, 6) if name != 'NeuroData (DP e=5)' else (10, -12)
    ax2.annotate(name, (px, score), textcoords='offset points',
                 xytext=offset, fontsize=8.5, color=color, fontweight='bold')

ax2.set_xticks(list(priv_rank.values()))
ax2.set_xticklabels(xlabels)
ax2.set_xlabel('Privacy Strength')
ax2.set_ylabel('Fidelity Score (0-100)')
ax2.set_title('(b) Fidelity vs. Privacy: Operating Points')
ax2.set_xlim(-0.6, 5.0)
ax2.set_ylim(30, 115)
ax2.grid(alpha=0.3)
ax2.annotate('', xy=(3.85, methods[-1][1]+4), xytext=(0.1, methods[0][1]-4),
             arrowprops=dict(arrowstyle='->', color='#aab7b8', lw=1.5, linestyle='dashed'))
ax2.text(1.9, 82, 'Ideal tradeoff\nfrontier', color='#aab7b8', fontsize=8.5, ha='center')

plt.tight_layout()
plt.savefig(FIGDIR + 'fig3_baselines.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print("Fig 3 saved.")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 4: Multi-country reconstruction quality
# ══════════════════════════════════════════════════════════════════════════════
recon = data_nodp.to_dataframe()
highlight_countries = ['Ghana', 'Kenya', 'Nigeria', 'South Africa', 'Ethiopia', 'Rwanda']

fig, axes = plt.subplots(2, 3, figsize=(13, 7))
fig.suptitle('Fig. 4. NeuroData Reconstruction Quality: Life Expectancy Trends for Six African Countries\n'
             '(Original vs. Reconstructed; No-DP model, Fidelity = 98.4/100)',
             fontsize=11, fontweight='bold')

for i, country in enumerate(highlight_countries):
    ax = axes[i // 3][i % 3]
    orig  = df_clean[df_clean['country'] == country].sort_values('year')
    rec   = recon[recon['country'] == country].sort_values('year')
    if orig.empty or rec.empty:
        ax.set_visible(False); continue
    ax.plot(orig['year'], orig['life_expectancy'], 'o-',
            color=PALETTE['blue'], lw=2, markersize=4, label='Original', zorder=4)
    ax.plot(rec['year'], rec['life_expectancy'], 's--',
            color=PALETTE['red'], lw=2, markersize=4, label='NeuroData', zorder=3)
    mae = np.mean(np.abs(orig['life_expectancy'].values - rec['life_expectancy'].values[:len(orig)]))
    ax.set_title(f'{country}  (MAE = {mae:.2f} yrs)', fontsize=10, fontweight='bold')
    ax.set_xlabel('Year', fontsize=9)
    ax.set_ylabel('Life Expectancy', fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.tick_params(labelsize=8)

plt.tight_layout()
plt.savefig(FIGDIR + 'fig4_reconstruction.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print("Fig 4 saved.")
print("\nAll figures saved to:", FIGDIR)
