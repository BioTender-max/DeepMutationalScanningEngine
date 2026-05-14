
import numpy as np
np.random.seed(42)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
import os, shutil

OUT_DIR = '/mnt/shared-workspace/shared'
os.makedirs(OUT_DIR, exist_ok=True)

N_POSITIONS = 200
AA_LIST = list('ACDEFGHIKLMNPQRSTVWY')
N_AA = 20

# ── Fitness landscape ─────────────────────────────────────────────────────────
# fitness_matrix: positions x AA
fitness_matrix = np.random.normal(0, 0.8, (N_POSITIONS, N_AA))
# Functional sites: positions 40-60, 100-120 are constrained
for pos in range(40, 60):
    fitness_matrix[pos, :] -= np.random.uniform(0.5, 2.0, N_AA)
    fitness_matrix[pos, np.random.randint(0, N_AA)] += 1.5  # preferred AA
for pos in range(100, 120):
    fitness_matrix[pos, :] -= np.random.uniform(0.3, 1.5, N_AA)
# Wild-type diagonal (WT = 0 by definition)
wt_aa = np.random.randint(0, N_AA, N_POSITIONS)
for i, wt in enumerate(wt_aa):
    fitness_matrix[i, wt] = 0.0

# ── Fitness distribution ──────────────────────────────────────────────────────
all_fitness = fitness_matrix.flatten()
beneficial = (all_fitness > 0.5).sum()
neutral = ((all_fitness >= -0.5) & (all_fitness <= 0.5)).sum()
deleterious = (all_fitness < -0.5).sum()

# ── Epistasis detection ───────────────────────────────────────────────────────
n_double = 500
pos1 = np.random.randint(0, N_POSITIONS, n_double)
pos2 = np.random.randint(0, N_POSITIONS, n_double)
aa1 = np.random.randint(0, N_AA, n_double)
aa2 = np.random.randint(0, N_AA, n_double)
f1 = fitness_matrix[pos1, aa1]
f2 = fitness_matrix[pos2, aa2]
additive_expected = f1 + f2
double_fitness = additive_expected + np.random.normal(0, 0.5, n_double)
epistasis = double_fitness - additive_expected
epi_r, epi_p = stats.pearsonr(additive_expected, double_fitness)

# ── Conservation-fitness correlation ──────────────────────────────────────────
conservation = np.random.beta(2, 2, N_POSITIONS)
# Functional sites are more conserved
conservation[40:60] = np.random.beta(5, 1, 20)
conservation[100:120] = np.random.beta(5, 1, 20)
mean_fitness_per_pos = fitness_matrix.mean(axis=1)
cons_r, cons_p = stats.pearsonr(conservation, mean_fitness_per_pos)

# ── Functional sites ──────────────────────────────────────────────────────────
functional_sites = np.where(mean_fitness_per_pos < -1.0)[0]

# ── Secondary structure ───────────────────────────────────────────────────────
ss_labels = np.random.choice(['Helix', 'Sheet', 'Loop'], N_POSITIONS, p=[0.4, 0.3, 0.3])
ss_fitness = {ss: fitness_matrix[ss_labels == ss].mean() for ss in ['Helix', 'Sheet', 'Loop']}

# ── Mutational tolerance ──────────────────────────────────────────────────────
tolerance = fitness_matrix.std(axis=1)  # high std = tolerant

# ── Dashboard ─────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 3, figsize=(20, 15))
fig.patch.set_facecolor('#0d1117')
fig.suptitle('Deep Mutational Scanning Engine — 200 Positions × 20 AA = 4000 Variants',
             color='white', fontsize=16, fontweight='bold', y=0.98)

def style_ax(ax, title, xlabel='', ylabel=''):
    ax.set_facecolor('#161b22')
    ax.set_title(title, color='white', fontsize=11, fontweight='bold', pad=8)
    ax.set_xlabel(xlabel, color='#8b949e', fontsize=9)
    ax.set_ylabel(ylabel, color='#8b949e', fontsize=9)
    ax.tick_params(colors='#8b949e', labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor('#30363d')

# Panel 1: Fitness heatmap
ax = axes[0, 0]
im = ax.imshow(fitness_matrix.T, cmap='RdBu_r', aspect='auto', vmin=-3, vmax=3)
ax.set_yticks(range(N_AA))
ax.set_yticklabels(AA_LIST, fontsize=6, color='#8b949e')
cb = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
cb.set_label('Fitness (log2)', color='#8b949e', fontsize=8)
plt.setp(cb.ax.yaxis.get_ticklabels(), color='#8b949e', fontsize=7)
style_ax(ax, 'Fitness Landscape Heatmap', 'Position', 'Amino Acid')

# Panel 2: Fitness distribution
ax = axes[0, 1]
ax.hist(all_fitness, bins=60, color='#58a6ff', alpha=0.85, edgecolor='#0d1117', linewidth=0.3)
ax.axvline(0, color='white', lw=1, linestyle=':')
ax.axvline(0.5, color='#3fb950', lw=1.5, linestyle='--', label=f'Beneficial: {beneficial}')
ax.axvline(-0.5, color='#f78166', lw=1.5, linestyle='--', label=f'Deleterious: {deleterious}')
ax.legend(fontsize=8, facecolor='#21262d', labelcolor='white')
style_ax(ax, 'Fitness Score Distribution', 'Fitness (log2 enrichment)', 'Count')

# Panel 3: Epistasis scatter
ax = axes[0, 2]
sc = ax.scatter(additive_expected, double_fitness, c=np.abs(epistasis), cmap='plasma',
                alpha=0.5, s=15, edgecolors='none')
lims = [min(additive_expected.min(), double_fitness.min()),
        max(additive_expected.max(), double_fitness.max())]
ax.plot(lims, lims, 'r--', lw=1.5, label='Additive')
ax.legend(fontsize=8, facecolor='#21262d', labelcolor='white')
cb2 = fig.colorbar(sc, ax=ax, shrink=0.8, pad=0.02)
cb2.set_label('|Epistasis|', color='#8b949e', fontsize=8)
plt.setp(cb2.ax.yaxis.get_ticklabels(), color='#8b949e', fontsize=7)
style_ax(ax, f'Epistasis: Double vs Additive (r={epi_r:.3f})',
         'Additive Expected Fitness', 'Double Mutant Fitness')

# Panel 4: Conservation-fitness correlation
ax = axes[1, 0]
ax.scatter(conservation, mean_fitness_per_pos, color='#3fb950', alpha=0.5, s=15, edgecolors='none')
m, b, r, p, _ = stats.linregress(conservation, mean_fitness_per_pos)
xfit = np.linspace(conservation.min(), conservation.max(), 100)
ax.plot(xfit, m*xfit+b, color='#f78166', lw=2, label=f'r={r:.3f}')
ax.legend(fontsize=8, facecolor='#21262d', labelcolor='white')
style_ax(ax, 'Conservation vs Mean Fitness', 'Conservation Score', 'Mean Fitness')

# Panel 5: Functional site map
ax = axes[1, 1]
colors_pos = np.where(mean_fitness_per_pos < -1.0, '#f78166', '#58a6ff')
ax.bar(np.arange(N_POSITIONS), mean_fitness_per_pos, color=colors_pos, width=1.0, edgecolor='none')
ax.axhline(-1.0, color='#ffa657', lw=1.5, linestyle='--', label=f'Functional threshold ({len(functional_sites)} sites)')
ax.legend(fontsize=8, facecolor='#21262d', labelcolor='white')
style_ax(ax, 'Functional Site Map', 'Position', 'Mean Fitness')

# Panel 6: AA preference (simplified logo)
ax = axes[1, 2]
aa_mean_fitness = fitness_matrix.mean(axis=0)
colors_aa = ['#3fb950' if f > 0 else '#f78166' for f in aa_mean_fitness]
bars = ax.bar(AA_LIST, aa_mean_fitness, color=colors_aa, edgecolor='#0d1117', linewidth=0.5)
ax.axhline(0, color='white', lw=1, linestyle=':')
style_ax(ax, 'Mean Fitness by Amino Acid', 'Amino Acid', 'Mean Fitness')
ax.tick_params(axis='x', labelsize=8)

# Panel 7: Fitness by secondary structure
ax = axes[2, 0]
ss_types = ['Helix', 'Sheet', 'Loop']
ss_vals = [ss_fitness[ss] for ss in ss_types]
ss_colors = ['#58a6ff', '#3fb950', '#ffa657']
bars2 = ax.bar(ss_types, ss_vals, color=ss_colors, edgecolor='#0d1117', linewidth=0.5)
for bar, val in zip(bars2, ss_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01 if val >= 0 else bar.get_height() - 0.05,
            f'{val:.3f}', ha='center', va='bottom' if val >= 0 else 'top', color='white', fontsize=9)
ax.axhline(0, color='white', lw=1, linestyle=':')
style_ax(ax, 'Mean Fitness by Secondary Structure', 'Secondary Structure', 'Mean Fitness')

# Panel 8: Mutational tolerance
ax = axes[2, 1]
ax.bar(np.arange(N_POSITIONS), tolerance, color='#d2a8ff', width=1.0, edgecolor='none', alpha=0.8)
ax.axhline(tolerance.mean(), color='#f78166', lw=2, linestyle='--',
           label=f'Mean={tolerance.mean():.3f}')
ax.legend(fontsize=8, facecolor='#21262d', labelcolor='white')
style_ax(ax, 'Mutational Tolerance per Position', 'Position', 'Fitness Std Dev')

# Panel 9: Summary
ax = axes[2, 2]
ax.set_facecolor('#161b22')
ax.axis('off')
summary_text = (
    f"  Deep Mutational Scanning Summary\n"
    f"  {'─'*32}\n"
    f"  Positions:             {N_POSITIONS}\n"
    f"  Amino acids:           {N_AA}\n"
    f"  Total variants:        {N_POSITIONS * N_AA}\n"
    f"  Beneficial (>0.5):     {beneficial} ({100*beneficial/(N_POSITIONS*N_AA):.1f}%)\n"
    f"  Neutral (-0.5–0.5):    {neutral} ({100*neutral/(N_POSITIONS*N_AA):.1f}%)\n"
    f"  Deleterious (<-0.5):   {deleterious} ({100*deleterious/(N_POSITIONS*N_AA):.1f}%)\n"
    f"  Functional sites:      {len(functional_sites)}\n"
    f"  Epistasis r:           {epi_r:.3f}\n"
    f"  Conservation-fitness r:{cons_r:.3f}\n"
    f"  Mean tolerance:        {tolerance.mean():.3f}\n"
    f"  SS fitness (Helix):    {ss_fitness['Helix']:.3f}\n"
    f"  SS fitness (Sheet):    {ss_fitness['Sheet']:.3f}\n"
    f"  SS fitness (Loop):     {ss_fitness['Loop']:.3f}\n"
)
ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=9,
        verticalalignment='top', fontfamily='monospace',
        color='#e6edf3', bbox=dict(boxstyle='round', facecolor='#21262d', alpha=0.8))
ax.set_title('Summary Statistics', color='white', fontsize=11, fontweight='bold', pad=8)

plt.tight_layout(rect=[0, 0, 1, 0.97])
out_png = f'{OUT_DIR}/deep_mutational_scanning_engine_dashboard.png'
plt.savefig(out_png, dpi=100, bbox_inches='tight', facecolor='#0d1117')
plt.close()
print(f"Saved: {out_png}")

print("\n=== DeepMutationalScanningEngine Key Results ===")
print(f"N positions: {N_POSITIONS}, N AA: {N_AA}, Total variants: {N_POSITIONS*N_AA}")
print(f"Beneficial (>0.5): {beneficial} ({100*beneficial/(N_POSITIONS*N_AA):.1f}%)")
print(f"Neutral (-0.5 to 0.5): {neutral} ({100*neutral/(N_POSITIONS*N_AA):.1f}%)")
print(f"Deleterious (<-0.5): {deleterious} ({100*deleterious/(N_POSITIONS*N_AA):.1f}%)")
print(f"Functional sites: {len(functional_sites)}")
print(f"Epistasis correlation: r={epi_r:.4f}, p={epi_p:.4e}")
print(f"Conservation-fitness correlation: r={cons_r:.4f}, p={cons_p:.4e}")
print(f"Mean mutational tolerance: {tolerance.mean():.4f}")
