"""Figures for the EchoNet-Dynamic reproduction, built from the saved test results.

Reads results/<run>/seed_<seed>/predictions.csv, results/<run>/summary.csv and
the fine-tuning histories (whichever exist) and writes PNGs to results/figures/.
No GPU needed.
"""
import argparse
import os
import re

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from matplotlib.lines import Line2D  # noqa: E402
from sklearn import metrics  # noqa: E402

import common  # noqa: E402

# Fixed colors per entity, so each one looks the same in every figure
COLORS = {
    'zeroshot': '#2a78d6',       # blue
    'finetune': '#eb6834',       # orange
    'zeroshot_head': '#1baf7a',  # aqua: zero-shot LV systolic function head
}
LABELS = {
    'zeroshot': 'Zero-shot',
    'finetune': 'Fine-tuned',
    'zeroshot_head': 'Zero-shot, LV function head',
}
INK = '#0b0b0b'
INK_SECONDARY = '#52514e'
INK_MUTED = '#898781'
GRID = '#e1e0d9'
AXIS = '#c3c2b7'

# Memory of the GPU the runs were measured on (RTX 3070 Laptop), the limit on the memory panel
GPU_MEMORY_GB = 8.0

# True-EF ranges for the error breakdown
EF_BINS = [0, 30, 40, 50, 60, 70, 101]
EF_BIN_LABELS = ['<30', '30-40', '40-50', '50-60', '60-70', '>=70']

# Metrics shown in the seed-variability figure: (summary.csv key, title, number format)
SEED_METRICS = [
    ('ef_mae', 'EF MAE (EF points)', '.2f'),
    ('ef_rmse', 'EF RMSE (EF points)', '.2f'),
    ('ef_r2', 'EF R$^2$', '.3f'),
    ('auroc_ef', f'AUROC, EF < {common.LV_DYSFUNCTION_EF}%', '.3f'),
]


# ----------------------------------------------------------------------------
# set_style
# ----------------------------------------------------------------------------
def set_style():
    """Apply one consistent, recessive chart style to every figure."""
    plt.rcParams.update({
        'figure.dpi': 100,
        'savefig.dpi': 300,
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'font.family': 'sans-serif',
        'font.sans-serif': ['Segoe UI', 'DejaVu Sans', 'Arial'],
        'font.size': 10,
        'axes.titlesize': 11,
        'axes.titleweight': 'bold',
        'axes.titlecolor': INK,
        'axes.labelcolor': INK_SECONDARY,
        'axes.edgecolor': AXIS,
        'axes.linewidth': 0.8,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': True,
        'grid.color': GRID,
        'grid.linewidth': 0.6,
        'axes.axisbelow': True,
        'xtick.color': INK_MUTED,
        'ytick.color': INK_MUTED,
        'xtick.labelcolor': INK_SECONDARY,
        'ytick.labelcolor': INK_SECONDARY,
        'legend.frameon': False,
        'legend.fontsize': 9,
        'lines.linewidth': 2,
    })


# ----------------------------------------------------------------------------
# save_figure
# ----------------------------------------------------------------------------
def save_figure(fig, folder, name, note):
    """Add a footnote, save a figure as PNG into folder (created if needed) and close it.

    Args:
        fig: matplotlib Figure.
        folder: output folder.
        name: file name, e.g. 'ef_scatter_test.png'.
        note: footnote naming the data behind the figure (split, seed or seeds).
    """
    fig.text(0.0, -0.02, note, ha='left', va='top', color=INK_MUTED, fontsize=8, transform=fig.transFigure)

    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, name)
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f'   {os.path.relpath(path, common.REPO_DIR)}')


# ----------------------------------------------------------------------------
# load_predictions / load_summaries / load_histories
# ----------------------------------------------------------------------------
def load_predictions(results_dir, seed):
    """Load one seed's saved test predictions for each run that exists.

    Args:
        results_dir: the reproduction/results folder.
        seed: which seed's predictions to plot.

    Returns:
        dict run name ('zeroshot', 'finetune') -> DataFrame.
    """
    runs = {}
    for run in ['zeroshot', 'finetune']:
        path = os.path.join(results_dir, run, f'seed_{seed}', 'predictions.csv')
        if os.path.exists(path):
            runs[run] = pd.read_csv(path)

    return runs


def load_summaries(results_dir):
    """Load the per-seed metrics (summary.csv) of each run that exists.

    Returns:
        dict run name -> DataFrame with seed, metric, label, value, ci_low, ci_high.
    """
    summaries = {}
    for run in ['zeroshot', 'finetune']:
        path = os.path.join(results_dir, run, 'summary.csv')
        if os.path.exists(path):
            summaries[run] = pd.read_csv(path)

    return summaries


def load_histories(results_dir):
    """Load the per-epoch fine-tuning history of every seed.

    Returns:
        dict seed -> DataFrame with epoch, train_loss, val_loss, val_mae.
    """
    histories = {}
    finetune_dir = os.path.join(results_dir, 'finetune')
    if os.path.isdir(finetune_dir):
        for name in sorted(os.listdir(finetune_dir)):
            path = os.path.join(finetune_dir, name, 'history.csv')
            if name.startswith('seed_') and os.path.exists(path):
                histories[int(name.split('_')[1])] = pd.read_csv(path)

    return histories


# ----------------------------------------------------------------------------
# dysfunction_scores
# ----------------------------------------------------------------------------
def dysfunction_scores(runs):
    """Scores for LV dysfunction detection (true EF < threshold), one per entity.

    Lower predicted EF means more likely dysfunction, so predicted-EF scores are
    negated. The zero-shot model also has a dedicated LV systolic function head.

    Returns:
        dict entity -> (labels, scores).
    """
    scores = {}
    for run, df in runs.items():
        labels = (df['EF'].values < common.LV_DYSFUNCTION_EF).astype(int)
        scores[run] = (labels, -df['EF_pred'].values)
        if 'p_lv_dysfunction' in df:
            scores[f'{run}_head'] = (labels, df['p_lv_dysfunction'].values)

    return scores


# ----------------------------------------------------------------------------
# plot_ef_scatter
# ----------------------------------------------------------------------------
def plot_ef_scatter(ax, df, run):
    """Predicted vs. true EF with the identity line and summary metrics."""
    ef_true, ef_pred = df['EF'].values, df['EF_pred'].values
    m = common.regression_metrics(ef_true, ef_pred)

    ax.plot([0, 100], [0, 100], color=INK_MUTED, lw=1, ls='--', zorder=1)
    ax.scatter(ef_true, ef_pred, s=10, color=COLORS[run], alpha=0.35, linewidths=0, zorder=2)
    ax.text(0.04, 0.96, f'MAE {m["MAE"]:.2f}\nRMSE {m["RMSE"]:.2f}\nR$^2$ {m["R2"]:.3f}\nN = {len(df):,}', transform=ax.transAxes, va='top', color=INK_SECONDARY, fontsize=9)

    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_aspect('equal')
    ax.set_xlabel('True EF (%)')
    ax.set_ylabel('Predicted EF (%)')
    ax.set_title(LABELS[run])


# ----------------------------------------------------------------------------
# plot_bland_altman
# ----------------------------------------------------------------------------
def plot_bland_altman(ax, df, run):
    """Bland-Altman plot: difference (predicted - true) against the mean of the two."""
    mean = (df['EF'].values + df['EF_pred'].values) / 2
    diff = df['EF_pred'].values - df['EF'].values
    bias, sd = diff.mean(), diff.std(ddof=1)

    ax.scatter(mean, diff, s=10, color=COLORS[run], alpha=0.35, linewidths=0, zorder=2)
    ax.axhline(0, color=AXIS, lw=0.8, zorder=1)
    ax.axhline(bias, color=INK_SECONDARY, lw=1.5, zorder=3)
    for limit in [bias - 1.96 * sd, bias + 1.96 * sd]:
        ax.axhline(limit, color=INK_MUTED, lw=1.2, ls='--', zorder=3)

    # Direct labels at the right edge instead of a legend
    ax.set_xlim(0, 100)
    label_x = ax.get_xlim()[1]
    ax.text(label_x, bias, f' bias {bias:+.2f}', va='center', color=INK_SECONDARY, fontsize=9)
    ax.text(label_x, bias + 1.96 * sd, f' +1.96 SD {bias + 1.96 * sd:+.2f}', va='center', color=INK_SECONDARY, fontsize=9)
    ax.text(label_x, bias - 1.96 * sd, f' -1.96 SD {bias - 1.96 * sd:+.2f}', va='center', color=INK_SECONDARY, fontsize=9)

    ax.set_xlabel('Mean of true and predicted EF (%)')
    ax.set_ylabel('Predicted - true EF (EF points)')
    ax.set_title(LABELS[run])


# ----------------------------------------------------------------------------
# error_by_ef_range
# ----------------------------------------------------------------------------
def error_by_ef_range(df, n_boot, seed):
    """MAE with bootstrap 95% CI in each true-EF range.

    Returns:
        DataFrame with columns range, n, mae, low, high.
    """
    bins = pd.cut(df['EF'], EF_BINS, labels=EF_BIN_LABELS, right=False)
    abs_error = np.abs(df['EF_pred'] - df['EF']).values

    rows = []
    for label in EF_BIN_LABELS:
        errors = abs_error[(bins == label).values]
        if errors.size == 0:
            rows.append({'range': label, 'n': 0, 'mae': np.nan, 'low': np.nan, 'high': np.nan})
            continue
        mae, low, high = common.with_bootstrap_ci(lambda e: {'MAE': e.mean()}, [errors], n_boot, seed)['MAE']
        rows.append({'range': label, 'n': errors.size, 'mae': mae, 'low': low, 'high': high})

    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# plot_error_by_ef_range
# ----------------------------------------------------------------------------
def plot_error_by_ef_range(ax, tables):
    """Grouped bars of MAE per true-EF range, with 95% CI whiskers and video counts.

    Args:
        ax: matplotlib axes.
        tables: dict run -> DataFrame from error_by_ef_range.
    """
    x = np.arange(len(EF_BIN_LABELS))
    width = 0.8 / len(tables)

    for i, (run, table) in enumerate(tables.items()):
        offset = (i - (len(tables) - 1) / 2) * width
        ax.bar(x + offset, table['mae'], width=width, color=COLORS[run], edgecolor='white', linewidth=2, label=LABELS[run], zorder=2)
        ax.errorbar(x + offset, table['mae'], yerr=[table['mae'] - table['low'], table['high'] - table['mae']], fmt='none', ecolor=INK_SECONDARY, elinewidth=1, capsize=2, zorder=3)

    # Video counts under each range (identical across runs: same test videos)
    counts = next(iter(tables.values()))['n']
    ax.set_xticks(x)
    ax.set_xticklabels([f'{label}\nn={n}' for label, n in zip(EF_BIN_LABELS, counts)])
    ax.set_xlabel('True EF range (%)')
    ax.set_ylabel('MAE (EF points)')
    ax.grid(axis='x', visible=False)
    if len(tables) > 1:
        ax.legend(loc='upper right')


# ----------------------------------------------------------------------------
# plot_roc / plot_pr
# ----------------------------------------------------------------------------
def plot_roc(ax, scores):
    """ROC curves for LV dysfunction detection, one curve per entity."""
    ax.plot([0, 1], [0, 1], color=INK_MUTED, lw=1, ls='--', zorder=1)
    for entity, (labels, score) in scores.items():
        fpr, tpr, _ = metrics.roc_curve(labels, score)
        ax.plot(fpr, tpr, color=COLORS[entity], label=f'{LABELS[entity]} (AUROC {metrics.roc_auc_score(labels, score):.3f})', zorder=2)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.01)
    ax.set_aspect('equal')
    ax.set_xlabel('1 - specificity')
    ax.set_ylabel('Sensitivity')
    ax.set_title(f'LV dysfunction (EF < {common.LV_DYSFUNCTION_EF}%): ROC')
    ax.legend(loc='lower right')


def plot_pr(ax, scores):
    """Precision-recall curves for LV dysfunction detection, with the prevalence baseline."""
    labels = next(iter(scores.values()))[0]
    ax.axhline(labels.mean(), color=INK_MUTED, lw=1, ls='--', zorder=1)
    ax.text(1.0, labels.mean(), f'prevalence {labels.mean():.2f}', ha='right', va='bottom', color=INK_SECONDARY, fontsize=9)
    for entity, (labels, score) in scores.items():
        precision, recall, _ = metrics.precision_recall_curve(labels, score)
        ax.plot(recall, precision, color=COLORS[entity], label=f'{LABELS[entity]} (AP {metrics.average_precision_score(labels, score):.3f})', zorder=2)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.01)
    ax.set_aspect('equal')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title(f'LV dysfunction (EF < {common.LV_DYSFUNCTION_EF}%): precision-recall')
    ax.legend(loc='lower left')


# ----------------------------------------------------------------------------
# plot_metric_comparison
# ----------------------------------------------------------------------------
def plot_metric_comparison(runs, n_boot, seed):
    """Small multiples of MAE, RMSE and R^2 per run, with 95% CI whiskers.

    Returns:
        matplotlib Figure.
    """
    results = {run: common.with_bootstrap_ci(common.regression_metrics, [df['EF'].values, df['EF_pred'].values], n_boot, seed) for run, df in runs.items()}

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.4))
    for ax, (metric, ylabel, fmt) in zip(axes, [('MAE', 'EF points', '.2f'), ('RMSE', 'EF points', '.2f'), ('R2', 'R$^2$', '.3f')]):
        for i, run in enumerate(runs):
            value, low, high = results[run][metric]
            ax.bar(i, value, width=0.6, color=COLORS[run], zorder=2)
            ax.errorbar(i, value, yerr=[[value - low], [high - value]], fmt='none', ecolor=INK_SECONDARY, elinewidth=1, capsize=3, zorder=3)
            ax.text(i, high, f'{value:{fmt}}', ha='center', va='bottom', color=INK, fontsize=9)
        ax.set_xticks(range(len(runs)))
        ax.set_xticklabels([LABELS[run] for run in runs])
        ax.set_title(metric.replace('R2', 'R$^2$'))
        ax.set_ylabel(ylabel)
        ax.grid(axis='x', visible=False)
        ax.margins(y=0.15)

    fig.suptitle('EF agreement on the test split (95% bootstrap CI)', color=INK, fontsize=11, fontweight='bold', y=1.03)
    fig.tight_layout()

    return fig


# ----------------------------------------------------------------------------
# plot_seed_variability
# ----------------------------------------------------------------------------
def plot_seed_variability(summaries):
    """Small multiples of each metric across seeds: one dot per seed, plus mean +/- SD.

    Args:
        summaries: dict run -> summary.csv DataFrame.

    Returns:
        matplotlib Figure.
    """
    fig, axes = plt.subplots(1, len(SEED_METRICS), figsize=(3 * len(SEED_METRICS), 3.6))
    for ax, (key, title, fmt) in zip(axes, SEED_METRICS):
        for i, (run, table) in enumerate(summaries.items()):
            values = table.loc[table['metric'] == key, 'value'].values
            if values.size == 0:
                continue
            mean = values.mean()
            sd = values.std(ddof=1) if values.size > 1 else 0.0

            # Seeds as dots on the left, mean +/- SD on the right of each run
            jitter = np.linspace(-0.12, 0.12, values.size) if values.size > 1 else np.zeros(1)
            ax.scatter(i - 0.1 + jitter, values, s=40, color=COLORS[run], edgecolors='white', linewidths=1.5, zorder=3)
            ax.errorbar(i + 0.2, mean, yerr=sd, fmt='o', color=INK_SECONDARY, ms=5, capsize=3, elinewidth=1.2, zorder=3)
            ax.text(i + 0.3, mean, f'{mean:{fmt}}\n+/- {sd:{fmt}}', va='center', color=INK_SECONDARY, fontsize=8)

        ax.set_xticks(range(len(summaries)))
        ax.set_xticklabels([LABELS[run] for run in summaries])
        ax.set_xlim(-0.6, len(summaries) - 0.1)
        ax.set_title(title, fontsize=10)
        ax.grid(axis='x', visible=False)
        ax.margins(y=0.25)

    n_seeds = max(table['seed'].nunique() for table in summaries.values())
    fig.suptitle(f'Variation across seeds (dots: individual seeds, whisker: mean +/- SD, n = {n_seeds})', color=INK, fontsize=11, fontweight='bold', y=1.03)
    fig.tight_layout()

    return fig


# ----------------------------------------------------------------------------
# plot_training_curves
# ----------------------------------------------------------------------------
def plot_training_curves(histories):
    """Fine-tuning loss and validation MAE per epoch, one line per seed.

    Args:
        histories: dict seed -> history.csv DataFrame.

    Returns:
        matplotlib Figure.
    """
    fig, (ax_loss, ax_mae) = plt.subplots(1, 2, figsize=(9.5, 3.6))
    color = COLORS['finetune']

    for seed, history in histories.items():
        ax_loss.plot(history['epoch'], history['train_loss'], color=color, ls='--', marker='o', ms=4, alpha=0.8)
        ax_loss.plot(history['epoch'], history['val_loss'], color=color, marker='o', ms=4)
        ax_mae.plot(history['epoch'], history['val_mae'], color=color, marker='o', ms=4)

        # Seeds share one color (same model); a direct label at the line end tells them apart
        last = history.iloc[-1]
        ax_mae.annotate(f'seed {seed}', (last['epoch'], last['val_mae']), xytext=(6, 0), textcoords='offset points', va='center', color=INK_SECONDARY, fontsize=8)

    epochs = sorted({int(e) for history in histories.values() for e in history['epoch']})
    for ax in [ax_loss, ax_mae]:
        ax.set_xticks(epochs)
        ax.set_xlabel('Epoch')
        ax.margins(x=0.15)

    ax_loss.set_ylabel('Loss (MSE / train mean EF)')
    ax_loss.set_title('Training and validation loss')
    ax_loss.legend(handles=[Line2D([0], [0], color=color, ls='--', marker='o', ms=4, label='train'), Line2D([0], [0], color=color, marker='o', ms=4, label='validation')], loc='upper right')
    ax_mae.set_ylabel('Validation MAE (EF points)')
    ax_mae.set_title('Validation MAE')

    fig.tight_layout()

    return fig


# ----------------------------------------------------------------------------
# read_complexity / complexity_value
# ----------------------------------------------------------------------------
def read_complexity(path):
    """Parse a complexity.txt written by the run scripts.

    Returns:
        dict label -> value text, for every measurement row.
    """
    rows = {}
    with open(path, encoding='utf-8') as f:
        for line in f:
            match = re.match(r'^   (\S.*?)\s{2,}(\S.*)$', line.rstrip())
            if match:
                rows[match.group(1)] = match.group(2)

    return rows


def complexity_value(rows, prefix):
    """First number in the value of the first row whose label starts with prefix."""
    for label, value in rows.items():
        if label.startswith(prefix):
            return float(re.search(r'\d+\.?\d*', value).group())

    raise KeyError(f'No complexity row starting with "{prefix}"')


# ----------------------------------------------------------------------------
# plot_cost
# ----------------------------------------------------------------------------
def plot_cost(costs, histories):
    """Computational cost: compute, GPU memory, fine-tuning time and inference time.

    Args:
        costs: dict run -> rows from read_complexity.
        histories: dict seed -> fine-tuning history DataFrame.

    Returns:
        matplotlib Figure.
    """
    zeroshot, finetune = costs['zeroshot'], costs['finetune']
    fig, ((ax_flops, ax_memory), (ax_time, ax_speed)) = plt.subplots(2, 2, figsize=(10.5, 7.6))

    # Compute: one forward pass vs. one training step on a single clip (forward cost is shared by both models)
    forward = complexity_value(finetune, 'Forward GFLOPs per clip')
    train = complexity_value(finetune, 'Training-step GFLOPs per video')
    ax_flops.bar([0, 1], [forward, train], width=0.55, color=[INK_MUTED, COLORS['finetune']], edgecolor='white', linewidth=2, zorder=2)
    for x, value in enumerate([forward, train]):
        ax_flops.text(x, value, f'{value:.0f}', ha='center', va='bottom', color=INK, fontsize=10)
    ax_flops.set_xticks([0, 1])
    ax_flops.set_xticklabels(['Forward pass\n(inference, both models)', 'Forward + backward\n(training step)'])
    ax_flops.set_ylabel('GFLOPs per 16-frame clip')
    ax_flops.set_title(f'Compute (training step = {train / forward:.1f}x forward)')
    ax_flops.grid(axis='x', visible=False)
    ax_flops.margins(y=0.15)

    # GPU memory: peak allocated, against the memory of the card
    names = ['Inference, zero-shot', 'Inference, fine-tuned', 'Training, fine-tuned']
    memory = [complexity_value(zeroshot, 'Peak allocated (inference)'), complexity_value(finetune, 'Peak allocated (test inference)'), complexity_value(finetune, 'Peak allocated (training)')]
    ax_memory.barh(range(3), memory, height=0.55, color=[COLORS['zeroshot'], COLORS['finetune'], COLORS['finetune']], edgecolor='white', linewidth=2, zorder=2)
    for y, value in enumerate(memory):
        ax_memory.text(value + 0.1, y, f'{value:.2f} GB', va='center', color=INK, fontsize=10)
    ax_memory.axvline(GPU_MEMORY_GB, color=INK_SECONDARY, lw=1.2, ls='--', zorder=3)
    ax_memory.text(GPU_MEMORY_GB - 0.1, -0.6, f'GPU memory {GPU_MEMORY_GB:.0f} GB', ha='right', va='center', color=INK_SECONDARY, fontsize=9)
    ax_memory.set_yticks(range(3))
    ax_memory.set_yticklabels(names)
    ax_memory.set_xlim(0, GPU_MEMORY_GB + 0.4)
    ax_memory.set_ylim(2.6, -0.9)
    ax_memory.set_xlabel('Peak allocated memory (GB)')
    ax_memory.set_title('GPU memory')
    ax_memory.grid(axis='y', visible=False)

    # Fine-tuning time per seed, split into training and validation
    seeds = sorted(histories)
    train_minutes = np.array([histories[s]['train_minutes'].sum() for s in seeds])
    val_minutes = np.array([histories[s]['val_minutes'].sum() for s in seeds])
    x = np.arange(len(seeds))
    ax_time.bar(x, train_minutes, width=0.55, color=COLORS['finetune'], edgecolor='white', linewidth=2, label='training', zorder=2)
    ax_time.bar(x, val_minutes, width=0.55, bottom=train_minutes, color=COLORS['finetune'], alpha=0.45, edgecolor='white', linewidth=2, label='validation', zorder=2)
    for xi, total in zip(x, train_minutes + val_minutes):
        ax_time.text(xi, total, f'{total:.0f} min', ha='center', va='bottom', color=INK, fontsize=10)
    ax_time.set_xticks(x)
    ax_time.set_xticklabels([f'seed {s}' for s in seeds])
    ax_time.set_ylabel('Minutes')
    ax_time.set_title(f'Fine-tuning time ({len(histories[seeds[0]])} epochs)')
    ax_time.grid(axis='x', visible=False)
    ax_time.margins(y=0.3)
    ax_time.legend(loc='upper center', ncol=2)

    # Inference time per video: the model alone, and including video decoding
    groups = ['Model only', 'Including video decoding']
    width = 0.34
    for i, run in enumerate(['zeroshot', 'finetune']):
        values = [complexity_value(costs[run], 'Model time per video'), complexity_value(costs[run], 'Wall time per video')]
        positions = np.arange(2) + (i - 0.5) * width
        ax_speed.bar(positions, values, width=width, color=COLORS[run], edgecolor='white', linewidth=2, label=LABELS[run], zorder=2)
        for position, value in zip(positions, values):
            ax_speed.text(position, value, f'{value:.0f}', ha='center', va='bottom', color=INK, fontsize=10)
    ax_speed.set_xticks(range(2))
    ax_speed.set_xticklabels(groups)
    ax_speed.set_ylabel('Milliseconds per video (4 clips)')
    ax_speed.set_title('Inference time per video')
    ax_speed.grid(axis='x', visible=False)
    ax_speed.margins(y=0.2)
    ax_speed.legend(loc='upper left')

    fig.suptitle(f'Computational cost ({zeroshot["GPU"]})', color=INK, fontsize=12, fontweight='bold', y=1.0)
    fig.tight_layout()

    return fig


# ----------------------------------------------------------------------------
# make_cost_figure
# ----------------------------------------------------------------------------
def make_cost_figure(results_dir, histories, figures_dir):
    """Cost figure from both runs' complexity.txt and the fine-tuning histories."""
    paths = {run: os.path.join(results_dir, run, 'complexity.txt') for run in ['zeroshot', 'finetune']}
    if not all(os.path.exists(path) for path in paths.values()) or len(histories) == 0:
        print(' Cost figure skipped: it needs complexity.txt of both runs and the fine-tuning histories.')
        return

    costs = {run: read_complexity(path) for run, path in paths.items()}
    params = complexity_value(costs['zeroshot'], 'Parameters (EF')
    note = (f'Both models: {params:.1f} M parameters (same architecture), so inference compute is identical; fine-tuning only adds the one-time training cost.\n'
            'Inference time was measured separately in each run on the same laptop GPU; the gap between runs is probably thermal throttling (not verified).')
    save_figure(plot_cost(costs, histories), os.path.join(figures_dir, 'comparison'), 'cost_comparison.png', note)


# ----------------------------------------------------------------------------
# make_run_figures
# ----------------------------------------------------------------------------
def make_run_figures(run, df, figures_dir, args):
    """Single-seed figures for one run, each in its own subfolder."""
    folder = os.path.join(figures_dir, run)
    note = f'EchoNet-Dynamic test split, {LABELS[run].lower()}, seed {args.seed}'

    fig, ax = plt.subplots(figsize=(4.6, 4.6))
    plot_ef_scatter(ax, df, run)
    save_figure(fig, os.path.join(folder, 'ef_scatter'), 'ef_scatter_test.png', note)

    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    plot_bland_altman(ax, df, run)
    save_figure(fig, os.path.join(folder, 'bland_altman'), 'bland_altman_test.png', note)

    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    plot_error_by_ef_range(ax, {run: error_by_ef_range(df, args.n_boot, args.seed)})
    ax.set_title(f'{LABELS[run]}: error by true EF range (95% CI)')
    save_figure(fig, os.path.join(folder, 'error_by_ef_range'), 'error_by_ef_range_test.png', note)

    scores = dysfunction_scores({run: df})
    fig, ax = plt.subplots(figsize=(4.8, 4.8))
    plot_roc(ax, scores)
    save_figure(fig, os.path.join(folder, 'lv_dysfunction_roc'), 'roc_test.png', note)

    fig, ax = plt.subplots(figsize=(4.8, 4.8))
    plot_pr(ax, scores)
    save_figure(fig, os.path.join(folder, 'lv_dysfunction_roc'), 'pr_test.png', note)


# ----------------------------------------------------------------------------
# make_comparison_figures
# ----------------------------------------------------------------------------
def make_comparison_figures(runs, figures_dir, args):
    """Zero-shot vs. fine-tuned single-seed figures, on the videos both runs predicted."""
    shared = set(runs['zeroshot']['fname']) & set(runs['finetune']['fname'])
    runs = {run: df[df['fname'].isin(shared)].sort_values('fname').reset_index(drop=True) for run, df in runs.items()}
    folder = os.path.join(figures_dir, 'comparison')
    note = f'EchoNet-Dynamic test split, seed {args.seed}'

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.6), sharex=True, sharey=True)
    for ax, (run, df) in zip(axes, runs.items()):
        plot_ef_scatter(ax, df, run)
    fig.tight_layout()
    save_figure(fig, folder, 'ef_scatter_comparison_test.png', note)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
    for ax, (run, df) in zip(axes, runs.items()):
        plot_bland_altman(ax, df, run)
    fig.tight_layout()
    save_figure(fig, folder, 'bland_altman_comparison_test.png', note)

    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    plot_error_by_ef_range(ax, {run: error_by_ef_range(df, args.n_boot, args.seed) for run, df in runs.items()})
    ax.set_title('Error by true EF range (95% CI)')
    save_figure(fig, folder, 'error_by_ef_range_comparison_test.png', note)

    scores = dysfunction_scores(runs)
    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    plot_roc(ax, scores)
    save_figure(fig, folder, 'roc_comparison_test.png', note)

    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    plot_pr(ax, scores)
    save_figure(fig, folder, 'pr_comparison_test.png', note)

    save_figure(plot_metric_comparison(runs, args.n_boot, args.seed), folder, 'metrics_comparison_test.png', note)


# ----------------------------------------------------------------------------
# make_seed_figures
# ----------------------------------------------------------------------------
def make_seed_figures(summaries, histories, figures_dir):
    """Figures across seeds: metric variability per run and together, and training curves."""
    for run, table in summaries.items():
        seeds = ', '.join(str(s) for s in sorted(table['seed'].unique()))
        save_figure(plot_seed_variability({run: table}), os.path.join(figures_dir, run, 'seed_variability'), 'seed_variability_test.png', f'EchoNet-Dynamic test split, {LABELS[run].lower()}, seeds {seeds}')

    if len(summaries) == 2:
        save_figure(plot_seed_variability(summaries), os.path.join(figures_dir, 'comparison'), 'seed_variability_comparison_test.png', 'EchoNet-Dynamic test split, all seeds')

    if len(histories) > 0:
        seeds = ', '.join(str(s) for s in histories)
        save_figure(plot_training_curves(histories), os.path.join(figures_dir, 'finetune', 'training_curves'), 'training_curves.png', f'EchoNet-Dynamic train/validation splits, fine-tuned, seeds {seeds}')


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------
def main(args):
    set_style()

    runs = load_predictions(args.results_dir, args.seed)
    summaries = load_summaries(args.results_dir)
    histories = load_histories(args.results_dir)
    if len(runs) == 0 and len(summaries) == 0:
        print(' No results found. Run echonetdynamic_zeroshot.py and/or echonetdynamic_finetune.py first.')
        return

    figures_dir = os.path.join(args.results_dir, 'figures')
    print(f' Figures from: {", ".join(LABELS[run] for run in runs)} (seed {args.seed}, bootstrap {args.n_boot} resamples)')

    for run, df in runs.items():
        make_run_figures(run, df, figures_dir, args)

    if len(runs) == 2:
        make_comparison_figures(runs, figures_dir, args)
    else:
        print(' Comparison figures skipped: they need both zero-shot and fine-tuned predictions.')

    make_seed_figures(summaries, histories, figures_dir)
    make_cost_figure(args.results_dir, histories, figures_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Figures from the saved EchoNet-Dynamic test results.')
    parser.add_argument('--results_dir', type=str, default=common.RESULTS_DIR)
    parser.add_argument('--seed', type=int, default=0, help="which seed's predictions to plot (also seeds the bootstrap)")
    parser.add_argument('--n_boot', type=int, default=1000, help='bootstrap resamples for the 95%% CIs')

    args = parser.parse_args()

    main(args)
