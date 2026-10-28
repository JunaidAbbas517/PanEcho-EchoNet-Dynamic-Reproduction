
import contextlib
import functools
import io
import os
import random
import sys
import warnings

import numpy as np
import pandas as pd
import torch
import tqdm

from sklearn import metrics

import complexity

REPRO_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(REPRO_DIR)
SRC_DIR = os.path.join(REPO_DIR, 'src')
DATA_DIR = os.path.join(REPO_DIR, 'Dataset', 'EchoNet-Dynamic')
RESULTS_DIR = os.path.join(REPRO_DIR, 'results')

# Make the authors' code in src/ importable
sys.path.insert(0, SRC_DIR)

from dataset import EchoNetDynamicDataset  # noqa: E402
from utils import Task, set_seed, worker_init_fn  # noqa: E402,F401

# Moderate or worse LV systolic dysfunction (ASE: moderately reduced EF is 30-40%)
LV_DYSFUNCTION_EF = 40

# Batches skipped before timing starts (CUDA kernel selection and memory warm-up)
WARMUP_BATCHES = 2

# Windows PyTorch builds lack flash attention; the fallback kernel is used silently
warnings.filterwarnings('ignore', message='.*flash attention.*')


# ----------------------------------------------------------------------------
# load_panecho
# ----------------------------------------------------------------------------
def load_panecho(tasks, clip_len=16, pretrained=True):
    """Load PanEcho through the authors' local hubconf.py.

    Args:
        tasks: list of task names to keep (e.g. ['EF']), or 'all' for all 39 heads.
        clip_len: number of frames per clip.
        pretrained: load the released PanEcho weights.

    Returns:
        PanEcho model on the CPU.
    """
    with contextlib.redirect_stdout(io.StringIO()):
        model = torch.hub.load(REPO_DIR, 'PanEcho', source='local', verbose=False, pretrained=pretrained, tasks=tasks, clip_len=clip_len)

    return model


# ----------------------------------------------------------------------------
# build_dataset
# ----------------------------------------------------------------------------
def build_dataset(split, clip_len=16, num_clips=4, num_videos=0):
    """Build the authors' EchoNet-Dynamic dataset for one split.

    Preprocessing matches the authors' fine-tuning setup: resize to 256x256,
    center crop to 224x224, ImageNet normalization, no augmentation. The
    training split returns one random clip per video; val/test return num_clips.

    Args:
        split: 'train', 'val' or 'test'.
        clip_len: number of frames per clip.
        num_clips: clips per video for val/test.
        num_videos: keep only the first num_videos videos (0 keeps all).

    Returns:
        torch Dataset yielding dicts with 'x', 'fname' and 'EF'.
    """
    data_df = pd.read_csv(os.path.join(DATA_DIR, 'FileList.csv'))

    # A fresh Task per dataset: the authors' loader overwrites task.mean with
    # each split's mean, so a shared Task would leak val/test statistics.
    tasks = [Task('EF', 'regression', np.array([]))]

    # The authors' loader prints the whole file list; keep the terminal clean
    with contextlib.redirect_stdout(io.StringIO()):
        dataset = EchoNetDynamicDataset(data_dir=DATA_DIR, data_df=data_df, tasks=tasks, split=split, clip_len=clip_len, num_clips=num_clips, augment=False, normalization='imagenet')

    if num_videos > 0:
        dataset = torch.utils.data.Subset(dataset, range(min(num_videos, len(dataset))))

    return dataset


# ----------------------------------------------------------------------------
# split_mean_ef
# ----------------------------------------------------------------------------
def split_mean_ef(split):
    """Mean true EF of one split, read directly from FileList.csv."""
    data_df = pd.read_csv(os.path.join(DATA_DIR, 'FileList.csv'))

    return float(data_df.loc[data_df['Split'] == split.upper(), 'EF'].mean())


# ----------------------------------------------------------------------------
# eval_worker_init_fn
# ----------------------------------------------------------------------------
def eval_worker_init_fn(worker_id, seed=0):
    """Seed clip sampling in a val/test loader worker.

    The authors' val_worker_init_fn seeds each worker with its worker id only,
    so every run would pick the same clips. Adding 1000 x seed gives each seed
    its own clips while seed 0 stays identical to the authors' seeding.
    """
    np.random.seed(worker_id + 1000 * seed)
    random.seed(worker_id + 1000 * seed)


# ----------------------------------------------------------------------------
# make_loader
# ----------------------------------------------------------------------------
def make_loader(dataset, batch_size, num_workers, train=False, seed=0):
    """Wrap a dataset in a DataLoader with seeded workers.

    Args:
        dataset: dataset from build_dataset.
        batch_size: videos per batch.
        num_workers: background processes decoding videos.
        train: shuffle and use the authors' training worker seeding.
        seed: run seed, for the shuffle order and the clip sampling.

    Returns:
        torch DataLoader.
    """
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        # Pinned memory runs out on Windows after ~200 batches of 4-clip videos (~150 MB each)
        pin_memory=False,
        worker_init_fn=worker_init_fn if train else functools.partial(eval_worker_init_fn, seed=seed),
        generator=torch.Generator().manual_seed(seed),
    )


# ----------------------------------------------------------------------------
# predict_videos
# ----------------------------------------------------------------------------
def predict_videos(model, loader, amp, desc):
    """Predict every video by averaging the model outputs over its clips.

    This follows the authors' evaluate_echonetdynamic in src/utils.py.

    Args:
        model: PanEcho model on the GPU.
        loader: DataLoader from make_loader (val or test split).
        amp: run the forward pass in float16.
        desc: label for the progress bar.

    Returns:
        fnames: list of video names.
        ef_true: true EF per video, shape (N,).
        outputs: dict task name -> clip-averaged output, shape (N, K).
        model_seconds: model time per video for each timed batch.
    """
    model.eval()

    fnames, ef_true, outputs, model_seconds = [], [], {}, []
    with torch.no_grad():
        for b, batch in enumerate(tqdm.tqdm(loader, desc=desc, unit='batch', dynamic_ncols=True)):
            x = batch['x']
            # (B, 3, num_clips, clip_len, H, W); a single clip arrives without the clip dimension
            if x.ndim == 5:
                x = x.unsqueeze(2)
            x = x.cuda(non_blocking=True)

            with complexity.CudaTimer() as timer:
                with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=amp):
                    clip_outputs = [model(x[:, :, clip]) for clip in range(x.shape[2])]

            if b >= WARMUP_BATCHES:
                model_seconds.append(timer.seconds / x.shape[0])

            for task in clip_outputs[0]:
                mean_output = torch.stack([out[task] for out in clip_outputs], dim=0).float().mean(dim=0)
                outputs.setdefault(task, []).append(mean_output.cpu().numpy())

            fnames += list(batch['fname'])
            ef_true.append(batch['EF'].numpy().ravel())

    outputs = {task: np.concatenate(values, axis=0) for task, values in outputs.items()}

    return fnames, np.concatenate(ef_true), outputs, model_seconds


# ----------------------------------------------------------------------------
# regression_metrics / classification_metrics
# ----------------------------------------------------------------------------
def regression_metrics(ef_true, ef_pred):
    """EF agreement metrics: MAE, RMSE, R^2 and bias (mean of predicted minus true)."""
    error = ef_pred - ef_true

    return {
        'MAE': np.mean(np.abs(error)),
        'RMSE': np.sqrt(np.mean(error ** 2)),
        'R2': metrics.r2_score(ef_true, ef_pred),
        'Bias': np.mean(error),
    }


def classification_metrics(labels, scores):
    """AUROC and average precision; NaN when only one class is present."""
    if labels.min() == labels.max():
        return {'AUROC': np.nan, 'AP': np.nan}

    return {
        'AUROC': metrics.roc_auc_score(labels, scores),
        'AP': metrics.average_precision_score(labels, scores),
    }


# ----------------------------------------------------------------------------
# with_bootstrap_ci
# ----------------------------------------------------------------------------
def with_bootstrap_ci(metric_fn, arrays, n_boot=1000, seed=0):
    """Compute metrics with 95% percentile bootstrap confidence intervals.

    Videos are resampled with replacement n_boot times; the interval is the
    2.5th to 97.5th percentile of each metric across resamples.

    Args:
        metric_fn: function taking the arrays and returning dict name -> value.
        arrays: list of equal-length numpy arrays (e.g. [ef_true, ef_pred]).
        n_boot: number of bootstrap resamples.
        seed: seed for the resampling.

    Returns:
        dict name -> (value, ci_low, ci_high).
    """
    point = metric_fn(*arrays)

    rng = np.random.default_rng(seed)
    n = len(arrays[0])
    samples = {name: [] for name in point}
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        for name, value in metric_fn(*[a[idx] for a in arrays]).items():
            samples[name].append(value)

    return {name: (point[name], np.nanpercentile(samples[name], 2.5), np.nanpercentile(samples[name], 97.5)) for name in point}


# ----------------------------------------------------------------------------
# RunLogger
# ----------------------------------------------------------------------------
class RunLogger:
    """Print to the terminal and append the same text to a log file (run.log)."""
    def __init__(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.file = open(path, 'w', encoding='utf-8')

    def log(self, text=''):
        print(text)
        self.file.write(text + '\n')
        self.file.flush()

    def close(self):
        self.file.close()


# ----------------------------------------------------------------------------
# environment_info
# ----------------------------------------------------------------------------
def environment_info():
    """GPU name and library versions, recorded next to every timing."""
    return {
        'GPU': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none',
        'CUDA': torch.version.cuda,
        'PyTorch': torch.__version__,
    }


# ----------------------------------------------------------------------------
# log_header / log_section
# ----------------------------------------------------------------------------
LINE_WIDTH = 64


def log_header(logger, title, settings):
    """Log a run header: title, run settings and hardware."""
    logger.log('=' * LINE_WIDTH)
    logger.log(f' {title}')
    logger.log('=' * LINE_WIDTH)
    for key, value in {**settings, **environment_info()}.items():
        logger.log(f' {key:<18}{value}')
    logger.log('-' * LINE_WIDTH)


def log_section(logger, title):
    """Log a section divider with a title."""
    logger.log('')
    logger.log(f' {title}')
    logger.log('-' * LINE_WIDTH)


# ----------------------------------------------------------------------------
# format_metrics_table
# ----------------------------------------------------------------------------
def format_metrics_table(rows):
    """Format metrics with confidence intervals as aligned text lines.

    Args:
        rows: list of (key, label, (value, ci_low, ci_high), number format such as '.2f').

    Returns:
        List of text lines.
    """
    lines = [f' {"Metric":<34}{"Value":>8}   95% CI', ' ' + '-' * (LINE_WIDTH - 2)]
    for _, label, (value, low, high), fmt in rows:
        lines.append(f' {label:<34}{value:>8{fmt}}   [{low:{fmt}}, {high:{fmt}}]')

    return lines


# ----------------------------------------------------------------------------
# summarize_seeds
# ----------------------------------------------------------------------------
def summarize_seeds(seed_rows):
    """Mean +/- SD of every metric across seeds, with each seed's value.

    Args:
        seed_rows: dict seed -> rows in the format_metrics_table format.

    Returns:
        lines: aligned text lines for the terminal and summary.txt.
        table: DataFrame with seed, metric key, label, value, ci_low, ci_high (summary.csv).
    """
    seeds = list(seed_rows)
    header = f' {"Metric":<34}{"Mean +/- SD":>18}' + ''.join(f'{"seed " + str(s):>10}' for s in seeds)
    lines = [header, ' ' + '-' * (len(header) - 1)]

    records = []
    for i, (key, label, _, fmt) in enumerate(seed_rows[seeds[0]]):
        values = np.array([seed_rows[s][i][2][0] for s in seeds], dtype=float)
        mean = np.nanmean(values) if not np.isnan(values).all() else np.nan
        # Sample SD (n - 1); undefined for a single seed
        mean_sd = f'{mean:{fmt}} +/- {np.std(values, ddof=1):{fmt}}' if len(values) > 1 else f'{mean:{fmt}}'
        lines.append(f' {label:<34}{mean_sd:>18}' + ''.join(f'{v:>10{fmt}}' for v in values))

        for s in seeds:
            value, low, high = seed_rows[s][i][2]
            records.append({'seed': s, 'metric': key, 'label': label, 'value': value, 'ci_low': low, 'ci_high': high})

    return lines, pd.DataFrame(records)


# ----------------------------------------------------------------------------
# write_lines
# ----------------------------------------------------------------------------
def write_lines(path, lines):
    """Write text lines to a file, creating its folder if needed."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
