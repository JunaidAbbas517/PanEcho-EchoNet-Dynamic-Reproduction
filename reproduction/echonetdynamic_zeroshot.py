"""Zero-shot evaluation of pretrained PanEcho on EchoNet-Dynamic (no training).

Uses the authors' released weights and their EchoNet-Dynamic data loader, then
reports EF agreement and LV dysfunction detection with bootstrap 95% CIs. With
several seeds, each seed samples different clips and the summary reports the
mean +/- SD across seeds.
"""
import argparse
import os
import time

import numpy as np
import pandas as pd
import torch

import common
import complexity


# ----------------------------------------------------------------------------
# load_model
# ----------------------------------------------------------------------------
def load_model(clip_len):
    """Load pretrained PanEcho with the EF and LV systolic function heads.

    Args:
        clip_len: number of frames per clip.

    Returns:
        model: PanEcho on the GPU in eval mode.
        dysfunction_idx: index of the 'Moderately|Severely Decreased' class
            in the LVSystolicFunction output.
    """
    model = common.load_panecho(['EF', 'LVSystolicFunction'], clip_len).cuda().eval()

    task = next(t for t in model.tasks if t.task_name == 'LVSystolicFunction')
    dysfunction_idx = int(np.where(task.class_names == 'Moderately|Severely Decreased')[0][0])

    return model, dysfunction_idx


# ----------------------------------------------------------------------------
# measure_model_size
# ----------------------------------------------------------------------------
def measure_model_size(model, args):
    """Count parameters and measure forward-pass GFLOPs for one clip and one video.

    Args:
        model: PanEcho on the GPU.
        args: parsed command-line arguments (clip_len, num_clips).

    Returns:
        dict with parameter counts and GFLOPs.
    """
    full_model = common.load_panecho('all', args.clip_len, pretrained=False)
    full_params = complexity.count_parameters(full_model)
    del full_model

    clip = torch.zeros(1, 3, args.clip_len, 224, 224, device='cuda')
    gflops_clip = complexity.count_gflops(model, clip)

    return {
        'params': complexity.count_parameters(model),
        'params_full': full_params,
        'gflops_clip': gflops_clip,
        'gflops_video': gflops_clip * args.num_clips,
    }


# ----------------------------------------------------------------------------
# build_results_table
# ----------------------------------------------------------------------------
def build_results_table(predictions, n_boot, seed):
    """Compute all metrics with bootstrap 95% CIs.

    Args:
        predictions: DataFrame with EF, EF_pred and p_lv_dysfunction per video.
        n_boot: bootstrap resamples.
        seed: seed for the bootstrap resampling.

    Returns:
        List of table rows for common.format_metrics_table.
    """
    ef_true = predictions['EF'].values
    ef_pred = predictions['EF_pred'].values
    labels = (ef_true < common.LV_DYSFUNCTION_EF).astype(int)

    ef = common.with_bootstrap_ci(common.regression_metrics, [ef_true, ef_pred], n_boot, seed)
    head = common.with_bootstrap_ci(common.classification_metrics, [labels, predictions['p_lv_dysfunction'].values], n_boot, seed)
    # Lower predicted EF means more likely dysfunction, so the score is -EF_pred
    from_ef = common.with_bootstrap_ci(common.classification_metrics, [labels, -ef_pred], n_boot, seed)

    threshold = f'EF<{common.LV_DYSFUNCTION_EF}'
    return [
        ('ef_mae', 'EF MAE (EF points)', ef['MAE'], '.2f'),
        ('ef_rmse', 'EF RMSE (EF points)', ef['RMSE'], '.2f'),
        ('ef_r2', 'EF R^2', ef['R2'], '.3f'),
        ('ef_bias', 'EF bias (pred - true)', ef['Bias'], '.2f'),
        ('auroc_head', f'{threshold} AUROC, LV function head', head['AUROC'], '.3f'),
        ('ap_head', f'{threshold} AP,    LV function head', head['AP'], '.3f'),
        ('auroc_ef', f'{threshold} AUROC, predicted EF', from_ef['AUROC'], '.3f'),
        ('ap_ef', f'{threshold} AP,    predicted EF', from_ef['AP'], '.3f'),
    ]


# ----------------------------------------------------------------------------
# run_seed
# ----------------------------------------------------------------------------
def run_seed(model, dysfunction_idx, dataset, seed, args, logger):
    """Evaluate every video once with one seed and save that seed's results.

    Args:
        model: PanEcho on the GPU.
        dysfunction_idx: LV dysfunction class index from load_model.
        dataset: dataset from common.build_dataset.
        seed: seed for the clip sampling and the bootstrap.
        args: parsed command-line arguments.
        logger: common.RunLogger.

    Returns:
        rows: metric rows for this seed.
        model_seconds: model time per video for each timed batch.
        wall_seconds: total time of the pass, including video decoding.
    """
    common.set_seed(seed)
    loader = common.make_loader(dataset, args.batch_size, args.num_workers, seed=seed)

    start = time.perf_counter()
    fnames, ef_true, outputs, model_seconds = common.predict_videos(model, loader, not args.fp32, desc=f'Seed {seed} [{args.split}]')
    wall_seconds = time.perf_counter() - start

    predictions = pd.DataFrame({
        'fname': fnames,
        'EF': ef_true,
        'EF_pred': outputs['EF'].ravel(),
        'p_lv_dysfunction': outputs['LVSystolicFunction'][:, dysfunction_idx],
    })
    seed_dir = os.path.join(args.out_dir, f'seed_{seed}')
    os.makedirs(seed_dir, exist_ok=True)
    predictions.to_csv(os.path.join(seed_dir, 'predictions.csv'), index=False)

    rows = build_results_table(predictions, args.n_boot, seed)
    table = common.format_metrics_table(rows)
    common.log_section(logger, f'Seed {seed}: {args.split} split (N={len(predictions):,}, {wall_seconds / 60:.1f} min)')
    for line in table:
        logger.log(line)
    common.write_lines(os.path.join(seed_dir, 'metrics.txt'), [f' PanEcho zero-shot, EchoNet-Dynamic {args.split} split (N={len(predictions):,}), seed {seed}', ''] + table)

    return rows, model_seconds, wall_seconds


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------
def main(args):
    logger = common.RunLogger(os.path.join(args.out_dir, 'run.log'))
    dataset = common.build_dataset(args.split, args.clip_len, args.num_clips, args.num_videos)

    common.log_header(logger, 'PanEcho zero-shot evaluation: EchoNet-Dynamic', {
        'Split': f'{args.split} ({len(dataset):,} videos)',
        'Seeds': ', '.join(str(s) for s in args.seeds),
        'Clips/video': f'{args.num_clips} x {args.clip_len} frames',
        'Precision': 'float32' if args.fp32 else 'float16 (autocast)',
        'Batch size': args.batch_size,
        'Bootstrap': f'{args.n_boot} resamples',
        'Output': os.path.relpath(args.out_dir, common.REPO_DIR),
    })

    # Model and its size (the model is identical for every seed)
    logger.log(' Loading pretrained PanEcho...')
    common.set_seed(args.seeds[0])
    model, dysfunction_idx = load_model(args.clip_len)
    size = measure_model_size(model, args)

    # One full pass per seed
    seed_rows, model_seconds, wall_seconds = {}, [], []
    complexity.reset_peak_memory()
    for seed in args.seeds:
        seed_rows[seed], seconds, wall = run_seed(model, dysfunction_idx, dataset, seed, args, logger)
        model_seconds += seconds
        wall_seconds.append(wall)

    # Mean +/- SD across seeds
    summary, summary_table = common.summarize_seeds(seed_rows)
    common.log_section(logger, f'Summary: mean +/- SD across {len(args.seeds)} seeds')
    for line in summary:
        logger.log(line)
    summary_table.to_csv(os.path.join(args.out_dir, 'summary.csv'), index=False)
    common.write_lines(os.path.join(args.out_dir, 'summary.txt'), [f' PanEcho zero-shot, EchoNet-Dynamic {args.split} split (N={len(dataset):,}), mean +/- SD across seeds {", ".join(str(s) for s in args.seeds)}', ''] + summary)

    # Complexity
    report = complexity.format_report([
        ('Hardware', list(common.environment_info().items())),
        ('Model size', [
            ('Parameters (EF + LV function heads)', f'{size["params"] / 1e6:.2f} M'),
            ('Parameters (full model, 39 heads)', f'{size["params_full"] / 1e6:.2f} M'),
        ]),
        ('Compute (one forward pass)', [
            (f'GFLOPs per clip ({args.clip_len} frames)', f'{size["gflops_clip"]:.1f}'),
            (f'GFLOPs per video ({args.num_clips} clips)', f'{size["gflops_video"]:.1f}'),
        ]),
        ('Time (all seeds)', [
            ('Model time per video', complexity.summarize_ms(model_seconds)),
            ('Wall time per video (with decoding)', f'{np.mean(wall_seconds) / len(dataset) * 1000:.1f} ms'),
            ('Time per seed', ', '.join(f'{w / 60:.1f} min' for w in wall_seconds)),
        ]),
        ('GPU memory', [
            ('Peak allocated (inference)', f'{complexity.peak_memory_gb():.2f} GB'),
        ]),
    ])
    common.log_section(logger, 'Complexity')
    for line in report:
        logger.log(line)
    common.write_lines(os.path.join(args.out_dir, 'complexity.txt'), report)

    logger.log(f' Saved: {os.path.relpath(args.out_dir, common.REPO_DIR)}{os.sep}{{summary.txt, summary.csv, complexity.txt, run.log, seed_*/}}')
    logger.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Zero-shot evaluation of pretrained PanEcho on EchoNet-Dynamic.')
    parser.add_argument('--split', type=str, default='test', choices=['val', 'test'])
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2], help='one full pass per seed, e.g. --seeds 0 1 2')
    parser.add_argument('--num_videos', type=int, default=0, help='evaluate only the first N videos (0 = all)')
    parser.add_argument('--clip_len', type=int, default=16)
    parser.add_argument('--num_clips', type=int, default=4, help='clips per video, predictions are averaged')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--n_boot', type=int, default=1000, help='bootstrap resamples for the 95%% CIs')
    parser.add_argument('--fp32', action='store_true', default=False, help='run in float32 instead of float16')
    parser.add_argument('--out_dir', type=str, default=os.path.join(common.RESULTS_DIR, 'zeroshot'))

    args = parser.parse_args()

    main(args)
