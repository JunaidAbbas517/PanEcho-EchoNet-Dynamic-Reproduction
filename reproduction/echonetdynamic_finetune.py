"""Fine-tune pretrained PanEcho on EchoNet-Dynamic EF, then evaluate on the test split.

Single-GPU version of the authors' src/train_echonetdynamic.py. Their run used
8 GPUs x 16 videos per step; here 4 videos x 32 accumulated batches keep the
same 128 videos per optimizer step on one 8 GB GPU. Each seed is a separate
fine-tuning run from the same pretrained weights; the summary reports the
mean +/- SD across seeds.
"""
import argparse
import os
import time

import numpy as np
import pandas as pd
import torch
import tqdm

import common
import complexity


# ----------------------------------------------------------------------------
# build_model
# ----------------------------------------------------------------------------
def build_model(clip_len):
    """Load pretrained PanEcho with only the EF head, set up as in the authors' fine-tuning.

    Args:
        clip_len: number of frames per clip.

    Returns:
        PanEcho on the GPU.
    """
    model = common.load_panecho(['EF'], clip_len)

    # The authors fine-tune with --fc_dropout 0; the released model uses 0.25
    model.EF_head[0].p = 0.0

    return model.cuda()


# ----------------------------------------------------------------------------
# train_one_epoch
# ----------------------------------------------------------------------------
def train_one_epoch(model, loader, optimizer, scaler, ef_mean, desc, args):
    """Train for one epoch with gradient accumulation.

    Args:
        model: PanEcho on the GPU.
        loader: training DataLoader (one random clip per video).
        optimizer: Adam optimizer.
        scaler: GradScaler for float16 training.
        ef_mean: training-set mean EF, used to scale the loss as the authors do.
        desc: label for the progress bar.
        args: parsed command-line arguments.

    Returns:
        mean_loss: mean training loss over the epoch.
        batch_seconds: time per batch (forward, backward and any optimizer step), warm-up skipped.
    """
    model.train()
    optimizer.zero_grad(set_to_none=True)

    num_batches = len(loader) if args.limit_train_batches == 0 else min(args.limit_train_batches, len(loader))
    losses, batch_seconds = [], []

    pbar = tqdm.tqdm(total=num_batches, desc=desc, unit='batch', dynamic_ncols=True)
    for b, batch in enumerate(loader):
        if b == num_batches:
            break

        with complexity.CudaTimer() as timer:
            x = batch['x'].cuda(non_blocking=True)
            y = batch['EF'].cuda(non_blocking=True)

            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=args.amp):
                yhat = model(x)['EF']
            # Same loss as the authors: MSE divided by the training-set mean EF
            loss = torch.nn.functional.mse_loss(yhat.float(), y) / ef_mean

            scaler.scale(loss / args.accum_steps).backward()

            # One optimizer step every accum_steps batches, and at the end of the epoch
            if (b + 1) % args.accum_steps == 0 or b + 1 == num_batches:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

        if b >= common.WARMUP_BATCHES:
            batch_seconds.append(timer.seconds)
        losses.append(loss.item())

        pbar.update(1)
        pbar.set_postfix(loss=f'{np.mean(losses):.4f}')
    pbar.close()

    return float(np.mean(losses)), batch_seconds


# ----------------------------------------------------------------------------
# evaluate
# ----------------------------------------------------------------------------
def evaluate(model, loader, ef_mean, amp, desc):
    """Evaluate EF on a val/test loader, averaging the prediction over each video's clips.

    Args:
        model: PanEcho on the GPU.
        loader: val or test DataLoader.
        ef_mean: training-set mean EF, for the loss.
        amp: run in float16.
        desc: label for the progress bar.

    Returns:
        predictions: DataFrame with fname, EF and EF_pred per video.
        loss: same loss as training (MSE / training mean EF).
        model_seconds: model time per video, warm-up skipped.
    """
    fnames, ef_true, outputs, model_seconds = common.predict_videos(model, loader, amp, desc)
    predictions = pd.DataFrame({'fname': fnames, 'EF': ef_true, 'EF_pred': outputs['EF'].ravel()})
    loss = float(np.mean((predictions['EF_pred'] - predictions['EF']) ** 2) / ef_mean)

    return predictions, loss, model_seconds


# ----------------------------------------------------------------------------
# build_results_table
# ----------------------------------------------------------------------------
def build_results_table(predictions, n_boot, seed):
    """Compute test metrics with bootstrap 95% CIs.

    Args:
        predictions: DataFrame with EF and EF_pred per video.
        n_boot: bootstrap resamples.
        seed: seed for the bootstrap resampling.

    Returns:
        List of table rows for common.format_metrics_table.
    """
    ef_true = predictions['EF'].values
    ef_pred = predictions['EF_pred'].values
    labels = (ef_true < common.LV_DYSFUNCTION_EF).astype(int)

    ef = common.with_bootstrap_ci(common.regression_metrics, [ef_true, ef_pred], n_boot, seed)
    # Lower predicted EF means more likely dysfunction, so the score is -EF_pred
    from_ef = common.with_bootstrap_ci(common.classification_metrics, [labels, -ef_pred], n_boot, seed)

    threshold = f'EF<{common.LV_DYSFUNCTION_EF}'
    return [
        ('ef_mae', 'EF MAE (EF points)', ef['MAE'], '.2f'),
        ('ef_rmse', 'EF RMSE (EF points)', ef['RMSE'], '.2f'),
        ('ef_r2', 'EF R^2', ef['R2'], '.3f'),
        ('ef_bias', 'EF bias (pred - true)', ef['Bias'], '.2f'),
        ('auroc_ef', f'{threshold} AUROC, predicted EF', from_ef['AUROC'], '.3f'),
        ('ap_ef', f'{threshold} AP,    predicted EF', from_ef['AP'], '.3f'),
    ]


# ----------------------------------------------------------------------------
# run_seed
# ----------------------------------------------------------------------------
def run_seed(seed, datasets, ef_mean, args, logger):
    """Fine-tune from the pretrained weights with one seed, then test the best epoch.

    Args:
        seed: seed for initialization order, shuffling, clip sampling and the bootstrap.
        datasets: dict split -> dataset.
        ef_mean: training-set mean EF.
        args: parsed command-line arguments.
        logger: common.RunLogger.

    Returns:
        dict with the metric rows, timings, epoch history and peak GPU memory of this seed.
    """
    common.set_seed(seed)
    seed_dir = os.path.join(args.out_dir, f'seed_{seed}')
    checkpoint_path = os.path.join(seed_dir, 'checkpoints', 'best.pt')
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    train_loader = common.make_loader(datasets['train'], args.batch_size, args.num_workers, train=True, seed=seed)
    val_loader = common.make_loader(datasets['val'], args.eval_batch_size, args.num_workers, seed=seed)
    test_loader = common.make_loader(datasets['test'], args.eval_batch_size, args.num_workers, seed=seed)

    common.log_section(logger, f'Seed {seed}')
    model = build_model(args.clip_len)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    # Training: keep the epoch with the lowest validation loss, as the authors do
    history, batch_seconds = [], []
    best_loss, best_epoch = float('inf'), 0
    complexity.reset_peak_memory()
    run_start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()
        train_loss, seconds = train_one_epoch(model, train_loader, optimizer, scaler, ef_mean, f'Seed {seed} | epoch {epoch}/{args.epochs} [train]', args)
        train_minutes = (time.perf_counter() - epoch_start) / 60
        batch_seconds += seconds

        val_start = time.perf_counter()
        val_predictions, val_loss, _ = evaluate(model, val_loader, ef_mean, args.amp, desc=f'Seed {seed} | epoch {epoch}/{args.epochs} [val]')
        val_minutes = (time.perf_counter() - val_start) / 60
        val_mae = float(np.mean(np.abs(val_predictions['EF_pred'] - val_predictions['EF'])))

        improved = val_loss < best_loss
        if improved:
            best_loss, best_epoch = val_loss, epoch
            torch.save({'weights': model.state_dict(), 'epoch': epoch, 'seed': seed, 'val_loss': val_loss}, checkpoint_path)

        history.append({'epoch': epoch, 'train_loss': train_loss, 'val_loss': val_loss, 'val_mae': val_mae, 'train_minutes': train_minutes, 'val_minutes': val_minutes})
        logger.log(f' Epoch {epoch}/{args.epochs} | train loss {train_loss:.4f} | val loss {val_loss:.4f} | val MAE {val_mae:.2f} | {train_minutes + val_minutes:.1f} min' + (' | saved best' if improved else ''))

    train_peak_gb = complexity.peak_memory_gb()
    training_minutes = (time.perf_counter() - run_start) / 60
    pd.DataFrame(history).to_csv(os.path.join(seed_dir, 'history.csv'), index=False)

    # Test with the best weights
    model.load_state_dict(torch.load(checkpoint_path, map_location='cuda', weights_only=True)['weights'])
    complexity.reset_peak_memory()
    test_start = time.perf_counter()
    predictions, _, model_seconds = evaluate(model, test_loader, ef_mean, args.amp, desc=f'Seed {seed} | test')
    test_seconds = time.perf_counter() - test_start
    test_peak_gb = complexity.peak_memory_gb()
    predictions.to_csv(os.path.join(seed_dir, 'predictions.csv'), index=False)

    rows = build_results_table(predictions, args.n_boot, seed)
    table = common.format_metrics_table(rows)
    logger.log('')
    logger.log(f' Test split, best epoch {best_epoch} (N={len(predictions):,})')
    for line in table:
        logger.log(line)
    common.write_lines(os.path.join(seed_dir, 'metrics.txt'), [f' PanEcho fine-tuned ({args.epochs} epochs, best epoch {best_epoch}), EchoNet-Dynamic test split (N={len(predictions):,}), seed {seed}', ''] + table)

    del model, optimizer
    torch.cuda.empty_cache()

    return {
        'rows': rows,
        'batch_seconds': batch_seconds,
        'history': history,
        'training_minutes': training_minutes,
        'model_seconds': model_seconds,
        'test_seconds_per_video': test_seconds / len(predictions),
        'train_peak_gb': train_peak_gb,
        'test_peak_gb': test_peak_gb,
    }


# ----------------------------------------------------------------------------
# measure_model_size
# ----------------------------------------------------------------------------
def measure_model_size(clip_len):
    """Parameters and GFLOPs (forward and one training step) on a single clip.

    Args:
        clip_len: number of frames per clip.

    Returns:
        dict with parameter counts and GFLOPs.
    """
    model = build_model(clip_len)
    clip = torch.zeros(1, 3, clip_len, 224, 224, device='cuda')

    size = {
        'params': complexity.count_parameters(model),
        'params_trainable': complexity.count_parameters(model, trainable_only=True),
        'gflops_forward': complexity.count_gflops(model, clip),
        'gflops_train': complexity.count_gflops(model, clip, backward=True),
    }

    del model
    torch.cuda.empty_cache()

    return size


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------
def main(args):
    args.amp = not args.fp32
    logger = common.RunLogger(os.path.join(args.out_dir, 'run.log'))

    datasets = {
        'train': common.build_dataset('train', args.clip_len),
        'val': common.build_dataset('val', args.clip_len, args.num_clips, args.limit_eval_videos),
        'test': common.build_dataset('test', args.clip_len, args.num_clips, args.limit_eval_videos),
    }
    ef_mean = common.split_mean_ef('train')

    common.log_header(logger, 'PanEcho fine-tuning: EchoNet-Dynamic EF', {
        'Data': f'train {len(datasets["train"]):,} | val {len(datasets["val"]):,} | test {len(datasets["test"]):,} videos',
        'Seeds': ', '.join(str(s) for s in args.seeds),
        'Epochs': args.epochs,
        'Batch': f'{args.batch_size} x {args.accum_steps} accumulated = {args.batch_size * args.accum_steps} videos/step',
        'Optimizer': f'Adam, lr {args.lr:g}, no weight decay',
        'Loss': f'MSE / train mean EF ({ef_mean:.2f})',
        'Clips (val/test)': f'{args.num_clips} x {args.clip_len} frames',
        'Precision': 'float32' if args.fp32 else 'float16 (autocast + GradScaler)',
        'Bootstrap': f'{args.n_boot} resamples',
        'Output': os.path.relpath(args.out_dir, common.REPO_DIR),
    })

    # Model size and FLOPs, measured once (identical for every seed)
    logger.log(' Measuring model size and FLOPs...')
    common.set_seed(args.seeds[0])
    size = measure_model_size(args.clip_len)

    # One full fine-tuning run per seed
    runs = {seed: run_seed(seed, datasets, ef_mean, args, logger) for seed in args.seeds}

    # Mean +/- SD across seeds
    summary, summary_table = common.summarize_seeds({seed: run['rows'] for seed, run in runs.items()})
    common.log_section(logger, f'Summary: mean +/- SD across {len(args.seeds)} seeds')
    for line in summary:
        logger.log(line)
    summary_table.to_csv(os.path.join(args.out_dir, 'summary.csv'), index=False)
    common.write_lines(os.path.join(args.out_dir, 'summary.txt'), [f' PanEcho fine-tuned ({args.epochs} epochs), EchoNet-Dynamic test split (N={len(datasets["test"]):,}), mean +/- SD across seeds {", ".join(str(s) for s in args.seeds)}', ''] + summary)

    # Complexity
    batch_seconds = [s for run in runs.values() for s in run['batch_seconds']]
    model_seconds = [s for run in runs.values() for s in run['model_seconds']]
    report = complexity.format_report([
        ('Hardware', list(common.environment_info().items())),
        ('Model size', [
            ('Parameters (EF head)', f'{size["params"] / 1e6:.2f} M'),
            ('Trainable parameters', f'{size["params_trainable"] / 1e6:.2f} M'),
        ]),
        ('Compute', [
            (f'Forward GFLOPs per clip ({args.clip_len} frames)', f'{size["gflops_forward"]:.1f}'),
            (f'Forward GFLOPs per test video ({args.num_clips} clips)', f'{size["gflops_forward"] * args.num_clips:.1f}'),
            ('Training-step GFLOPs per video', f'{size["gflops_train"]:.1f} (forward + backward, 1 clip)'),
        ]),
        ('Fine-tuning time (all seeds)', [
            (f'Time per batch ({args.batch_size} videos)', complexity.summarize_ms(batch_seconds)),
            ('Time per epoch (train + val)', f'{np.mean([h["train_minutes"] + h["val_minutes"] for run in runs.values() for h in run["history"]]):.1f} min mean'),
            ('Fine-tuning time per seed', ', '.join(f'{run["training_minutes"]:.1f} min' for run in runs.values())),
        ]),
        ('Test inference time (all seeds)', [
            ('Model time per video', complexity.summarize_ms(model_seconds)),
            ('Wall time per video (with decoding)', f'{np.mean([run["test_seconds_per_video"] for run in runs.values()]) * 1000:.1f} ms'),
        ]),
        ('GPU memory (max over seeds)', [
            ('Peak allocated (training)', f'{max(run["train_peak_gb"] for run in runs.values()):.2f} GB'),
            ('Peak allocated (test inference)', f'{max(run["test_peak_gb"] for run in runs.values()):.2f} GB'),
        ]),
    ])
    common.log_section(logger, 'Complexity')
    for line in report:
        logger.log(line)
    common.write_lines(os.path.join(args.out_dir, 'complexity.txt'), report)

    logger.log(f' Saved: {os.path.relpath(args.out_dir, common.REPO_DIR)}{os.sep}{{summary.txt, summary.csv, complexity.txt, run.log, seed_*/}}')
    logger.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fine-tune PanEcho on EchoNet-Dynamic EF and evaluate on the test split.')
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2], help='one fine-tuning run per seed, e.g. --seeds 0 1 2')
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=4, help='videos per batch on the GPU')
    parser.add_argument('--accum_steps', type=int, default=32, help='batches accumulated per optimizer step')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--clip_len', type=int, default=16)
    parser.add_argument('--num_clips', type=int, default=4, help='clips per val/test video, predictions are averaged')
    parser.add_argument('--eval_batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--n_boot', type=int, default=1000, help='bootstrap resamples for the 95%% CIs')
    parser.add_argument('--fp32', action='store_true', default=False, help='train in float32 instead of float16')
    parser.add_argument('--limit_train_batches', type=int, default=0, help='train on only N batches per epoch (0 = all), for test runs')
    parser.add_argument('--limit_eval_videos', type=int, default=0, help='evaluate only the first N val/test videos (0 = all), for test runs')
    parser.add_argument('--out_dir', type=str, default=os.path.join(common.RESULTS_DIR, 'finetune'))

    args = parser.parse_args()

    main(args)
