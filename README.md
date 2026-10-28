# PanEcho on EchoNet-Dynamic: reproduction

Reproduces PanEcho's EchoNet-Dynamic results with the authors' code and released
weights ([JAMA 2025](https://doi.org/10.1001/jama.2025.8731)), in two settings:

- **Zero-shot**: the pretrained model as released, no training.
- **Fine-tuned**: the pretrained model fine-tuned for 2 epochs on the EchoNet-Dynamic training split.

Both are run with **3 seeds (0, 1, 2)** and reported as mean +/- SD across seeds,
next to each seed's own value and 95% bootstrap CI. The authors' code in `../src/`
and `../hubconf.py` is used unchanged.

## Environment

Conda environment `myenv2` (Python 3.8, PyTorch 2.2.2 + CUDA 11.8, torchvision 0.17,
timm, OpenCV, pandas, scikit-learn, matplotlib, tqdm, grad-cam). The first run
downloads the PanEcho weights from GitHub into the PyTorch hub cache.

## Run order

Run every command from this folder (`reproduction/`) with `myenv2` active.

| Step | Command | Time (RTX 3070 Laptop, estimated) |
|---|---|---|
| 1. Zero-shot, 3 seeds | `python echonetdynamic_zeroshot.py` | about 30 min |
| 2. Fine-tune and test, 3 seeds | `python echonetdynamic_finetune.py` | about 1 h 45 min |
| 3. Figures | `python echonetdynamic_figures.py` | about 1 min |
| 4. Grad-CAM (seed 0) | `python echonetdynamic_gradcam.py` | a few min |

Other seeds: `--seeds 0 1 2 3 4`. Short test runs, to check that everything works
before the full runs:

```
python echonetdynamic_zeroshot.py --seeds 0 --num_videos 10
python echonetdynamic_finetune.py --seeds 0 --limit_train_batches 40 --limit_eval_videos 10
```

Test runs write to the same result folders as full runs, so run the full
commands afterwards to replace their outputs.

## Outputs

```
results/
├── zeroshot/
│   ├── summary.txt        mean +/- SD across seeds, and each seed's value
│   ├── summary.csv        the same, one row per seed and metric (with 95% CI)
│   ├── complexity.txt     parameters, GFLOPs, time, GPU memory
│   ├── run.log            full terminal output
│   ├── seed_0/            predictions.csv, metrics.txt
│   ├── seed_1/
│   └── seed_2/
├── finetune/
│   ├── summary.txt, summary.csv, complexity.txt, run.log
│   ├── seed_0/            predictions.csv, metrics.txt, history.csv, checkpoints/best.pt
│   ├── seed_1/
│   └── seed_2/
├── figures/
│   ├── zeroshot/
│   │   ├── ef_scatter/            predicted vs. true EF                  (seed 0)
│   │   ├── bland_altman/          agreement, bias, limits of agreement   (seed 0)
│   │   ├── error_by_ef_range/     MAE per true-EF range, 95% CI          (seed 0)
│   │   ├── lv_dysfunction_roc/    ROC and precision-recall               (seed 0)
│   │   └── seed_variability/      each seed's metrics, mean +/- SD       (all seeds)
│   ├── finetune/
│   │   ├── same five folders
│   │   └── training_curves/       loss and validation MAE per epoch     (all seeds)
│   └── comparison/                zero-shot vs. fine-tuned: scatter, Bland-Altman,
│                                  error by range, ROC, PR, metrics, seed variability,
│                                  and cost (compute, GPU memory, fine-tuning and inference time)
└── gradcam/
    ├── selected_videos.csv        the 9 videos and why they were picked
    ├── zeroshot/                  best/, median/, worst/  (one .mp4 and one .png per video)
    ├── finetune/                  same 9 videos
    └── comparison/                original | zero-shot | fine-tuned
```

Every figure carries a footnote naming its split and seed(s).

## Protocol

- **Data**: official EchoNet-Dynamic splits from `FileList.csv` (test: 1,277 videos).
- **Preprocessing** (the authors' loader): resize to 256x256, center crop to 224x224, ImageNet normalization.
- **Prediction**: 4 random 16-frame clips per video, outputs averaged (the authors' evaluation).
- **Seeds**: each seed sets PyTorch, NumPy and Python seeding (the authors' `set_seed`), the
  training shuffle, and the clip sampling. For val/test, each loader worker is seeded with
  `worker id + 1000 x seed`, so seed 0 matches the authors' `val_worker_init_fn` exactly.
  Zero-shot seeds differ only in the sampled clips; fine-tuning seeds also differ in training.
- **Metrics**: EF MAE, RMSE, R^2 and bias, with 95% bootstrap CIs (1,000 resamples) per seed;
  mean +/- sample SD (n - 1) across seeds.
- **LV dysfunction**: true EF < 40% (moderately reduced or worse). This threshold is our
  assumption for the paper's "moderate or severe LV systolic dysfunction". Zero-shot AUROC
  is reported from the model's LV systolic function head and from predicted EF.
- **Complexity**: parameters, GFLOPs (PyTorch FLOP counter, 1 MAC = 2 FLOPs; the depthwise
  convolution backward is counted as 2x its forward, see `complexity.py`), time per video and
  per training batch (GPU synchronized, warm-up batches skipped), peak GPU memory.

## Differences from the authors' fine-tuning

| | Authors (`src/train_echonetdynamic.py`) | Here |
|---|---|---|
| Hardware | 8 GPUs, Linux | 1 GPU (8 GB), Windows |
| Batch | 16 videos x 8 GPUs = 128 per step | 4 videos x 32 accumulated = 128 per step |
| Mixed precision | float16 autocast | float16 autocast + GradScaler (prevents gradient underflow) |
| Loss scaling mean | shared task object (ends up as the test-split mean) | training-split mean |
| Epochs | early stopping, up to 30 | 2, best epoch by validation loss |
| Seeds | 1 | 3 (0, 1, 2) |

Everything else follows the authors' command: Adam, learning rate 1e-4, no weight
decay, no augmentation, head dropout 0, MSE loss divided by the mean EF.

## Grad-CAM

The 9 videos (3 best, 3 median, 3 worst by absolute error) are picked from the
seed 0 fine-tuned test predictions and shown for both models (zero-shot weights
and the seed 0 fine-tuned checkpoint). Maps are the Grad-CAM of predicted EF at
the last ConvNeXt stage, computed on the center 16-frame clip, normalized per
frame, and mapped back onto the original 112x112 frames. They show where each
model looks, not why its prediction is right.
