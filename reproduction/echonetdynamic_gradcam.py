"""Grad-CAM overlays for PanEcho EF predictions on selected EchoNet-Dynamic test videos.

Picks the best, median and worst test predictions of one run, then shows where
the zero-shot and fine-tuned models look on the same videos. The map is the
Grad-CAM of predicted EF at the last ConvNeXt stage of the image encoder,
normalized per frame, so it shows relative attention rather than absolute size.
"""
import argparse
import os

import cv2
import numpy as np
import pandas as pd
import torch

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torchvision import tv_tensors

import common

# Grad-CAM overlay color (#e34948 as BGR) and its maximum opacity
OVERLAY_BGR = np.array([72, 73, 227], dtype=np.float32)
OVERLAY_STRENGTH = 0.6

# Preprocessing geometry of the authors' loader: resize to 256, center crop to 224
RESIZE, CROP = 256, 224

LABELS = {'zeroshot': 'Zero-shot', 'finetune': 'Fine-tuned'}


# ----------------------------------------------------------------------------
# FramesToEF
# ----------------------------------------------------------------------------
class FramesToEF(torch.nn.Module):
    """Wrap PanEcho so Grad-CAM can pass one clip as a batch of frames.

    pytorch-grad-cam treats a 5D input as a 3D volume, but PanEcho's image
    encoder makes one 2D feature map per frame. Passing the clip as
    (clip_len, 3, H, W) gives one Grad-CAM map per frame of the video-level EF.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, frames):
        x = frames.permute(1, 0, 2, 3).unsqueeze(0)  # (F, 3, H, W) -> (1, 3, F, H, W)
        return self.model(x)['EF']  # (1, 1)


# ----------------------------------------------------------------------------
# select_videos
# ----------------------------------------------------------------------------
def select_videos(predictions_path, per_group):
    """Pick the best, median and worst predictions by absolute EF error.

    Args:
        predictions_path: predictions.csv of the run used for selection.
        per_group: videos per group.

    Returns:
        DataFrame with group, fname, EF, EF_pred and abs_error.
    """
    df = pd.read_csv(predictions_path)
    df['abs_error'] = (df['EF_pred'] - df['EF']).abs()
    df = df.sort_values('abs_error').reset_index(drop=True)

    middle = len(df) // 2 - per_group // 2
    groups = {
        'best': df.iloc[:per_group],
        'median': df.iloc[middle:middle + per_group],
        'worst': df.iloc[-per_group:].iloc[::-1],
    }

    return pd.concat([g.assign(group=name) for name, g in groups.items()], ignore_index=True)[['group', 'fname', 'EF', 'EF_pred', 'abs_error']]


# ----------------------------------------------------------------------------
# load_models
# ----------------------------------------------------------------------------
def load_models(checkpoint_path, clip_len):
    """Load the zero-shot model and, if its checkpoint exists, the fine-tuned model.

    Returns:
        dict run name -> PanEcho (EF head only) on the GPU in eval mode.
    """
    models = {'zeroshot': common.load_panecho(['EF'], clip_len).cuda().eval()}

    if os.path.exists(checkpoint_path):
        model = common.load_panecho(['EF'], clip_len)
        model.load_state_dict(torch.load(checkpoint_path, map_location='cpu', weights_only=True)['weights'])
        models['finetune'] = model.cuda().eval()
    else:
        print(f' Fine-tuned checkpoint not found ({os.path.relpath(checkpoint_path, common.REPO_DIR)}): zero-shot only.')

    return models


# ----------------------------------------------------------------------------
# load_clip
# ----------------------------------------------------------------------------
def load_clip(video_path, clip_len):
    """Read the clip_len frames at the center of a video.

    Short videos are padded by repeating the last frame, as in the authors' loader.

    Returns:
        Original frames, shape (clip_len, H, W, 3), uint8 BGR.
    """
    capture = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    capture.release()

    start = max(0, (len(frames) - clip_len) // 2)
    clip = frames[start:start + clip_len]
    clip += [clip[-1]] * (clip_len - len(clip))

    return np.stack(clip)


# ----------------------------------------------------------------------------
# preprocess
# ----------------------------------------------------------------------------
def preprocess(frames, transform):
    """Apply the authors' test preprocessing to original frames.

    Args:
        frames: original frames, (F, H, W, 3) uint8 BGR.
        transform: the test transform of the authors' dataset (center crop, ImageNet normalization).

    Returns:
        Model input frames, (F, 3, 224, 224) float tensor.
    """
    resized = np.stack([cv2.resize(f, (RESIZE, RESIZE), interpolation=cv2.INTER_AREA) for f in frames])

    return transform(tv_tensors.Video(np.transpose(resized, (0, 3, 1, 2))))


# ----------------------------------------------------------------------------
# compute_gradcam
# ----------------------------------------------------------------------------
def compute_gradcam(model, frames):
    """Grad-CAM of predicted EF for every frame of one clip.

    Args:
        model: PanEcho (EF head) on the GPU.
        frames: model input frames, (F, 3, 224, 224).

    Returns:
        maps: per-frame maps in [0, 1], shape (F, 224, 224).
        ef_pred: predicted EF for this clip.
    """
    wrapper = FramesToEF(model).eval()
    target_layer = model.encoder.encoder.model.features[-1]  # last ConvNeXt stage, 7x7 maps

    with GradCAM(model=wrapper, target_layers=[target_layer]) as cam:
        maps = cam(input_tensor=frames.cuda(), targets=[ClassifierOutputTarget(0)])

    with torch.no_grad():
        ef_pred = wrapper(frames.cuda()).item()

    return maps, ef_pred


# ----------------------------------------------------------------------------
# map_to_original
# ----------------------------------------------------------------------------
def map_to_original(cam, frame_shape):
    """Undo the center crop and resize so a 224x224 map lines up with the original frame.

    Args:
        cam: Grad-CAM map for the 224x224 model input.
        frame_shape: shape of the original frame (H, W, ...).

    Returns:
        Map with the original frame's height and width.
    """
    canvas = np.zeros((RESIZE, RESIZE), dtype=np.float32)
    offset = (RESIZE - CROP) // 2
    canvas[offset:offset + CROP, offset:offset + CROP] = cam

    return cv2.resize(canvas, (frame_shape[1], frame_shape[0]), interpolation=cv2.INTER_LINEAR)


# ----------------------------------------------------------------------------
# overlay
# ----------------------------------------------------------------------------
def overlay(frame, cam):
    """Blend the Grad-CAM map onto a frame: one color, opacity proportional to the map."""
    alpha = (cam * OVERLAY_STRENGTH)[..., None]
    blended = frame.astype(np.float32) * (1 - alpha) + OVERLAY_BGR * alpha

    return blended.clip(0, 255).astype(np.uint8)


# ----------------------------------------------------------------------------
# labeled_panel
# ----------------------------------------------------------------------------
def labeled_panel(image, title, height):
    """Resize an image to the given height and add a white title band on top."""
    scale = height / image.shape[0]
    image = cv2.resize(image, (round(image.shape[1] * scale), height), interpolation=cv2.INTER_LINEAR)

    band = np.full((26, image.shape[1], 3), 255, dtype=np.uint8)
    cv2.putText(band, title, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (11, 11, 11), 1, cv2.LINE_AA)

    return np.vstack([band, image])


# ----------------------------------------------------------------------------
# save_video / save_strip
# ----------------------------------------------------------------------------
def save_video(path, rows, titles, height, fps):
    """Save panels side by side as an MP4, one panel per sequence of frames.

    Args:
        path: output .mp4 path.
        rows: list of frame sequences, each (F, H, W, 3) uint8 BGR.
        titles: one title per sequence.
        height: panel height in pixels.
        fps: playback frames per second.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    frames = [np.hstack([labeled_panel(seq[i], title, height) for seq, title in zip(rows, titles)]) for i in range(len(rows[0]))]

    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frames[0].shape[1], frames[0].shape[0]))
    for frame in frames:
        writer.write(frame)
    writer.release()


def save_strip(path, rows, titles, height, num_frames=8):
    """Save evenly spaced frames as a PNG, one row per sequence, for reports."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    idx = np.linspace(0, len(rows[0]) - 1, num_frames).round().astype(int)

    strip_rows = []
    for seq, title in zip(rows, titles):
        images = [cv2.resize(seq[i], (round(seq[i].shape[1] * height / seq[i].shape[0]), height)) for i in idx]
        row = np.hstack(images)
        band = np.full((26, row.shape[1], 3), 255, dtype=np.uint8)
        cv2.putText(band, title, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (11, 11, 11), 1, cv2.LINE_AA)
        strip_rows.append(np.vstack([band, row]))

    cv2.imwrite(path, np.vstack(strip_rows))


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------
def main(args):
    common.set_seed(args.seed)
    checkpoint = args.checkpoint or os.path.join(common.RESULTS_DIR, 'finetune', f'seed_{args.seed}', 'checkpoints', 'best.pt')

    selection = select_videos(os.path.join(common.RESULTS_DIR, args.select_from, f'seed_{args.seed}', 'predictions.csv'), args.per_group)
    os.makedirs(args.out_dir, exist_ok=True)
    selection.to_csv(os.path.join(args.out_dir, 'selected_videos.csv'), index=False)

    print(f' Grad-CAM: {len(selection)} test videos selected from {LABELS[args.select_from]} predictions (seed {args.seed})')
    models = load_models(checkpoint, args.clip_len)
    transform = common.build_dataset('test', args.clip_len).transform

    for _, row in selection.iterrows():
        original = load_clip(os.path.join(common.DATA_DIR, 'Videos', f'{row["fname"]}.avi'), args.clip_len)
        frames = preprocess(original, transform)

        overlays, preds = {}, {}
        for run, model in models.items():
            maps, preds[run] = compute_gradcam(model, frames)
            overlays[run] = np.stack([overlay(f, map_to_original(m, f.shape)) for f, m in zip(original, maps)])

        true_title = f'Original | true EF {row["EF"]:.1f}'
        for run in models:
            run_title = f'{LABELS[run]} | clip pred {preds[run]:.1f}'
            name = os.path.join(args.out_dir, run, row['group'], row['fname'])
            save_video(f'{name}.mp4', [original, overlays[run]], [true_title, run_title], args.panel_height, args.fps)
            save_strip(f'{name}.png', [original, overlays[run]], [true_title, run_title], args.strip_height)

        if len(models) == 2:
            name = os.path.join(args.out_dir, 'comparison', row['group'], row['fname'])
            rows = [original, overlays['zeroshot'], overlays['finetune']]
            titles = [true_title] + [f'{LABELS[run]} | clip pred {preds[run]:.1f}' for run in ['zeroshot', 'finetune']]
            save_video(f'{name}.mp4', rows, titles, args.panel_height, args.fps)
            save_strip(f'{name}.png', rows, titles, args.strip_height)

        print(f'   {row["group"]:<7}{row["fname"]}  true {row["EF"]:5.1f} | ' + ' | '.join(f'{LABELS[run]} {preds[run]:5.1f}' for run in models))

    print(f' Saved: {os.path.relpath(args.out_dir, common.REPO_DIR)}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Grad-CAM overlays for PanEcho EF on selected EchoNet-Dynamic test videos.')
    parser.add_argument('--select_from', type=str, default='finetune', choices=['finetune', 'zeroshot'], help='run whose predictions pick the best/median/worst videos')
    parser.add_argument('--per_group', type=int, default=3, help='videos per group (best, median, worst)')
    parser.add_argument('--checkpoint', type=str, default=None, help='fine-tuned weights (default: results/finetune/seed_<seed>/checkpoints/best.pt)')
    parser.add_argument('--clip_len', type=int, default=16)
    parser.add_argument('--panel_height', type=int, default=336, help='height of each video panel in pixels')
    parser.add_argument('--strip_height', type=int, default=160, help='height of each frame in the PNG strips')
    parser.add_argument('--fps', type=int, default=8, help='playback speed of the overlay videos')
    parser.add_argument('--seed', type=int, default=0, help="which seed's predictions and fine-tuned checkpoint to use")
    parser.add_argument('--out_dir', type=str, default=os.path.join(common.RESULTS_DIR, 'gradcam'))

    args = parser.parse_args()

    main(args)
