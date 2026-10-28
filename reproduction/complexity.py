"""Model size, FLOPs, timing and GPU memory measurements shared by the evaluation scripts."""
import time

import numpy as np
import torch

from torch.utils.flop_counter import FlopCounterMode


# ----------------------------------------------------------------------------
# count_parameters
# ----------------------------------------------------------------------------
def count_parameters(model, trainable_only=False):
    """Count the parameters of a model.

    Args:
        model: any torch.nn.Module.
        trainable_only: count only parameters with requires_grad=True.

    Returns:
        Number of parameters (int).
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad or not trainable_only)


# ----------------------------------------------------------------------------
# count_gflops
# ----------------------------------------------------------------------------
def count_gflops(model, x, backward=False):
    """Measure the FLOPs of one forward pass (plus the backward pass if requested).

    Multiply and add are counted separately (1 MAC = 2 FLOPs). Gradients stay
    enabled even for a forward-only count, because PyTorch's fused Transformer
    fast path (which the counter cannot see) only runs with gradients disabled.

    PyTorch's counter ignores `groups` in the convolution backward, which
    overcounts ConvNeXt's depthwise convolutions about 100-fold. The backward of
    a convolution is therefore counted as 2x its forward (input and weight
    gradients), the same rule the counter applies correctly to linear layers.

    Args:
        model: PanEcho model whose forward returns a dict of task outputs.
        x: input clip(s), shape (B, 3, clip_len, H, W), on the model's device.
        backward: also count the backward pass, i.e. one training step.

    Returns:
        GFLOPs for the given input (float).
    """
    with torch.enable_grad(), FlopCounterMode(display=False) as counter:
        outputs = model(x)
        if backward:
            sum(out.float().sum() for out in outputs.values()).backward()

    model.zero_grad(set_to_none=True)

    flops = {str(op): count for op, count in counter.get_flop_counts()['Global'].items()}
    total = counter.get_total_flops()
    if backward:
        total += 2 * flops.get('aten.convolution', 0) - flops.get('aten.convolution_backward', 0)

    return total / 1e9


# ----------------------------------------------------------------------------
# CudaTimer
# ----------------------------------------------------------------------------
class CudaTimer:
    """Wall-clock timer that waits for queued GPU work before reading the clock.

    Usage:
        with CudaTimer() as timer:
            model(x)
        print(timer.seconds)
    """
    def __enter__(self):
        torch.cuda.synchronize()
        self.start = time.perf_counter()
        return self

    def __exit__(self, *exc):
        torch.cuda.synchronize()
        self.seconds = time.perf_counter() - self.start


# ----------------------------------------------------------------------------
# summarize_ms
# ----------------------------------------------------------------------------
def summarize_ms(seconds):
    """Format a list of durations as median and mean milliseconds.

    Args:
        seconds: list of durations in seconds.

    Returns:
        Text such as '45.1 ms median | 46.0 ms mean (n=318)', or 'n/a' if empty.
    """
    if len(seconds) == 0:
        return 'n/a'

    ms = np.array(seconds) * 1000

    return f'{np.median(ms):.1f} ms median | {ms.mean():.1f} ms mean (n={ms.size})'


# ----------------------------------------------------------------------------
# reset_peak_memory / peak_memory_gb
# ----------------------------------------------------------------------------
def reset_peak_memory():
    """Start a new peak GPU memory measurement."""
    torch.cuda.reset_peak_memory_stats()


def peak_memory_gb():
    """Peak GPU memory allocated by PyTorch since the last reset, in GB."""
    return torch.cuda.max_memory_allocated() / 1024 ** 3


# ----------------------------------------------------------------------------
# format_report
# ----------------------------------------------------------------------------
def format_report(sections):
    """Format complexity measurements as aligned text lines.

    Args:
        sections: list of (section title, list of (label, value text)).

    Returns:
        List of text lines, ready to print or write to complexity.txt.
    """
    lines = []
    for title, rows in sections:
        lines.append(f' {title}')
        for label, value in rows:
            lines.append(f'   {label:<42}{value}')
        lines.append('')

    return lines
