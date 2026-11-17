"""Code shared by the coordinator and every worker.

Everything here has to behave identically on every computer: building the model,
laying out MONAI's sliding-window patch grid, generating LocBAM's coordinate
channels, running a block of patches, and moving arrays over the network.
"""
from __future__ import annotations

import io
import os
import platform
import socket
import subprocess
import sys
import time
import warnings

import numpy as np
import torch
import torch.nn as nn

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from BasicUNet.LocBAM.LocBAM_1D import BasicUnetLocBAMs2  # noqa: E402

# Official BTCV inference settings (BasicUNet/BTCV_training_script.py)
FEATURES = (32, 32, 64, 128, 256, 32)
ROI_SIZE = (32, 32, 32)
OVERLAP = 0.25
SPACING = (1.5, 1.5, 2.0)
HU_WINDOW = (-175.0, 250.0)

# LocBAM's attention gates call dropout2d on 1D features; in eval mode it does nothing,
# but torch warns on every call.
warnings.filterwarnings("ignore", message="dropout2d: Received a 3D input")


# ----------------------------------------------------------------------------- device

def pick_device(name: str = "auto") -> torch.device:
    """CUDA GPU, then Apple GPU (MPS), then CPU, unless a device is named explicitly."""
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


class _AxisPool(nn.Module):
    """Stand-in for nn.AdaptiveAvg/MaxPool3d when every output size is 1 or None (LocBAM only
    pools whole axes away). MPS has no adaptive 3D pooling. Max is bitwise identical; mean is the
    same mean summed in a different order, so it can differ in the last float digit (~1e-7)."""

    def __init__(self, output_size, reduce: str):
        super().__init__()
        self.dims = tuple(2 + i for i, s in enumerate(output_size) if s == 1)
        self.reduce = reduce

    def forward(self, x):
        if self.reduce == "mean":
            return x.mean(dim=self.dims, keepdim=True)
        return x.amax(dim=self.dims, keepdim=True)


class _MaxPool2(nn.Module):
    """Bitwise-identical stand-in for nn.MaxPool3d(kernel_size=2, stride=2): the max of each
    2x2x2 block. MPS in torch < 2.9 has no 3D max pooling."""

    def forward(self, x):
        b, c, w, h, d = x.shape
        x = x[:, :, : w // 2 * 2, : h // 2 * 2, : d // 2 * 2]
        return x.reshape(b, c, w // 2, 2, h // 2, 2, d // 2, 2).amax(dim=(3, 5, 7))


def _is_plain_2x_maxpool(m: nn.MaxPool3d) -> bool:
    def triple(v):
        return tuple(v) if isinstance(v, (tuple, list)) else (v, v, v)
    return (triple(m.kernel_size) == (2, 2, 2) and triple(m.stride) == (2, 2, 2) and triple(m.padding) == (0, 0, 0)
            and triple(m.dilation) == (1, 1, 1) and not m.ceil_mode and not m.return_indices)


def apply_mps_shims(model: nn.Module) -> int:
    """Swap the pooling layers MPS can't run for equivalent versions (same math, equal up to float
    rounding). Weights are untouched."""
    replaced = 0
    for module in list(model.modules()):
        for name, child in list(module.named_children()):
            if isinstance(child, nn.AdaptiveAvgPool3d) and all(s in (None, 1) for s in child.output_size):
                setattr(module, name, _AxisPool(child.output_size, "mean"))
            elif isinstance(child, nn.AdaptiveMaxPool3d) and all(s in (None, 1) for s in child.output_size):
                setattr(module, name, _AxisPool(child.output_size, "max"))
            elif isinstance(child, nn.MaxPool3d) and _is_plain_2x_maxpool(child):
                setattr(module, name, _MaxPool2())
            else:
                continue
            replaced += 1
    return replaced


# ----------------------------------------------------------------------------- model

def load_state_dict(data: bytes) -> dict:
    """Read a checkpoint safely (tensors only, no arbitrary code)."""
    obj = torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
    for key in ("state_dict", "model", "model_state_dict"):
        if isinstance(obj, dict) and key in obj and isinstance(obj[key], dict):
            return obj[key]
    return obj


def build_model(state_dict: dict, device: torch.device) -> nn.Module:
    """LocBAM (BasicUnetLocBAMs2, official BTCV configuration) with the given weights, in eval mode."""
    num_classes = int(state_dict["final_conv.conv.weight"].shape[0])
    model = BasicUnetLocBAMs2(spatial_dims=3, in_channels=1, out_channels=num_classes,
                              features=FEATURES, dropout=0.1, LocBAM_params=None)
    model.load_state_dict(state_dict, strict=True)
    if device.type == "mps":
        apply_mps_shims(model)
    return model.to(device).eval()


# ----------------------------------------------------------------------------- patches and coordinates

def patch_grid(spatial_shape, roi_size=ROI_SIZE, overlap=OVERLAP):
    """The exact window layout of MONAI's sliding_window_inference.

    Returns (padded_shape, pad_before, starts): volumes smaller than the window are
    zero-padded first (half before, half after, like MONAI), and `starts` (N x 3) are
    patch corners in that padded frame, in MONAI's order.
    """
    from monai.data.utils import dense_patch_slices
    from monai.inferers.utils import _get_scan_interval

    spatial_shape = tuple(int(s) for s in spatial_shape)
    pad_before = tuple(max(r - s, 0) // 2 for s, r in zip(spatial_shape, roi_size))
    padded = tuple(max(s, r) for s, r in zip(spatial_shape, roi_size))
    interval = _get_scan_interval(padded, roi_size, len(roi_size), (overlap,) * len(roi_size))
    slices = dense_patch_slices(padded, roi_size, interval, return_slice=False)
    starts = np.array([[lo for lo, _ in s] for s in slices], dtype=np.int64)
    return padded, pad_before, starts


def coordinate_channels(offset, size, orig_shape, pad_before) -> np.ndarray:
    """The block [offset, offset + size) of MONAI's AddCoordinateChannels(spatial_dims=[0, 1, 2]).

    LocBAM's location context is linspace(-0.5, 0.5) along each axis of the *whole* scan,
    so any worker can rebuild it for its own block from the scan's shape alone; the
    coordinates never have to be sent. Padding voxels get 0 in every channel, as in
    sliding_window_inference (which pads after the coordinates are added).
    """
    axes, inside = [], []
    for o, n, s, p in zip(offset, size, orig_shape, pad_before):
        idx = np.arange(o, o + n) - p
        axes.append(np.linspace(-0.5, 0.5, s)[np.clip(idx, 0, s - 1)].astype(np.float32))
        inside.append((idx >= 0) & (idx < s))
    coords = np.stack(np.meshgrid(*axes, indexing="ij"))
    in_scan = np.logical_and.reduce(np.meshgrid(*inside, indexing="ij"))
    return np.where(in_scan, coords, np.float32(0.0)).astype(np.float32)


@torch.no_grad()
def infer_region(model, device, image, offset, starts, orig_shape, pad_before, batch_size,
                 roi_size=ROI_SIZE) -> torch.Tensor:
    """Run the model on every patch in `starts` and add up the outputs.

    `image` is the block of the (padded) preprocessed scan that begins at `offset`;
    `starts` are patch corners in the padded frame. Returns the (C, *image.shape)
    float32 sum of patch logits, accumulated in MONAI's order.
    """
    size = tuple(image.shape)
    x = torch.from_numpy(np.concatenate(
        [np.asarray(image, dtype=np.float32)[None], coordinate_channels(offset, size, orig_shape, pad_before)]))
    rel = np.asarray(starts) - np.asarray(offset)
    rw, rh, rd = roi_size
    acc = None
    for i in range(0, len(rel), batch_size):
        chunk = rel[i:i + batch_size]
        batch = torch.stack([x[:, a:a + rw, b:b + rh, c:c + rd] for a, b, c in chunk]).to(device)
        out = model(batch).float().cpu()
        if acc is None:
            acc = torch.zeros((out.shape[1],) + size, dtype=torch.float32)
        for (a, b, c), o in zip(chunk, out):
            acc[:, a:a + rw, b:b + rh, c:c + rd] += o
    if acc is None:
        raise ValueError("no patches to run")
    return acc


def run_job(model, device, batch_size, job: dict) -> dict:
    """A worker's whole task: run its patches, then return labels for its block plus raw
    logit sums only where its patches overlap another worker's (the coordinator adds those up)."""
    t0 = time.perf_counter()
    image = job["image"]
    acc = infer_region(model, device, image, tuple(job["offset"]), job["starts"],
                       tuple(job["orig_shape"]), tuple(job["pad_before"]), batch_size)
    shared = np.unpackbits(job["shared_bits"], count=image.size).reshape(image.shape).astype(bool)
    labels = acc.argmax(0).to(torch.uint8).numpy()
    shared_logits = np.ascontiguousarray(acc.numpy()[:, shared])
    return {"labels": labels, "shared_logits": shared_logits,
            "compute_s": np.array(time.perf_counter() - t0)}


# ----------------------------------------------------------------------------- benchmark and specs

def benchmark(model, device, batch_sizes=(8, 16, 32, 64, 128)) -> dict:
    """Measure this machine's real throughput rather than guess it from datasheet numbers.

    1. Find the fastest batch size (larger isn't always faster: on CPU it often gets slower).
    2. Time a full run of patches through infer_region, so patch gathering and summing count too.
    """
    by_bs = {}
    for bs in batch_sizes:
        try:
            x = torch.rand(bs, 4, *ROI_SIZE, device=device)
            with torch.no_grad():
                model(x)
                synchronize(device)
                t = time.perf_counter()
                model(x)
                synchronize(device)
            by_bs[bs] = bs / (time.perf_counter() - t)
        except RuntimeError:  # out of memory
            break
        if by_bs[bs] < 0.85 * max(by_bs.values()):
            break
    if not by_bs:
        raise RuntimeError("model does not run on this device")
    best = max(by_bs, key=lambda k: by_bs[k])

    n = max(2 * best, 32)
    shape = (ROI_SIZE[0] + (ROI_SIZE[0] - int(ROI_SIZE[0] * OVERLAP)) * (n - 1), ROI_SIZE[1], ROI_SIZE[2])
    _, _, starts = patch_grid(shape)
    image = np.random.default_rng(0).random(shape, dtype=np.float32)
    t = time.perf_counter()
    infer_region(model, device, image, (0, 0, 0), starts, shape, (0, 0, 0), best)
    return {"batch_size": best, "patches_per_s": len(starts) / (time.perf_counter() - t),
            "model_only_by_batch_size": {str(k): round(v, 2) for k, v in by_bs.items()}}


def _processor_name() -> str:
    try:
        if sys.platform == "darwin":
            return subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                                  capture_output=True, text=True, timeout=5).stdout.strip()
        if sys.platform.startswith("linux"):
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.startswith("model name"):
                        return line.split(":", 1)[1].strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return platform.processor() or platform.machine()


def machine_specs(device: torch.device) -> dict:
    import psutil

    vm = psutil.virtual_memory()
    specs = {
        "hostname": socket.gethostname(),
        "os": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cpu": _processor_name(),
        "cpu_cores": os.cpu_count(),
        "torch_threads": torch.get_num_threads(),
        "ram_total_gb": round(vm.total / 1e9, 1),
        "ram_available_gb": round(vm.available / 1e9, 1),
        "device": device.type,
        "device_name": _processor_name(),
    }
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        specs["device_name"] = props.name
        specs["device_memory_gb"] = round(props.total_memory / 1e9, 1)
    return specs


# ----------------------------------------------------------------------------- wire format

def pack(**arrays) -> bytes:
    buf = io.BytesIO()
    np.savez(buf, **{k: np.asarray(v) for k, v in arrays.items()})
    return buf.getvalue()


def unpack(data: bytes) -> dict:
    with np.load(io.BytesIO(data), allow_pickle=False) as npz:
        return {k: npz[k] for k in npz.files}
