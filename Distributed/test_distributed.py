"""Checks for distributed LocBAM inference. From the LocBAM_distributed folder:

    python -m Distributed.test_distributed

Runs on CPU in about a minute, with a randomly initialised LocBAM (no data or checkpoint needed).
"""
from __future__ import annotations

import itertools
import os
import sys

import numpy as np
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from monai.inferers import sliding_window_inference  # noqa: E402
from monai.transforms import AddCoordinateChannels  # noqa: E402

from BasicUNet.LocBAM.LocBAM_1D import BasicUnetLocBAMs2  # noqa: E402
from Distributed.common import (FEATURES, OVERLAP, ROI_SIZE, _AxisPool, _MaxPool2, apply_mps_shims,  # noqa: E402
                                build_model, coordinate_channels, pack, patch_grid, run_job, unpack)
from Distributed.coordinator import make_jobs, merge  # noqa: E402
from Distributed.planner import WorkerProfile, plan  # noqa: E402

SHAPES = [(70, 45, 33), (40, 20, 50)]  # the second is thinner than a patch, so it gets padded
CPU = torch.device("cpu")


def _random_model(num_classes: int = 14):
    torch.manual_seed(0)
    net = BasicUnetLocBAMs2(spatial_dims=3, in_channels=1, out_channels=num_classes, features=FEATURES,
                            dropout=0.1, LocBAM_params=None)
    return build_model(net.state_dict(), CPU)


def _pad(volume, padded, pad_before):
    out = np.zeros(padded, np.float32)
    out[tuple(slice(p, p + s) for p, s in zip(pad_before, volume.shape))] = volume
    return out


def test_planner_is_optimal():
    """Compare the greedy plan with brute force over every possible split."""
    rng = np.random.default_rng(0)
    for _ in range(300):
        k, n = int(rng.integers(1, 4)), int(rng.integers(0, 13))
        workers = [WorkerProfile(f"w{i}", patches_per_s=float(rng.uniform(0.5, 10)),
                                 bytes_per_s=float(rng.uniform(1e3, 1e5)), latency_s=float(rng.uniform(0, 2)),
                                 max_patches=None if rng.random() < 0.7 else int(rng.integers(0, 8)))
                   for i in range(k)]
        b = float(rng.uniform(0, 1e4))
        best = None
        for split in itertools.product(range(n + 1), repeat=k):
            if sum(split) != n or any(w.max_patches is not None and s > w.max_patches for w, s in zip(workers, split)):
                continue
            t = max(w.finish_time(s, b) for w, s in zip(workers, split))
            best = t if best is None else min(best, t)
        if best is None:
            try:
                plan(n, workers, b)
            except ValueError:
                continue
            raise AssertionError("plan() should refuse when memory caps can't hold all patches")
        p = plan(n, workers, b)
        assert sum(p.counts) == n
        assert all(w.max_patches is None or c <= w.max_patches for w, c in zip(workers, p.counts))
        assert abs(p.makespan_s - best) < 1e-9, (p.makespan_s, best)
        assert p.continuous_makespan_s <= p.makespan_s + 1e-9 or any(w.max_patches is not None for w in workers)


def test_coordinates_match_monai():
    """Coordinates rebuilt from a slab equal MONAI's AddCoordinateChannels on the whole scan."""
    for shape in SHAPES:
        padded, pad_before, _ = patch_grid(shape)
        full = np.asarray(AddCoordinateChannels(spatial_dims=(0, 1, 2))(np.zeros((1,) + shape, np.float32)))[1:]
        full_padded = np.zeros((3,) + padded, np.float32)
        full_padded[(slice(None),) + tuple(slice(p, p + s) for p, s in zip(pad_before, shape))] = full
        for offset in [(0, 0, 0), (5, 3, 7), (padded[0] - 9, 1, 0)]:
            size = tuple(min(9 + 2 * i, padded[i] - offset[i]) for i in range(3))
            want = full_padded[(slice(None),) + tuple(slice(o, o + s) for o, s in zip(offset, size))]
            got = coordinate_channels(offset, size, shape, pad_before)
            assert got.dtype == np.float32 and np.array_equal(got, want), (shape, offset)


def test_mps_shims_match():
    """Max pooling replacements are bitwise identical; the mean replacement sums in a different
    order, so the whole model agrees to float rounding."""
    x = torch.randn(4, 32, 16, 16, 16)
    assert torch.equal(torch.nn.MaxPool3d(2)(x), _MaxPool2()(x))
    for out in [(1, 1, None), (None, 1, 1), (1, None, 1)]:
        assert torch.equal(torch.nn.AdaptiveMaxPool3d(out)(x), _AxisPool(out, "max")(x))
    model = _random_model()
    shimmed = _random_model()
    assert apply_mps_shims(shimmed) > 0
    x = torch.rand(3, 4, *ROI_SIZE)
    with torch.no_grad():
        assert torch.allclose(model(x), shimmed(x), rtol=0, atol=1e-5)


def test_distributed_equals_single_machine():
    """Split across simulated workers (through the real wire format), merge, compare with
    MONAI's sliding_window_inference on the whole scan."""
    model = _random_model()
    rng = np.random.default_rng(1)
    for shape in SHAPES:
        volume = rng.random(shape, dtype=np.float32)
        padded, pad_before, starts = patch_grid(shape)
        x = np.asarray(AddCoordinateChannels(spatial_dims=(0, 1, 2))(volume[None]), dtype=np.float32)
        with torch.no_grad():
            logits = sliding_window_inference(torch.from_numpy(x)[None], ROI_SIZE, 8, model, overlap=OVERLAP)
        reference = torch.as_tensor(logits)[0].argmax(0).numpy()

        n = len(starts)
        splits = [[n], [n // 2, n - n // 2], [1, 0, n - 1], [n // 3, n // 3, n - 2 * (n // 3)]]
        for counts in splits:
            jobs, owners, layout = make_jobs(_pad(volume, padded, pad_before), shape, pad_before, starts, counts)
            results = [None if job is None else unpack(pack(**run_job(model, CPU, 8, unpack(pack(**job)))))
                       for job in jobs]
            labels = merge(results, layout, owners, 14)
            labels = labels[tuple(slice(p, p + s) for p, s in zip(pad_before, shape))]
            differ = int((labels != reference).sum())
            assert differ == 0, f"{shape} split {counts}: {differ} voxels differ"


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"all {len(tests)} tests passed")
