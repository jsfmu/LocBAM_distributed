"""Split one LocBAM segmentation across several computers.

    python -m Distributed.coordinator --image scan.nii.gz --checkpoint model.pth \\
        --workers 192.168.1.20:8765 192.168.1.21:8765 --out prediction.nii.gz [--verify]

1. Contact every worker: send it the model weights (once), read its specs, and let it
   benchmark itself. Measure the network link to it (latency and bandwidth).
2. Preprocess the scan once, here (HU window, foreground crop, RAS, 1.5 x 1.5 x 2 mm).
3. Lay out MONAI's sliding-window patch grid and decide how many patches each worker
   runs (Distributed/planner.py). Each worker gets a contiguous run of patches, which
   is one compact slab of the scan, so it only needs that slab.
4. Send every worker its slab and patch list; they all run at the same time.
5. Merge. Voxels that only one worker's patches cover take that worker's labels; where
   two workers' patches overlap, their logit sums are added before the argmax. The result
   is the same as running MONAI's sliding_window_inference on one machine.
6. Map the labels back onto the original scan's voxel grid and save them as NIfTI.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import nibabel as nib
import numpy as np
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from Distributed.common import (HU_WINDOW, OVERLAP, ROI_SIZE, SPACING, build_model, load_state_dict,  # noqa: E402
                                pack, patch_grid, pick_device, unpack)
from Distributed.planner import WorkerProfile, plan  # noqa: E402

# Memory a worker needs per voxel of its slab: the received slab, 3 coordinate channels,
# the 4-channel model input, the float32 logit sum per class, labels and mask.
def _worker_bytes_per_voxel(num_classes: int) -> int:
    return 4 + 12 + 16 + 4 * num_classes + 2


class WorkerError(RuntimeError):
    pass


class WorkerClient:
    def __init__(self, address: str, token: str | None):
        self.address = address
        self.url = address if address.startswith("http") else f"http://{address}"
        self.token = token

    def _request(self, path: str, body: bytes | None = None, timeout: float = 60.0) -> bytes:
        headers = {"Content-Type": "application/octet-stream"}
        if self.token:
            headers["X-LocBAM-Token"] = self.token
        request = urllib.request.Request(self.url + path, data=body, headers=headers,
                                         method="GET" if body is None else "POST")
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read()
        except urllib.error.HTTPError as e:
            detail = e.read().decode(errors="replace")
            try:
                detail = json.loads(detail).get("error", detail)
            except ValueError:
                pass
            raise WorkerError(f"{self.address}: {detail}") from None
        except (urllib.error.URLError, OSError) as e:
            raise WorkerError(f"{self.address}: can't reach the worker ({e})") from None

    def info(self) -> dict:
        return json.loads(self._request("/info", timeout=30))

    def load_model(self, checkpoint: bytes) -> dict:
        return json.loads(self._request("/model", checkpoint, timeout=900))

    def measure_link(self, n_bytes: int = 8_000_000):
        """(round-trip latency in s, upload bandwidth in bytes/s)."""
        rtts = []
        for _ in range(5):
            t = time.perf_counter()
            self._request("/ping", timeout=30)
            rtts.append(time.perf_counter() - t)
        latency = sorted(rtts)[len(rtts) // 2]
        t = time.perf_counter()
        self._request("/echo", os.urandom(n_bytes), timeout=600)
        return latency, n_bytes / max(time.perf_counter() - t - latency, 1e-6)

    def infer(self, job: dict) -> dict:
        return unpack(self._request("/infer", pack(**job), timeout=24 * 3600))


def connect(client: WorkerClient, checkpoint: bytes, sha256: str) -> dict:
    info = client.info()
    if info.get("model_sha256") != sha256 or not info.get("benchmark"):
        info = client.load_model(checkpoint)
    if info.get("model_sha256") != sha256:
        raise WorkerError(f"{client.address}: model checksum mismatch after upload")
    info["latency_s"], info["bytes_per_s"] = client.measure_link()
    return info


# ----------------------------------------------------------------------------- preprocessing

def preprocess(image_path: str):
    """The official validation pipeline, applied once to the whole scan (coordinates are
    added later, by each worker, for its own slab)."""
    from monai.transforms import Compose, CropForegroundd, LoadImaged, Orientationd, ScaleIntensityRanged, Spacingd

    transforms = Compose([
        LoadImaged(keys="image", ensure_channel_first=True, image_only=True),
        ScaleIntensityRanged(keys="image", a_min=HU_WINDOW[0], a_max=HU_WINDOW[1], b_min=0.0, b_max=1.0, clip=True),
        CropForegroundd(keys="image", source_key="image", allow_smaller=False),
        Orientationd(keys="image", axcodes="RAS"),
        Spacingd(keys="image", pixdim=SPACING, mode="bilinear"),
    ])
    image = transforms({"image": image_path})["image"]
    return transforms, image


def to_original_space(labels: np.ndarray, image, transforms) -> np.ndarray:
    """Undo resampling, reorientation and cropping, so labels line up with the original scan."""
    from monai.data import MetaTensor
    from monai.transforms import Invertd

    pred = MetaTensor(torch.from_numpy(labels[None].astype(np.float32)))
    out = Invertd(keys="pred", transform=transforms, orig_keys="image", nearest_interp=True)(
        {"image": image, "pred": pred})["pred"]
    return np.asarray(out[0]).round().astype(np.uint8)


# ----------------------------------------------------------------------------- split and merge

def _box(starts: np.ndarray):
    lo = starts.min(0)
    hi = starts.max(0) + np.asarray(ROI_SIZE)
    return lo, tuple(slice(int(a), int(b)) for a, b in zip(lo, hi)), tuple(int(v) for v in hi - lo)


def make_jobs(volume: np.ndarray, orig_shape, pad_before, starts: np.ndarray, counts):
    """Give each worker a contiguous run of patches and the slab of `volume` (padded frame)
    those patches cover. Returns (jobs, owners, layout); owners counts how many different
    workers cover each voxel, and layout holds what merge() needs per job."""
    rw, rh, rd = ROI_SIZE
    owners = np.zeros(volume.shape, np.uint8)
    runs, covers = [], []
    first = 0
    for n in counts:
        run = starts[first:first + n]
        first += n
        runs.append(run)
        if n == 0:
            covers.append(None)
            continue
        lo, box, size = _box(run)
        cover = np.zeros(size, bool)
        for a, b, c in run - lo:
            cover[a:a + rw, b:b + rh, c:c + rd] = True
        owners[box] += cover
        covers.append(cover)

    jobs, layout = [], []
    for run, cover in zip(runs, covers):
        if cover is None:
            jobs.append(None)
            layout.append(None)
            continue
        lo, box, _ = _box(run)
        shared = cover & (owners[box] >= 2)
        jobs.append({"image": np.ascontiguousarray(volume[box]), "offset": lo, "starts": run,
                     "orig_shape": np.asarray(orig_shape), "pad_before": np.asarray(pad_before),
                     "shared_bits": np.packbits(shared)})
        layout.append((box, cover, shared))
    if not (owners >= 1).all():
        raise AssertionError("patch grid does not cover the scan")
    return jobs, owners, layout


def merge(results, layout, owners: np.ndarray, num_classes: int) -> np.ndarray:
    labels = np.zeros(owners.shape, np.uint8)
    shared_all = owners >= 2
    index = np.full(owners.shape, -1, np.int32)
    index[shared_all] = np.arange(int(shared_all.sum()), dtype=np.int32)
    sums = np.zeros((num_classes, int(shared_all.sum())), np.float32)
    for result, lay in zip(results, layout):
        if lay is None:
            continue
        box, cover, shared = lay
        exclusive = cover & ~shared
        view = labels[box]
        view[exclusive] = result["labels"][exclusive]
        sums[:, index[box][shared]] += result["shared_logits"]
    labels[shared_all] = sums.argmax(0)
    return labels


def dispatch(clients, profiles, jobs, counts, bytes_per_patch):
    """Run all jobs at once. A job whose worker fails is re-run on the healthy worker
    that would finish it soonest."""
    results = [None] * len(jobs)
    timings = [None] * len(jobs)

    def run(slot: int, worker: int):
        t = time.perf_counter()
        result = clients[worker].infer(jobs[slot])
        return result, {"worker": clients[worker].address, "total_s": time.perf_counter() - t,
                        "compute_s": float(result["compute_s"])}

    failed, healthy = [], set(range(len(clients)))
    with ThreadPoolExecutor(max(1, len(clients))) as pool:
        futures = {slot: pool.submit(run, slot, slot) for slot, job in enumerate(jobs) if job is not None}
        for slot, future in futures.items():
            try:
                results[slot], timings[slot] = future.result()
            except WorkerError as e:
                print(f"    ! {e}")
                failed.append(slot)
                healthy.discard(slot)
    for slot in failed:
        if not healthy:
            raise WorkerError("every worker failed")
        worker = min(healthy, key=lambda w: profiles[w].finish_time(counts[w] + counts[slot], bytes_per_patch))
        print(f"    re-running {counts[slot]} patches from {clients[slot].address} on {clients[worker].address}")
        results[slot], timings[slot] = run(slot, worker)
        counts[worker] += counts[slot]
    return results, timings


# ----------------------------------------------------------------------------- reference

def single_machine_reference(volume: np.ndarray, checkpoint: bytes, device_name: str, batch_size: int) -> np.ndarray:
    """Plain MONAI sliding_window_inference on this computer, with MONAI's own coordinate
    transform: the result distributed inference has to reproduce."""
    from monai.inferers import sliding_window_inference
    from monai.transforms import AddCoordinateChannels

    device = pick_device(device_name)
    model = build_model(load_state_dict(checkpoint), device)
    x = np.asarray(AddCoordinateChannels(spatial_dims=(0, 1, 2))(volume[None]), dtype=np.float32)
    with torch.no_grad():
        logits = sliding_window_inference(torch.from_numpy(x)[None], ROI_SIZE, batch_size, model,
                                          overlap=OVERLAP, sw_device=device, device="cpu")
    return torch.as_tensor(logits)[0].argmax(0).to(torch.uint8).numpy()


# ----------------------------------------------------------------------------- main

def main():
    parser = argparse.ArgumentParser(description="Distributed LocBAM inference on one scan.")
    parser.add_argument("--image", required=True, help="CT scan (.nii / .nii.gz)")
    parser.add_argument("--checkpoint", required=True, help="trained LocBAM weights (.pth state dict)")
    parser.add_argument("--workers", nargs="+", required=True, help="worker addresses, host:port")
    parser.add_argument("--out", required=True, help="where to save the segmentation (.nii.gz)")
    parser.add_argument("--token", default=os.environ.get("LOCBAM_TOKEN"), help="shared secret (default: $LOCBAM_TOKEN)")
    parser.add_argument("--report", help="also save the plan, specs and timings as JSON here")
    parser.add_argument("--verify", action="store_true",
                        help="also run ordinary single-machine inference here and compare")
    parser.add_argument("--verify-device", default="auto")
    parser.add_argument("--verify-batch-size", type=int, default=64)
    args = parser.parse_args()

    with open(args.checkpoint, "rb") as f:
        checkpoint = f.read()
    sha256 = hashlib.sha256(checkpoint).hexdigest()
    num_classes = int(load_state_dict(checkpoint)["final_conv.conv.weight"].shape[0])

    # 1. workers, one at a time: if two workers share a computer, benchmarking them simultaneously
    # would make each look slower than it is. (Only the first connection to a worker benchmarks.)
    print(f"1/5 Connecting to {len(args.workers)} worker(s), sending model {sha256[:12]}...")
    infos, alive = [], []
    for c in (WorkerClient(a, args.token) for a in args.workers):
        try:
            infos.append(connect(c, checkpoint, sha256))
            alive.append(c)
        except WorkerError as e:
            print(f"    ! skipping {e}")
    if not alive:
        sys.exit("No workers available.")
    clients = alive

    # 2. preprocessing
    print(f"2/5 Preprocessing {args.image}")
    t0 = time.perf_counter()
    transforms, image = preprocess(args.image)
    volume = np.asarray(image.detach().cpu().numpy()[0], dtype=np.float32)
    orig_shape = volume.shape
    padded, pad_before, starts = patch_grid(orig_shape)
    volume_padded = np.zeros(padded, np.float32)
    volume_padded[tuple(slice(p, p + s) for p, s in zip(pad_before, orig_shape))] = volume
    n_patches, n_voxels = len(starts), int(np.prod(padded))
    print(f"    {orig_shape} voxels at {SPACING} mm -> {n_patches} patches of {ROI_SIZE} "
          f"({time.perf_counter() - t0:.1f} s)")

    # 3. plan
    voxels_per_patch = n_voxels / n_patches
    bytes_per_patch = 5 * voxels_per_patch  # float32 scan out + uint8 labels back
    profiles = []
    for c, info in zip(clients, infos):
        free = info["specs"]["ram_available_gb"] * 1e9
        profiles.append(WorkerProfile(
            name=c.address, patches_per_s=info["benchmark"]["patches_per_s"], bytes_per_s=info["bytes_per_s"],
            latency_s=info["latency_s"], specs=info["specs"],
            max_patches=int(0.5 * free / (_worker_bytes_per_voxel(num_classes) * voxels_per_patch))))
    split = plan(n_patches, profiles, bytes_per_patch)
    alone = min(p.finish_time(n_patches, bytes_per_patch) for p in profiles)
    print("3/5 Plan (patches/s and link speed are measured, not guessed)")
    print(f"    {'worker':<22}{'device':<26}{'patches/s':>10}{'link MB/s':>11}{'RTT ms':>8}{'patches':>9}{'est. s':>8}")
    for p, info, n, t in zip(profiles, infos, split.counts, split.predicted_s):
        dev = f"{info['specs']['device']} ({info['specs']['device_name']})"[:25]
        print(f"    {p.name:<22}{dev:<26}{p.patches_per_s:>10.1f}{p.bytes_per_s / 1e6:>11.0f}"
              f"{p.latency_s * 1e3:>8.1f}{n:>9}{t:>8.1f}")
    print(f"    estimated finish: {split.makespan_s:.1f} s together vs {alone:.1f} s on the fastest worker alone")

    # 4. run
    print("4/5 Running")
    t0 = time.perf_counter()
    jobs, owners, layout = make_jobs(volume_padded, orig_shape, pad_before, starts, split.counts)
    results, timings = dispatch(clients, profiles, jobs, list(split.counts), bytes_per_patch)
    wall = time.perf_counter() - t0
    for slot, (t, est) in enumerate(zip(timings, split.predicted_s)):
        if t is not None:
            print(f"    {clients[slot].address:<22} ran by {t['worker']:<22} {split.counts[slot]:>5} patches: "
                  f"compute {t['compute_s']:.1f} s, total {t['total_s']:.1f} s (estimated {est:.1f} s)")
    print(f"    all workers done after {wall:.1f} s")

    # 5. merge and save
    print("5/5 Merging and saving")
    labels_padded = merge(results, layout, owners, num_classes)
    inner = tuple(slice(p, p + s) for p, s in zip(pad_before, orig_shape))
    labels = labels_padded[inner]
    print(f"    {(owners[inner] >= 2).mean():.1%} of voxels were covered by more than one worker")
    original = nib.load(args.image)
    out = to_original_space(labels, image, transforms)
    if out.shape != original.shape[:3]:
        raise AssertionError(f"restored shape {out.shape} != scan shape {original.shape}")
    nib.save(nib.Nifti1Image(out, original.affine), args.out)
    found = {int(k): int(v) for k, v in zip(*np.unique(out, return_counts=True)) if k != 0}
    print(f"    saved {args.out}  (voxels per predicted class: {found})")

    report = {"image": args.image, "checkpoint_sha256": sha256, "n_patches": n_patches,
              "preprocessed_shape": list(orig_shape), "wall_s": wall,
              "estimated_makespan_s": split.makespan_s, "estimated_fastest_alone_s": alone,
              "workers": [{"address": p.name, "specs": p.specs, "benchmark": info["benchmark"],
                           "latency_s": p.latency_s, "bytes_per_s": p.bytes_per_s, "max_patches": p.max_patches,
                           "patches": n, "estimated_s": est, "timing": t}
                          for p, info, n, est, t in zip(profiles, infos, split.counts, split.predicted_s, timings)]}

    if args.verify:
        print("    verifying against single-machine inference on this computer...")
        t0 = time.perf_counter()
        ref = single_machine_reference(volume, checkpoint, args.verify_device, args.verify_batch_size)
        ref_s = time.perf_counter() - t0
        differ = int((ref != labels).sum())
        print(f"    single machine took {ref_s:.1f} s; labels agree on {1 - differ / labels.size:.5%} "
              f"of voxels ({differ} of {labels.size} differ)")
        report["verify"] = {"single_machine_s": ref_s, "differing_voxels": differ, "voxels": int(labels.size)}

    if args.report:
        with open(args.report, "w") as f:
            json.dump(report, f, indent=2, default=float)
        print(f"    report: {args.report}")


if __name__ == "__main__":
    main()
