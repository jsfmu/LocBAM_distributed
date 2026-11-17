# Distributed LocBAM inference

Segment one CT scan with several computers at once. The scan is preprocessed once on a
**coordinator**. A planner decides how many 32³ patches each **worker** should run, based on that
worker's measured speed, its network link and its free memory. Each worker runs the full LocBAM
network (U-Net + segmentation head) on its share, and the coordinator merges the parts into one
segmentation.

The merged result is the same as ordinary single-machine inference (MONAI
`sliding_window_inference`, 32³ windows, 0.25 overlap). `--verify` checks this on every run.

## Why LocBAM splits cleanly

LocBAM's location context is a normalised coordinate (−0.5 … 0.5) along each axis of the
**whole** scan. A worker can rebuild those coordinates for its own slab from the scan's shape alone,
so they are never sent. Each patch still "knows" where it is in the body, even though no worker
sees the whole scan.

Sliding-window inference adds up overlapping patch outputs and then takes the argmax. Each worker
gets one compact slab, sums its own patches, and returns:
- labels (1 byte per voxel) wherever only its patches reach, and
- raw logit sums only for the thin layer where its patches overlap another worker's (about 6% of
  voxels in the test below).

The coordinator adds the overlap sums and takes the argmax there.

## How the work is split (`planner.py`)

Worker *i* needs `T_i(n) = L_i + n · (1/r_i + b/B_i)` seconds for *n* patches, where:
- `r_i` is its patches per second, measured by its own benchmark when it gets the model,
- `B_i` is the link bandwidth and `L_i` the round-trip latency, both measured by the coordinator,
- `b` is the bytes moved per patch.

Free RAM caps how many patches a worker may take. The planner picks the whole-number split that
minimises the finish time `max_i T_i(n_i)`. It hands out patches one at a time, each to the worker
that would finish earliest after taking it, which is provably optimal for this cost model. The
continuous version of the answer gives each machine work in proportion to its effective speed.

The startup benchmark is short, so after every real job each worker moves its speed estimate 70%
of the way towards what it actually achieved. Plans get more accurate from the second run on.

## Running it

Do this on **every** computer: copy this repository, then from the `LocBAM_distributed` folder run

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m Distributed.worker --token SOME-SHARED-SECRET
```

Each worker prints the address to use, for example `192.168.1.20:8765`. It picks its device
automatically: an NVIDIA GPU, then an Apple-silicon GPU, then the CPU. macOS may ask whether to allow
incoming connections; allow them. The coordinator can also run a worker itself.

Then, on the coordinator:

```bash
python -m Distributed.coordinator \
    --image scan.nii.gz --checkpoint best_metric_model.pth \
    --workers 192.168.1.20:8765 192.168.1.21:8765 192.168.1.22:8765 \
    --out prediction.nii.gz --token SOME-SHARED-SECRET --report run.json --verify
```

- The coordinator sends the model weights to each worker (once per worker process) and checks each
  worker's copy by checksum.
- The output lines up voxel-for-voxel with the input scan.
- `--report` saves every worker's specs, measured speeds, the plan and the actual timings.
- `--verify` reruns the scan the ordinary way on the coordinator and counts differing voxels.

To try it on one machine, start several workers with different `--port` values. Use `--device cpu`
and `--threads N` to imitate slower computers.

## Tests

```bash
python -m Distributed.test_distributed
```

This needs no data or checkpoint. It checks that:
- the planner matches brute force on 300 random cases,
- rebuilt coordinates equal MONAI's,
- the Apple-GPU pooling replacements match the original layers,
- split-and-merge inference equals `sliding_window_inference` with **0 differing voxels**, for
  1–3 workers, a worker with no patches, and a scan thinner than one patch.

## Measured on KiTS23 case_00000

This used the BTCV LocBAM model from `proposal_idea` (`A_btcv_locbam_fold0`). Three workers ran on
one MacBook (Apple M5 Pro): the GPU, 6 CPU threads and 2 CPU threads.

- **Scan:** 295×208×154 voxels after preprocessing, 756 patches.
- **Verify:** 0 of 9,449,440 voxels differ from single-machine inference.
- **Plan accuracy, second run:** estimated 10.2 s, actual 10.5 s, with 439 / 184 / 133 patches.
- **Not a speedup measurement.** All three workers shared one computer, so these numbers only show
  correctness and planning. Real speedups need separate machines.
- **Prediction quality:** kidneys vs KiTS expert labels, Dice 0.82. The model has no tumor class,
  since it was trained on BTCV.

## Limits

- **Trusted network only.** Workers speak plain HTTP. Use `--token` and run only on a network you
  trust. Uploads are parsed without code execution (`weights_only=True`, `allow_pickle=False`), but
  nothing is encrypted.
- **Static plan.** If a worker drops out mid-run, its slab is re-run on the healthy worker that
  would finish it soonest. Work is not rebalanced while it runs.
- **Coordinator memory.** The coordinator holds the whole preprocessed scan and the merged labels.
  Each worker holds only its slab.
