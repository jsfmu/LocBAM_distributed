"""How many patches each computer should run.

Cost model. Worker i needs

    T_i(n) = L_i + n * c_i      with   c_i = 1/r_i + b/B_i         (and T_i(0) = 0)

seconds for n patches, where

    r_i   compute throughput in patches/s, measured by the worker's own benchmark,
    B_i   link bandwidth between coordinator and worker in bytes/s, measured,
    b     bytes moved per patch (its share of the scan going out, labels coming back),
    L_i   fixed cost of one job (round-trip latency), measured.

Every worker runs at the same time, so the scan is done after max_i T_i(n_i).
We want the whole numbers n_i >= 0 with sum(n_i) = N, and n_i <= M_i (the most
patches worker i's free memory can hold), that make that maximum smallest.

Continuous answer: all workers that take part finish together at time T, so

    n_i = (T - L_i) / c_i       and       T = (N + sum_i L_i/c_i) / sum_i 1/c_i

i.e. each computer gets work in proportion to its effective speed 1/c_i.

Whole-number answer (what `plan` returns): hand out the N patches one at a time,
each to the worker that would finish earliest after taking it. This greedy rule is
exactly optimal for this model: while fewer than N patches are assigned, some worker
is still below its share at the optimal finish time T*, so its next patch ends by T*,
and greedy picks a patch ending no later than that. Greedy therefore never exceeds T*.
"""
from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field


@dataclass
class WorkerProfile:
    name: str
    patches_per_s: float
    bytes_per_s: float = math.inf
    latency_s: float = 0.0
    max_patches: int | None = None
    specs: dict = field(default_factory=dict)

    def seconds_per_patch(self, bytes_per_patch: float) -> float:
        return 1.0 / self.patches_per_s + bytes_per_patch / self.bytes_per_s

    def finish_time(self, n: int, bytes_per_patch: float) -> float:
        return 0.0 if n == 0 else self.latency_s + n * self.seconds_per_patch(bytes_per_patch)


@dataclass
class Plan:
    counts: list
    predicted_s: list
    makespan_s: float
    continuous_makespan_s: float


def continuous_makespan(n_patches: int, workers, bytes_per_patch: float) -> float:
    """T from the closed form above (ignores memory caps); workers too slow to help drop out."""
    if n_patches == 0:
        return 0.0
    active = sorted(workers, key=lambda w: w.latency_s)
    while active:
        c = [w.seconds_per_patch(bytes_per_patch) for w in active]
        t = (n_patches + sum(w.latency_s / ci for w, ci in zip(active, c))) / sum(1.0 / ci for ci in c)
        if all(w.latency_s < t for w in active):
            return t
        active.pop()  # the one with the largest fixed cost can't help
    raise ValueError("no workers")


def plan(n_patches: int, workers, bytes_per_patch: float) -> Plan:
    """Optimal whole-number split of n_patches across workers (see module docstring)."""
    if not workers:
        raise ValueError("no workers")
    counts = [0] * len(workers)
    heap = [(w.finish_time(1, bytes_per_patch), i) for i, w in enumerate(workers)
            if w.max_patches is None or w.max_patches > 0]
    heapq.heapify(heap)
    for _ in range(n_patches):
        if not heap:
            raise ValueError("the workers' memory can't hold all patches; add computers or free memory")
        _, i = heapq.heappop(heap)
        counts[i] += 1
        w = workers[i]
        if w.max_patches is None or counts[i] < w.max_patches:
            heapq.heappush(heap, (w.finish_time(counts[i] + 1, bytes_per_patch), i))
    predicted = [w.finish_time(n, bytes_per_patch) for w, n in zip(workers, counts)]
    return Plan(counts, predicted, max(predicted), continuous_makespan(n_patches, workers, bytes_per_patch))
