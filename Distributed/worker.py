"""LocBAM worker: start this on every computer that should share the work.

    python -m Distributed.worker                      (from the LocBAM_distributed folder)
    python -m Distributed.worker --port 8765 --device auto --token SECRET

The coordinator sends it the model weights once (they're checked and benchmarked
here), then one block of the preprocessed scan and the list of patches to run.
The worker runs the full LocBAM network (U-Net + segmentation head) on those
patches and sends back its part of the segmentation.

Endpoints: GET /ping, GET /info, POST /model (checkpoint bytes), POST /echo (bandwidth test),
POST /infer (npz job). If --token is set, every request must carry it in the
X-LocBAM-Token header. Checkpoints are read with weights_only=True and arrays with
allow_pickle=False, so a request can't make the worker run code.
"""
from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import socket
import sys
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from Distributed.common import (benchmark, build_model, load_state_dict, machine_specs, pack, pick_device,  # noqa: E402
                                run_job, unpack)


class WorkerState:
    def __init__(self, device: torch.device):
        self.device = device
        self.model = None
        self.model_sha256 = None
        self.num_classes = None
        self.bench = None
        self.lock = threading.Lock()

    def info(self) -> dict:
        return {"specs": machine_specs(self.device), "model_sha256": self.model_sha256,
                "num_classes": self.num_classes, "benchmark": self.bench}

    def load_model(self, checkpoint: bytes) -> None:
        sha = hashlib.sha256(checkpoint).hexdigest()
        state_dict = load_state_dict(checkpoint)
        self.model = build_model(state_dict, self.device)
        self.num_classes = int(state_dict["final_conv.conv.weight"].shape[0])
        self.model_sha256 = sha
        print(f"[worker] model {sha[:12]} loaded on {self.device}; benchmarking...", flush=True)
        self.bench = benchmark(self.model, self.device)
        print(f"[worker] {self.bench['patches_per_s']:.1f} patches/s at batch size {self.bench['batch_size']}", flush=True)

    def infer(self, job: dict) -> dict:
        if self.model is None or self.bench is None:
            raise RuntimeError("no model loaded; the coordinator sends it first")
        n = len(job["starts"])
        print(f"[worker] running {n} patches on block {tuple(job['image'].shape)}...", flush=True)
        result = run_job(self.model, self.device, self.bench["batch_size"], job)
        achieved = n / float(result["compute_s"])
        # The startup benchmark is short and synthetic; real jobs are the better evidence, so the
        # speed the coordinator plans with moves 70% of the way to what this job achieved.
        self.bench["patches_per_s"] = 0.3 * self.bench["patches_per_s"] + 0.7 * achieved
        self.bench["last_job_patches_per_s"] = achieved
        print(f"[worker] done in {float(result['compute_s']):.1f} s ({achieved:.1f} patches/s; "
              f"planning speed now {self.bench['patches_per_s']:.1f})", flush=True)
        return result


def make_handler(state: WorkerState, token: str | None):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, format, *args):  # noqa: A002  (signature set by BaseHTTPRequestHandler)
            pass

        def _send(self, code: int, body: bytes, content_type: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _json(self, code: int, obj) -> None:
            self._send(code, json.dumps(obj).encode(), "application/json")

        def _authorized(self) -> bool:
            if not token:
                return True
            if hmac.compare_digest(self.headers.get("X-LocBAM-Token", ""), token):
                return True
            self.close_connection = True  # don't read whatever an unauthorized client is uploading
            self._json(401, {"error": "wrong or missing token"})
            return False

        def do_GET(self):
            if not self._authorized():
                return
            if self.path == "/ping":
                self._json(200, {})
            elif self.path == "/info":
                self._json(200, state.info())
            else:
                self._json(404, {"error": f"unknown path {self.path}"})

        def do_POST(self):
            if not self._authorized():
                return
            body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            try:
                if self.path == "/echo":
                    self._json(200, {"received": len(body)})
                elif self.path == "/model":
                    with state.lock:
                        state.load_model(body)
                    self._json(200, state.info())
                elif self.path == "/infer":
                    with state.lock:
                        result = state.infer(unpack(body))
                    self._send(200, pack(**result), "application/octet-stream")
                else:
                    self._json(404, {"error": f"unknown path {self.path}"})
            except Exception as e:  # report any failure to the coordinator instead of dropping the connection
                traceback.print_exc()
                self._json(500, {"error": f"{type(e).__name__}: {e}"})

    return Handler


def lan_address() -> str:
    """This computer's address on the local network (no packets are sent)."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("10.255.255.255", 1))
            return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"


def main():
    parser = argparse.ArgumentParser(description="Serve LocBAM patch inference to a coordinator.")
    parser.add_argument("--host", default="0.0.0.0", help="interface to listen on (default: all)")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--device", default="auto", help="auto, cuda, mps or cpu")
    parser.add_argument("--threads", type=int, default=None, help="limit CPU threads torch uses")
    parser.add_argument("--token", default=os.environ.get("LOCBAM_TOKEN"),
                        help="shared secret the coordinator must send (default: $LOCBAM_TOKEN)")
    args = parser.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)
    state = WorkerState(pick_device(args.device))
    server = ThreadingHTTPServer((args.host, args.port), make_handler(state, args.token))
    specs = machine_specs(state.device)
    print(f"[worker] {specs['hostname']}: {state.device} ({specs['device_name']}), "
          f"{specs['cpu_cores']} cores, {specs['ram_available_gb']} of {specs['ram_total_gb']} GB RAM free", flush=True)
    print(f"[worker] listening on port {args.port}. Give the coordinator: {lan_address()}:{args.port}"
          + ("" if args.token else "  (no --token set: anyone on this network can use it)"), flush=True)
    started = time.time()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print(f"\n[worker] stopped after {time.time() - started:.0f} s")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
