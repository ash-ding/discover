"""HTTP GPU kernel eval server. Runs independently on the eval node (Node 0).

Usage:
    python examples/gpu_mode/eval_server.py --port 8890 --num-gpus 2

The server accepts POST requests with JSON body {"code": "...", "task_name": "trimul", "gpu_type": "H100"}
and returns the evaluation result.

Health-aware routing: requests are routed only to healthy GPUs with the lowest
queue depth. Unhealthy GPUs fast-fail immediately; recovery runs in the background.
"""

import argparse
import itertools
import json
import time
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from concurrent.futures import ThreadPoolExecutor

from examples.gpu_mode.local_evaluator import LocalKernelEvaluator
from examples.gpu_mode.local_evaluator.evaluator import PooledKernelEvaluator


class EvalHandler(BaseHTTPRequestHandler):
    evaluators = []
    counter = itertools.count()
    lock = threading.Lock()
    request_timeout = 600

    def _select_evaluator(self, exclude=None):
        """Select an evaluator with the lowest queue depth.

        Prefers healthy GPUs. Falls back to recovering GPUs (the request
        will wait inside evaluate() until recovery completes). Returns None
        only when all GPUs are permanently FAILED.

        Args:
            exclude: set of gpu_ids to skip (already tried and permanently failed).
        """
        exclude = exclude or set()
        available = [ev for ev in self.evaluators
                     if ev.gpu_id not in exclude and ev.state != "FAILED"]
        if not available:
            return None

        healthy = [ev for ev in available if ev.healthy]
        candidates = healthy if healthy else available

        candidates.sort(key=lambda ev: ev.queue_depth)
        min_depth = candidates[0].queue_depth
        tied = [ev for ev in candidates if ev.queue_depth == min_depth]
        with self.lock:
            idx = next(self.counter) % len(tied)
        return tied[idx]

    def do_POST(self):
        import sys, datetime, traceback

        try:
            length = int(self.headers.get("Content-Length", 0))
            body_bytes = self.rfile.read(length)
            body = json.loads(body_bytes)

            code = body.get("code", "")
            task_name = body.get("task_name", "trimul")
            gpu_type = body.get("gpu_type", "H100")

            max_eval_retries = min(3, len(self.evaluators))
            tried_gpus = set()
            result = None

            for attempt in range(max_eval_retries):
                evaluator = self._select_evaluator(exclude=tried_gpus)
                if evaluator is None:
                    break
                tried_gpus.add(evaluator.gpu_id)

                print(f"[{datetime.datetime.now()}] POST: gpu={evaluator.gpu_id} task={task_name} "
                      f"code_len={len(code)} attempt={attempt}", file=sys.stderr, flush=True)

                result = evaluator.evaluate(code, task_name, gpu_type)

                if result.get("retriable"):
                    print(f"[{datetime.datetime.now()}] POST: GPU {evaluator.gpu_id} permanently failed, "
                          f"trying next GPU", file=sys.stderr, flush=True)
                    continue

                if result.get("success"):
                    print(f"[{datetime.datetime.now()}] POST: success, score_us={result.get('score_us')}",
                          file=sys.stderr, flush=True)
                else:
                    err_preview = str(result.get("error", ""))[:200].replace("\n", " ")
                    print(f"[{datetime.datetime.now()}] POST: eval error: {err_preview}",
                          file=sys.stderr, flush=True)
                break

            if result is None:
                gpu_states = [{"gpu_id": ev.gpu_id, "state": ev.state}
                              for ev in self.evaluators]
                result = {
                    "success": False, "score_us": -1_000_000,
                    "error": "All eval GPUs permanently failed",
                    "gpu_states": gpu_states,
                    "stdout": "", "stderr": "",
                }
                self.send_response(503)
            else:
                self.send_response(200)

            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(result).encode())

        except Exception as e:
            traceback.print_exc(file=sys.stderr)
            try:
                self.send_error(500, str(e))
            except:
                pass

    def do_GET(self):
        gpu_info = []
        healthy_count = 0
        for ev in self.evaluators:
            info = {
                "gpu_id": ev.gpu_id,
                "state": ev.state,
                "queue_depth": ev.queue_depth,
            }
            if ev.state == "RECOVERING" and hasattr(ev, "_recovery_start_time"):
                info["recovery_elapsed_s"] = round(
                    time.monotonic() - ev._recovery_start_time, 1
                )
            gpu_info.append(info)
            if ev.healthy:
                healthy_count += 1

        response = {
            "status": "ok" if healthy_count > 0 else "degraded",
            "gpus": len(self.evaluators),
            "healthy_gpus": healthy_count,
            "gpu_details": gpu_info,
        }
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(response).encode())

    def log_message(self, format, *args):
        pass


class ThreadedHTTPServer(HTTPServer):
    allow_reuse_address = True

    def __init__(self, *args, max_workers=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.executor = ThreadPoolExecutor(max_workers=max_workers or 8)

    def process_request(self, request, client_address):
        self.executor.submit(self.process_request_thread, request, client_address)

    def process_request_thread(self, request, client_address):
        try:
            self.finish_request(request, client_address)
        except Exception:
            self.handle_error(request, client_address)
        finally:
            self.shutdown_request(request)

    def shutdown(self):
        self.executor.shutdown(wait=True)
        super().shutdown()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8890)
    parser.add_argument("--num-gpus", type=int, default=2)
    parser.add_argument("--gpu-ids", type=str, default="")
    parser.add_argument("--timeout", type=int, default=530)
    parser.add_argument("--use-container", action="store_true", default=True)
    parser.add_argument("--no-container", dest="use_container", action="store_false")
    parser.add_argument("--pooled", action="store_true", default=True,
                        help="Use container pooling (persistent containers, default)")
    parser.add_argument("--no-pooled", dest="pooled", action="store_false",
                        help="Use per-eval container lifecycle (legacy)")
    args = parser.parse_args()

    gpu_ids = [int(x) for x in args.gpu_ids.split(",")] if args.gpu_ids else list(range(args.num_gpus))
    gpu_ids = gpu_ids[: args.num_gpus]

    use_pooling = args.pooled and args.use_container
    evaluators = []
    for gid in gpu_ids:
        if use_pooling:
            print(f"Initializing pooled evaluator for GPU {gid}...")
            ev = PooledKernelEvaluator(gpu_id=gid, timeout=args.timeout)
        else:
            print(f"Initializing evaluator for GPU {gid}...")
            ev = LocalKernelEvaluator(gpu_id=gid, timeout=args.timeout, use_container=args.use_container)
        evaluators.append(ev)
        print(f"  GPU {gid} ready")

    EvalHandler.evaluators = evaluators
    EvalHandler.request_timeout = args.timeout + 60

    import atexit
    def _shutdown_evaluators():
        for ev in evaluators:
            if hasattr(ev, "shutdown"):
                try:
                    ev.shutdown()
                except Exception:
                    pass
    atexit.register(_shutdown_evaluators)

    http_workers = max(len(evaluators) * 128, 512)
    server = ThreadedHTTPServer(("0.0.0.0", args.port), EvalHandler, max_workers=http_workers)
    mode = "pooled" if use_pooling else ("container" if args.use_container else "subprocess")
    print(f"Eval server listening on 0.0.0.0:{args.port} with {len(evaluators)} GPUs {gpu_ids} ({mode} mode)")
    print(f"HTTP workers: {http_workers}, Eval GPUs: {len(evaluators)}, Request timeout: {EvalHandler.request_timeout}s")
    server.serve_forever()


if __name__ == "__main__":
    main()
