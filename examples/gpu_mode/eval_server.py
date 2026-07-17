"""HTTP GPU kernel eval server with shared task pool.

Usage:
    python examples/gpu_mode/eval_server.py --port 8890 --num-gpus 4

Requests go into a shared queue. Per-GPU workers pull tasks when idle.
On GPU failure, tasks are re-queued for other GPUs automatically.
"""

import argparse
import concurrent.futures
import json
import queue
import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from concurrent.futures import ThreadPoolExecutor

from examples.gpu_mode.local_evaluator import LocalKernelEvaluator
from examples.gpu_mode.local_evaluator.evaluator import PooledKernelEvaluator


class SharedEvalPool:
    """Shared task pool: HTTP handlers submit work, per-GPU workers pull and evaluate.

    On infra failure (container crash, GPU error), the task is re-queued for
    another GPU — no request is permanently bound to a failing GPU.
    """

    def __init__(self, evaluators, request_timeout=3600):
        self.evaluators = evaluators
        self.request_timeout = request_timeout
        self._queue = queue.Queue()
        self._worker_busy = {ev.gpu_id: False for ev in evaluators}
        self._busy_lock = threading.Lock()
        for ev in evaluators:
            t = threading.Thread(target=self._worker, args=(ev,), daemon=True,
                                 name=f"pool-gpu-{ev.gpu_id}")
            t.start()

    def submit(self, code, task_name, gpu_type):
        future = concurrent.futures.Future()
        self._queue.put((code, task_name, gpu_type, future))
        try:
            return future.result(timeout=self.request_timeout)
        except concurrent.futures.TimeoutError:
            future.cancel()
            return {"success": False, "score_us": -1_000_000,
                    "error": f"Pool timeout ({self.request_timeout}s)"}
        except concurrent.futures.CancelledError:
            return {"success": False, "score_us": -1_000_000,
                    "error": "Request cancelled"}

    def _worker(self, evaluator):
        import sys, datetime
        gpu_id = evaluator.gpu_id
        while True:
            if not evaluator.healthy:
                evaluator._healthy_event.wait(timeout=5)
                if evaluator.state == "FAILED":
                    time.sleep(30)
                continue

            try:
                item = self._queue.get(timeout=1)
            except queue.Empty:
                continue

            code, task_name, gpu_type, future = item

            if future.cancelled() or future.done():
                continue

            evaluator.lock.acquire()
            try:
                with self._busy_lock:
                    self._worker_busy[gpu_id] = True

                if not evaluator.healthy:
                    self._queue.put(item)
                    continue

                print(f"[{datetime.datetime.now()}] pool-gpu-{gpu_id}: evaluating "
                      f"task={task_name} code_len={len(code)}", file=sys.stderr, flush=True)
                result = evaluator._evaluate_locked(code, task_name, gpu_type)
            finally:
                with self._busy_lock:
                    self._worker_busy[gpu_id] = False
                evaluator.lock.release()

            if result is None:
                print(f"[{datetime.datetime.now()}] pool-gpu-{gpu_id}: infra failure "
                      f"(kernel never evaluated), re-queuing task",
                      file=sys.stderr, flush=True)
                if not future.cancelled():
                    self._queue.put((code, task_name, gpu_type, future))
                continue

            try:
                future.set_result(result)
            except (concurrent.futures.InvalidStateError, Exception):
                continue

            if result.get("success"):
                print(f"[{datetime.datetime.now()}] pool-gpu-{gpu_id}: success, "
                      f"score_us={result.get('score_us')}", file=sys.stderr, flush=True)
            else:
                err = str(result.get("error", ""))[:200].replace("\n", " ")
                print(f"[{datetime.datetime.now()}] pool-gpu-{gpu_id}: eval error: {err}",
                          file=sys.stderr, flush=True)

    @property
    def queue_depth(self):
        return self._queue.qsize()

    def gpu_status(self):
        with self._busy_lock:
            busy = dict(self._worker_busy)
        return [{
            "gpu_id": ev.gpu_id,
            "state": ev.state,
            "busy": busy.get(ev.gpu_id, False),
        } for ev in self.evaluators]


class EvalHandler(BaseHTTPRequestHandler):
    evaluators = []
    pool = None

    def do_POST(self):
        import sys, datetime, traceback

        try:
            length = int(self.headers.get("Content-Length", 0))
            body_bytes = self.rfile.read(length)
            body = json.loads(body_bytes)

            code = body.get("code", "")
            task_name = body.get("task_name", "trimul")
            gpu_type = body.get("gpu_type", "H100")

            print(f"[{datetime.datetime.now()}] POST: task={task_name} code_len={len(code)} "
                  f"pool_depth={self.pool.queue_depth}", file=sys.stderr, flush=True)

            result = self.pool.submit(code, task_name, gpu_type)

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
        gpu_details = self.pool.gpu_status()
        healthy_count = sum(1 for g in gpu_details if g["state"] == "HEALTHY")

        response = {
            "status": "ok" if healthy_count > 0 else "degraded",
            "gpus": len(self.evaluators),
            "healthy_gpus": healthy_count,
            "pool_queue_depth": self.pool.queue_depth,
            "gpu_details": gpu_details,
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
    request_timeout = args.timeout * 6 + 300
    pool = SharedEvalPool(evaluators, request_timeout=request_timeout)
    EvalHandler.pool = pool

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
    print(f"HTTP workers: {http_workers}, Eval GPUs: {len(evaluators)}, Pool timeout: {request_timeout}s")
    server.serve_forever()


if __name__ == "__main__":
    main()
