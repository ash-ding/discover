"""HTTP GPU kernel eval server. Runs independently on the eval node (Node 0).

Usage:
    python examples/gpu_mode/eval_server.py --port 8890 --num-gpus 2

The server accepts POST requests with JSON body {"code": "...", "task_name": "trimul", "gpu_type": "H100"}
and returns the evaluation result.
"""

import argparse
import itertools
import json
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError

from examples.gpu_mode.local_evaluator import LocalKernelEvaluator


class EvalHandler(BaseHTTPRequestHandler):
    evaluators = []
    counter = itertools.count()
    lock = threading.Lock()
    executor = None  # Will be set to ThreadPoolExecutor
    request_timeout = 600  # Maximum time for one request (10 minutes)

    def do_POST(self):
        import sys, datetime, traceback
        from concurrent.futures import ThreadPoolExecutor as InnerExecutor

        dt = datetime.datetime.now()
        print(f"[{dt}] POST: Entry", file=sys.stderr, flush=True)

        try:
            print(f"[{datetime.datetime.now()}] POST: Reading headers", file=sys.stderr, flush=True)
            length = int(self.headers.get("Content-Length", 0))
            print(f"[{datetime.datetime.now()}] POST: Content-Length={length}", file=sys.stderr, flush=True)

            print(f"[{datetime.datetime.now()}] POST: Reading body", file=sys.stderr, flush=True)
            body_bytes = self.rfile.read(length)
            print(f"[{datetime.datetime.now()}] POST: Body read complete, size={len(body_bytes)}", file=sys.stderr, flush=True)

            print(f"[{datetime.datetime.now()}] POST: Parsing JSON", file=sys.stderr, flush=True)
            body = json.loads(body_bytes)
            print(f"[{datetime.datetime.now()}] POST: JSON parsed", file=sys.stderr, flush=True)

            code = body.get("code", "")
            task_name = body.get("task_name", "trimul")
            gpu_type = body.get("gpu_type", "H100")

            print(f"[{datetime.datetime.now()}] POST: Params - task={task_name}, code_len={len(code)}", file=sys.stderr, flush=True)

            print(f"[{datetime.datetime.now()}] POST: Acquiring lock", file=sys.stderr, flush=True)
            with self.lock:
                idx = next(self.counter) % len(self.evaluators)
            print(f"[{datetime.datetime.now()}] POST: Selected evaluator {idx}", file=sys.stderr, flush=True)
            evaluator = self.evaluators[idx]

            # Use ThreadPoolExecutor with timeout to prevent handler from hanging
            # signal.SIGALRM doesn't work in non-main threads
            print(f"[{datetime.datetime.now()}] POST: Creating executor", file=sys.stderr, flush=True)
            with InnerExecutor(max_workers=1) as executor:
                print(f"[{datetime.datetime.now()}] POST: Submitting eval task", file=sys.stderr, flush=True)
                future = executor.submit(evaluator.evaluate, code, task_name, gpu_type)
                print(f"[{datetime.datetime.now()}] POST: Waiting for result (timeout={self.request_timeout})", file=sys.stderr, flush=True)
                try:
                    result = future.result(timeout=self.request_timeout)
                    print(f"[{datetime.datetime.now()}] POST: Result received - success={result.get('success')}, score_us={result.get('score_us')}", file=sys.stderr, flush=True)
                    if not result.get("success"):
                        err_preview = str(result.get("error", ""))[:200].replace("\n", " ")
                        print(f"[{datetime.datetime.now()}] POST: Error detail: {err_preview}", file=sys.stderr, flush=True)
                except FutureTimeoutError:
                    result = {"success": False, "score_us": -1_000_000, "error": "Request timeout", "stdout": "", "stderr": ""}
                    print(f"[{datetime.datetime.now()}] POST: Timeout after {self.request_timeout}s", file=sys.stderr, flush=True)
                    future.cancel()
                except Exception as e:
                    result = {"success": False, "score_us": -1_000_000, "error": str(e), "stdout": "", "stderr": ""}
                    print(f"[{datetime.datetime.now()}] POST: Exception - {e}", file=sys.stderr, flush=True)
                    traceback.print_exc(file=sys.stderr)

            print(f"[{datetime.datetime.now()}] POST: Sending response", file=sys.stderr, flush=True)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(result).encode())
            print(f"[{datetime.datetime.now()}] POST: Response sent complete", file=sys.stderr, flush=True)

        except Exception as e:
            print(f"[{datetime.datetime.now()}] POST: Handler error - {e}", file=sys.stderr, flush=True)
            traceback.print_exc(file=sys.stderr)
            try:
                self.send_error(500, str(e))
            except:
                print(f"[{datetime.datetime.now()}] POST: Failed to send error response", file=sys.stderr, flush=True)

    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps({"status": "ok", "gpus": len(self.evaluators)}).encode())

    def log_message(self, format, *args):
        pass


class ThreadedHTTPServer(HTTPServer):
    allow_reuse_address = True

    def __init__(self, *args, max_workers=None, **kwargs):
        super().__init__(*args, **kwargs)
        # Limit concurrent requests to number of GPUs (one request per GPU max)
        self.executor = ThreadPoolExecutor(max_workers=max_workers or 8)

    def process_request(self, request, client_address):
        # Submit to thread pool instead of creating unlimited threads
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
    args = parser.parse_args()

    gpu_ids = [int(x) for x in args.gpu_ids.split(",")] if args.gpu_ids else list(range(args.num_gpus))
    gpu_ids = gpu_ids[: args.num_gpus]

    evaluators = []
    for gid in gpu_ids:
        print(f"Initializing evaluator for GPU {gid}...")
        ev = LocalKernelEvaluator(gpu_id=gid, timeout=args.timeout, use_container=args.use_container)
        evaluators.append(ev)
        print(f"  GPU {gid} ready")

    EvalHandler.evaluators = evaluators
    EvalHandler.request_timeout = args.timeout + 60  # Give 60s buffer beyond eval timeout

    # Limit concurrent requests to number of GPUs (one request per GPU max)
    server = ThreadedHTTPServer(("0.0.0.0", args.port), EvalHandler, max_workers=len(evaluators))
    print(f"Eval server listening on 0.0.0.0:{args.port} with {len(evaluators)} GPUs {gpu_ids}")
    print(f"Max concurrent requests: {len(evaluators)}, Request timeout: {EvalHandler.request_timeout}s")
    server.serve_forever()


if __name__ == "__main__":
    main()
