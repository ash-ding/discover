"""Remote GPU kernel evaluation via Ray Actor pool.

Each eval GPU gets a dedicated Ray Actor with a LocalKernelEvaluator instance.
RemoteEvalPool distributes eval tasks across actors via round-robin, ensuring
concurrent evals never share the same physical GPU.

Usage:
    from examples.gpu_mode.remote_evaluator import RemoteEvalPool

    pool = RemoteEvalPool(
        num_gpus=2,
        gpu_ids=[0, 1],
        timeout=1200,
        max_retries=2,
        use_container=True,
    )
    result = pool.evaluate("...", "trimul", "H100")
"""

import itertools

import ray


@ray.remote(num_cpus=0, resources={"eval_gpu": 1})
class GpuEvalActor:
    """Ray Actor pinned to one eval GPU. Holds a LocalKernelEvaluator for its lifetime."""

    def __init__(self, gpu_id: int, timeout: int, max_retries: int, use_container: bool):
        import os
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)

        from examples.gpu_mode.local_evaluator import LocalKernelEvaluator

        self.gpu_id = gpu_id
        self.evaluator = LocalKernelEvaluator(
            gpu_id=gpu_id,
            timeout=timeout,
            max_retries=max_retries,
            use_container=use_container,
        )

    def evaluate(self, submission_code: str, task_name: str, gpu_type: str) -> dict:
        return self.evaluator.evaluate(
            submission_code=submission_code,
            task_name=task_name,
            gpu_type=gpu_type,
        )

    def ping(self) -> dict:
        return {"gpu_id": self.gpu_id, "status": "ok"}


def create_eval_actors(num_gpus: int, gpu_ids: list[int], timeout: int = 1200,
                       max_retries: int = 2, use_container: bool = True):
    """Create named detached GpuEvalActors. Call OUTSIDE async contexts (e.g., cluster setup)."""
    if len(gpu_ids) < num_gpus:
        gpu_ids = list(range(num_gpus))
    gpu_ids = gpu_ids[:num_gpus]
    actors = []
    for gid in gpu_ids:
        name = f"GpuEvalActor_{gid}"
        actor = GpuEvalActor.options(name=name, namespace="eval", lifetime="detached").remote(
            gid, timeout, max_retries, use_container
        )
        actors.append(actor)
    print(f"Created {len(actors)} eval actors on GPUs {gpu_ids}")
    healths = ray.get([a.ping.remote() for a in actors])
    print(f"Health check: {healths}")
    return actors


class RemoteEvalPool:
    """Pool of GpuEvalActors with round-robin dispatch.
    Looks up pre-created named actors instead of creating new ones."""

    def __init__(self, num_gpus: int, gpu_ids: list[int], **kwargs):
        if len(gpu_ids) < num_gpus:
            gpu_ids = list(range(num_gpus))
        self.gpu_ids = gpu_ids[:num_gpus]

        self.actors = []
        for gid in self.gpu_ids:
            name = f"GpuEvalActor_{gid}"
            try:
                actor = ray.get_actor(name, namespace="eval")
            except ValueError:
                actor = GpuEvalActor.options(name=name, namespace="eval", lifetime="detached").remote(
                    gid, kwargs.get("timeout", 1200), kwargs.get("max_retries", 2),
                    kwargs.get("use_container", True),
                )
            self.actors.append(actor)

        self._counter = itertools.count()
        print(f"RemoteEvalPool: {len(self.actors)} actors on GPUs {self.gpu_ids}")

    def evaluate(self, submission_code: str, task_name: str, gpu_type: str) -> dict:
        idx = next(self._counter) % len(self.actors)
        return ray.get(
            self.actors[idx].evaluate.remote(submission_code, task_name, gpu_type)
        )

    async def evaluate_async(self, submission_code: str, task_name: str, gpu_type: str) -> dict:
        """Async version — avoids blocking ray.get inside async Ray actors."""
        idx = next(self._counter) % len(self.actors)
        ref = self.actors[idx].evaluate.remote(submission_code, task_name, gpu_type)
        return await ref

    def health_check(self) -> list[dict]:
        refs = [a.ping.remote() for a in self.actors]
        return ray.get(refs)


# --- Backward compatibility ---

def _run_gpu_eval(
    submission_code: str,
    task_name: str,
    gpu_type: str,
    gpu_id: int = 0,
    timeout: int = 1200,
    max_retries: int = 2,
    use_container: bool = True,
) -> dict:
    """Legacy stateless remote eval (single GPU). Kept for backward compat."""
    import os
    os.environ.pop("CUDA_VISIBLE_DEVICES", None)

    from examples.gpu_mode.local_evaluator import LocalKernelEvaluator

    evaluator = LocalKernelEvaluator(
        gpu_id=gpu_id,
        timeout=timeout,
        max_retries=max_retries,
        use_container=use_container,
    )
    return evaluator.evaluate(
        submission_code=submission_code,
        task_name=task_name,
        gpu_type=gpu_type,
    )


remote_gpu_eval = ray.remote(
    num_cpus=0,
    resources={"eval_gpu": 1},
)(_run_gpu_eval)
