#!/usr/bin/env python3
"""Fail loudly if the installed vLLM is the wrong build for H100.

The PyPI wheel and the GitHub-release +cu129 wheel share a version number but
ship different FlashAttention-3 binaries: sm_90 vs sm_90a. Only sm_90a enables
Hopper wgmma/TMA; the plain sm_90 build is ~7x slower at decode on H100.

Run this right after `pip install -r requirements/requirements-base.txt`.
"""
import pathlib
import shutil
import subprocess
import sys

EXPECTED_ARCH = "sm_90a"
WHEEL = ("https://github.com/vllm-project/vllm/releases/download/v0.23.0/"
         "vllm-0.23.0+cu129-cp38-abi3-manylinux_2_28_x86_64.whl")


def main() -> int:
    try:
        import vllm
    except ImportError:
        print("FAIL: vllm is not installed")
        return 1

    # vllm.__version__ drops the local version tag (+cu129);
    # pip metadata keeps it, so prefer that.
    try:
        from importlib.metadata import version as _pkg_version
        ver = _pkg_version("vllm")
    except Exception:
        ver = getattr(vllm, "__version__", "?")
    so = pathlib.Path(vllm.__file__).parent / "vllm_flash_attn" / "_vllm_fa3_C.abi3.so"
    print(f"vllm version : {ver}")
    print(f"FA3 binary   : {so}")

    if not so.exists():
        print("FAIL: bundled FlashAttention-3 binary not found")
        return 1

    cuobjdump = shutil.which("cuobjdump")
    if not cuobjdump:
        for cand in ("/usr/local/cuda/bin/cuobjdump",
                     pathlib.Path.home() / "install/cuda129/bin/cuobjdump"):
            if pathlib.Path(cand).exists():
                cuobjdump = str(cand)
                break
    if not cuobjdump:
        print("SKIP: cuobjdump not found; cannot verify the compiled arch.")
        print(f"      Expected vllm version to end in '+cu129', got '{ver}'.")
        return 0 if ver.endswith("+cu129") else 1

    out = subprocess.run([cuobjdump, "--list-elf", str(so)],
                         capture_output=True, text=True).stdout
    archs = sorted({tok.split(".")[-2] for tok in out.split()
                    if ".sm_" in tok and tok.endswith(".cubin")})
    print(f"FA3 archs    : {archs or '(none found)'}")

    if EXPECTED_ARCH in archs:
        print(f"OK: FlashAttention-3 is compiled for {EXPECTED_ARCH}.")
        return 0

    print()
    print("=" * 72)
    print(f"FAIL: FlashAttention-3 is NOT compiled for {EXPECTED_ARCH}.")
    print("This is the slow PyPI wheel. H100 decode will be roughly 7x slower.")
    print()
    print("Fix:")
    print(f"  pip install --no-deps --force-reinstall \\\n    \"{WHEEL}\"")
    print("=" * 72)
    return 1


if __name__ == "__main__":
    sys.exit(main())
