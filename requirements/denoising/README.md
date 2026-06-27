# OpenProblems Denoising Benchmark


## Setup

**Important**: Use Python 3.11 (not 3.12+).

**环境名**: `discover_denoising`

```bash
cd /path/to/ttt-discover

# 创建 conda 环境
conda create -n discover_denoising python=3.11 -y
conda activate discover_denoising

# 1. Install requirements
pip install -r requirements/denoising/requirements-denoising.txt

# 2. Install FlashInfer
pip install flashinfer-python -i https://flashinfer.ai/whl/cu129/torch2.11/

# 3. Git dependencies
pip install git+https://github.com/czbiohub/simscity.git
pip install --no-deps git+https://github.com/czbiohub/molecular-cross-validation.git

# 4. Clone and install openproblems (--no-deps to avoid version conflicts)
git clone https://github.com/openproblems-bio/openproblems.git
cd openproblems && git checkout v1.0.0 && cd ..

# 5. Apply patch (MUST be done before installing)
cd openproblems && git apply ../requirements/denoising/openproblems_api_fix.patch && cd ..

# 6. Install openproblems
pip install --no-deps -e ./openproblems
```

## Why --no-deps for openproblems?

`openproblems` v1.0.0 pins old dependencies (numpy 1.23.5, pandas 1.3.5, etc.) that conflict with modern packages like transformers and torch. Installing with `--no-deps` avoids these conflicts.

## What the patch fixes

The `openproblems_api_fix.patch` fixes three issues:

1. **CZI cellxgene API change** - Tabula Muris loader uses outdated API endpoints
2. **NumPy 2.x compatibility** - Replaces deprecated `np.int` with `int`
3. **Configurable cache directory** - Adds `OPENPROBLEMS_CACHE_DIR` env var support

## Caching datasets

Set `OPENPROBLEMS_CACHE_DIR` to persist downloaded datasets:

```bash
export OPENPROBLEMS_CACHE_DIR="/path/to/ttt-continuous/.openproblems_cache"
```

This avoids re-downloading data on each run and is required for distributed training where `/tmp` isn't shared across nodes.
