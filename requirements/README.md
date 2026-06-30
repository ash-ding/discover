# Requirements

**Server**: CUDA 12.9, 8x H100 80GB
**Python**: 3.11

## File Structure

```
requirements/
├── requirements-base.txt          # Core deps (all tasks share this)
├── requirements-math.txt          # Math tasks (no extra deps)
├── requirements-gpumode.txt       # GPU Mode tasks
├── requirements-ahc.txt           # AHC tasks
└── denoising/
    ├── requirements-denoising.txt # Denoising task
    └── openproblems_api_fix.patch # OpenProblems patch
```

## Quick Setup (single unified environment)

```bash
conda create -n verl_discover python=3.11 -y
conda activate verl_discover

# 1. Base dependencies (includes vLLM, PyTorch, Ray, etc.)
pip install -r requirements/requirements-base.txt

# 2. FlashInfer
pip install flashinfer-python -i https://flashinfer.ai/whl/cu129/torch2.11/

# 3. Flash Attention (limit jobs to avoid CPU overload)
MAX_JOBS=8 pip install flash-attn --no-build-isolation --no-cache-dir

# 4. VERL (from local fork with custom extensions)
pip install -e verl

# 5. Task-specific dependencies (install whichever you need)
pip install -r requirements/requirements-gpumode.txt    # GPU Mode
pip install -r requirements/requirements-ahc.txt        # AHC
pip install -r requirements/denoising/requirements-denoising.txt  # Denoising
# Math tasks need no extra deps
```

### Denoising extra steps

```bash
pip install git+https://github.com/czbiohub/simscity.git
pip install --no-deps git+https://github.com/czbiohub/molecular-cross-validation.git
git clone https://github.com/openproblems-bio/openproblems.git
cd openproblems && git checkout v1.0.0
git apply ../requirements/denoising/openproblems_api_fix.patch
pip install --no-deps -e ./openproblems
cd ..
```

## Running Tasks

All tasks use the unified launch script:

```bash
conda activate verl_discover
bash run_verl.sh <task>
```

Available tasks: `circle_packing`, `cp32`, `ac1`, `ac2`, `erdos`, `denoising`, `gpu_mode`, `ahc039`
