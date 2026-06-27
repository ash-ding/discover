# Denoising (Single-Cell RNA Sequencing)

## Overview

Denoise gene expression data from single-cell RNA sequencing experiments.

This task applies TTT-Discover to the biological problem of removing technical noise from single-cell RNA-seq data while preserving true biological signal. The goal is to recover underlying gene expression patterns from noisy measurements.

## Requirements

**Special dependency - openproblems patch required**:

See `requirements/denoising/README.md` for detailed installation instructions. The task requires a patched version of the `openproblems` library with PyTorch and RAPIDS support.

```bash
# Environment
conda activate discover_denoising

# Install patched openproblems
pip install openproblems[pytorch,rapids]
```

## Configuration

- **Paper config**: `config_paper.yaml` (50 epochs, standard parameters)
- **Validation config**: `config_validate.yaml` (1 epoch, quick test)

Uses standard paper parameters (no task-specific overrides).

## Running

```bash
# Quick validation (1 epoch)
bash run.sh validate

# Full training (50 epochs)
bash run.sh full
```

## Monitoring

**Important**: Requires special post-processing beyond raw_score.

Track in WandB:
- **Metric**: `env/all/raw_score/max` (maximization task)
- **Note**: Final metric requires additional processing (see task documentation)
- **Higher is better**: Better denoising quality

The evaluation involves biological simulations and correlation metrics specific to the openproblems benchmark.

## Performance Notes

- **CPU usage**: High (biological simulations are CPU-intensive)
- **GPU memory**: TP=4 inference + 1 training GPU
- **Expected runtime**:
  - Validation (1 epoch): ~30-60 minutes
  - Full training (50 epochs): ~40-60 hours
- **Evaluation timeout**: 600 seconds (longer than most tasks due to CPU-intensive eval)

## Environment

```bash
conda activate discover_denoising
```

## Dataset

Uses standard single-cell RNA-seq datasets from the openproblems benchmark:
- Mouse retina data
- Immune cell profiling
- Other scRNA-seq datasets

Data is automatically downloaded/cached on first run.

## Algorithm Approach

The model learns to generate denoising algorithms that:
- Apply dimensionality reduction (PCA, autoencoders)
- Use k-nearest neighbors smoothing
- Implement matrix factorization
- Leverage biological priors
- Combine multiple preprocessing strategies

Best results often come from sophisticated pipelines that balance noise reduction with biological signal preservation.

## Biological Context

**Technical noise sources in scRNA-seq**:
- Low mRNA capture efficiency
- PCR amplification bias
- Dropout events (false zeros)
- Batch effects

**Goal**: Remove technical noise while preserving:
- Cell type differences
- Rare cell populations
- Developmental trajectories
- True biological zeros

## Evaluation Metrics

The openproblems framework evaluates denoising quality via:
- Correlation with bulk RNA-seq
- Preservation of cell type structure
- Recovery of known gene-gene relationships
- Biological pathway coherence

See openproblems documentation for detailed metric definitions.
