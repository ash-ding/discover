# Experiment Results Log

This file tracks results from training experiments on the TTT-Discover tasks.

---

## Circle Packing (26 circles) - 2026-06-28

**Experiment**: `circle-packing-26-10epoch`  
**Model**: Qwen3-8B  
**Config**: 10 epochs, 64×8 samples/step (512 total), LoRA rank=32  
**Hardware**: TP=4 inference (GPUs 0-3), 4-GPU parallel training (GPUs 4-7)  

### Results

| Metric | Value |
|--------|-------|
| **Best Score (ever)** | **2.634** @ step 2 |
| Final Score (max) | 2.634 |
| Final Score (min) | 0.0 |
| Total Time | 2,134s (~0.59h / 35.6 min) |
| Training Time | 870s (40.8%) |
| Sampling Time | 1,092s (51.2%) |
| KL Penalty (final) | 0.000706 |

### Notes

- **Target score** (from paper): 2.636
- **Achievement**: 99.92% of target (2.634 / 2.636)
- Best score reached at step 2 (out of 10 epochs)
- Very fast convergence - practical training could stop at epoch 2-3
- Training efficiency: 4-GPU parallel training achieved ~40% time on training, 51% on sampling
- Checkpoint: `tinker_log/local_checkpoints/circle-packing-26-10epoch/sampler_final/`

---
