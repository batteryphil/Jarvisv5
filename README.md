# Mamba-3 V25.1 — Native ACT Neural Turing Engine

A custom recursive reasoning architecture built on top of the frozen `state-spaces/mamba-130m` backbone.

## Architecture Overview

Standard LLMs allocate identical compute per token regardless of problem difficulty. This project implements **Adaptive Computation Time (ACT)**: the model loops its reasoning logic N times before generating an answer, dynamically scaling depth based on task complexity.

### The Unitary MIMO Phase Rotator

The core memory module replaces standard dense matrix multiplications (`W^N`, which cause gradient collapse) with geometric phase rotations on the complex unit circle. Because `|cos(θ)| ≤ 1.0` and `|sin(θ)| ≤ 1.0` are mathematically hard constraints, state magnitudes **cannot explode or vanish** regardless of recursion depth N. This guarantees stable BPTT gradient flow.

**Key design decision — Static Parameters:** Unlike standard Mamba-1/2 (which uses Selective, data-dependent A/B/C matrices), the Mamba-3 Reasoning Block uses static `nn.Parameter` constants. This decouples the memory geometry from the noisy input sequence, preventing semantic tokens from corrupting the phase state across loops.

### V25 JIT NVRTC Fuser

`torch.cfloat` complex types halved GPU Tensor Core throughput. V25 replaces all complex operations with equivalent **real-valued 2D rotation algebra** (the cross-terms of complex multiplication), wrapped in `@torch.jit.script`. PyTorch's nvfuser compiles all 15 tensor operations into a **single fused C++ CUDA kernel**, eliminating Python dispatch overhead.

**Active Scaling Law (N-Scale Throttle):** TPS scales linearly with `1/N`.
- N=1 → ~4,350 TPS  
- N=2 → ~2,311 TPS (current live training)
- N=3 → ~1,500 TPS (projected)

### V25.1 Trig Tax Optimization

Pre-computing `torch.cos(A_theta)` and `torch.sin(A_theta)` **once outside the BPTT recursion loop** and passing them as arguments to the JIT kernel eliminates redundant trig recalculation. This beat the expected 50% TPS degradation by +10% at N=2.

## Curriculum Training

### Padding-Masked Accuracy Gate

Previous curriculum used cross-entropy loss thresholds — the model gamed this by correctly predicting EOS padding tokens, driving loss near zero while failing on reasoning tokens.

Fix: A `valid_mask` strips EOS padding from both the accuracy denominator and the CE loss calculation. Graduation to the next loop depth (N+1) requires **85%+ discrete literal token match** on actual answer tokens across a 250-step rolling window.

### The 50% Paradox — Padding Vector Target Collision (Hot-Patched)

**The Bug:** During intermediate loop training (`step_i < n_steps - 1`), `torch.full_like()` naively overwrote ALL target positions (including EOS padding) with the `<THINK>` token ID. This created a ~30-to-1 gradient volume imbalance:
- **Loop 1:** Evaluated CE against ~80 THINK targets → Loss → 0, Acc → 100%
- **Loop 2:** Evaluated CE against ~3 actual answer tokens → Acc → 0%
- **Rolling Accuracy:** Locked at exactly **(100 + 0) / 2 = 50%**

**The Fix:**
```python
# BROKEN (V24)
tgt = torch.full_like(tgt, THINK_TOKEN_ID)

# FIXED (V25)
think_tgt = torch.full_like(tgt, THINK_TOKEN_ID)
pad_mask = (tgt == tokenizer.eos_token_id)  # Preserve padding slots
think_tgt[pad_mask] = tokenizer.eos_token_id
tgt = think_tgt
```

### NaN VRAM Memory Leak (Hot-Patched)

`torch.empty()` used for LoRA A matrix initialization pulls raw uninitialized GPU VRAM which can contain `NaN` values, instantly corrupting the BPTT graph on inference.  
Fix: Replaced with `nn.init.kaiming_uniform_()` for clean initialization.

## Key Files

| File | Description |
|---|---|
| `finetune_mamba3.py` | Main training script — V25.1 full architecture |
| `v25_benchmark.py` | Inference stress tests (Dirty Fuel, Over-Rev, Latent Probe) |
| `mamba3_v25_architecture_summary.txt` | Engineering summary of all V25.1 findings |
| `generate_logic_v3.py` | Training data generator (15k logic samples) |
| `logic_v3.json` | 15,000 transitive reasoning samples |
| `mmlu_format_v17.json` | 10,000 MMLU 4-choice format samples |

## Current Training Status

```
Step 1000 | Loss: 0.0165 | Acc: 49.4% | TPS: 2317 | VRAM: 0.88GB | MaxN: 2
```
The BPTT graph is stable at exactly 0.88GB VRAM with zero memory fragmentation. The padding target collision fix was applied at Step 1050+ and is expected to break the 49.4% accuracy ceiling.

## Model Architecture Parameters

- **Backbone:** `state-spaces/mamba-130m` (24 layers, 768 d_model) — fully frozen
- **Trainable:** LoRA rank-8 adapters on layers 6-23 + step_emb + loop_norm = **~888k params**
- **MIMO channels:** 2  
- **State dimension:** 16  
- **Max loops:** 5 (curriculum-gated)
- **Optimizer:** AdamW | Group1: step_emb@1e-2 | Group2: LoRA@3e-4
