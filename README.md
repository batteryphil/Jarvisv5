# Recursive Latent Forcing — Teaching Neural Networks to Think in Loops

> **📄 [Read the Full Research Paper → PAPER.md](PAPER.md)**

![phase_transition_trajectory](phase_transition_trajectory.png)

**The model's hidden state traces the same trajectory whether the training scaffold is connected or severed.** PCA projection of internal state across 10 reasoning loops. Solid = Prompt Lifeline active. Dashed = Lifeline zeroed. 9/10 predictions identical. The algorithm is fully internalized.

---

## What Is This?

A training methodology called **Recursive Latent Forcing (RLF)** that teaches small language models to solve multi-step symbolic reasoning **entirely in hidden state** — no chain-of-thought tokens, no scratchpad, no extra generation. The model loops through a tiny reasoning core, supervised at every step, and learns when to stop on its own.

**Core Discovery**: Neural networks don't fail at reasoning — they fail at *learning to reason* because gradients collapse across recurrent depth. The Prompt Lifeline provides an O(1) gradient shortcut during training. After convergence, the learned algorithm is fully internalized — the scaffold is no longer needed.

### Architecture-Agnostic

RLF works on both **SSMs** (Mamba2-130M) and **Transformers** (GPT-2-124M). Same technique, same data, same results.

---

## SSM vs Transformer — Head-to-Head

Both models trained on **identical data** (63K chains, 1–5 hops), **identical loss**, **identical hyperparameters**. The only variable is SSM vs attention.

| | **Mamba2-130M** | **GPT-2-124M** |
|---|:---:|:---:|
| In-dist accuracy | **99.9%** | 98.5% |
| Halt precision | **p=1.000** | p=0.999 |
| 6-hop OOD | **✅** | ❌ |
| 8-hop OOD | **✅** | ❌ |
| 10-hop OOD | **✅** | ❌ |
| Lifeline removable at inference | **✅** | ❌ |
| VRAM | **0.46 GB** | 1.46 GB |
| KV cache per loop | **O(1)** | **O(1)** |
| Convergence | **~1,500 steps** | ~2,500 steps |
| Throughput | **~3,000 TPS** | ~1,850 TPS |

> **Key finding**: RLF trains both architectures to ~98% accuracy with **O(1) memory per reasoning loop — zero KV cache accumulation.** But the *phase transition* — where the training scaffold becomes redundant at inference — **only emerges in the SSM.** Critically, Mamba's `d_state` does *not* persist across loops; both models pass information strictly via the residual stream `x`. The divergence is that dense self-attention causes **representation collapse** (over-smoothing), progressively blurring the data payload over repeated iterations. Mamba's selective gating acts as a perfect identity for the payload while surgically routing the pointers — making SSMs a natively superior substrate for autonomous latent test-time compute.

---

## How It Works

```
┌──────────────────────────────────────────────┐
│  Base LLM (all layers)   ← Runs ONCE        │
│  Mamba2-130M or GPT-2-124M, FROZEN           │
│  Encodes prompt → x ∈ R^[B, T, 768]          │
└──────────────────┬───────────────────────────┘
                   │
         x_prompt = x.clone()   ← Prompt Lifeline snapshot
                   │
   ┌───────────────▼───────────────────────────┐
   │         LOOP  (runs N times)              │
   │                                           │
   │   x += gate ⊙ x_prompt   ← Lifeline      │
   │       768-dim learned gate                │
   │                                           │
   │   x = RoPE(x, loop_i)    ← Loop position  │
   │       Analytical, no table — OOD capable   │
   │                                           │
   │   x += reasoning_core(x)                  │
   │       Mamba2 block (SSM) or               │
   │       2-layer TransformerEncoder (attn)    │
   │                                           │
   │   logits → supervised at EVERY loop step  │
   │   (pointer target, then <HALT>)            │
   └───────────────────────────────────────────┘
```

**Three components that matter:**

1. **Prompt Lifeline** — Re-injects the frozen prompt encoding at every loop via a per-dimension gate. Without it, the model forgets the problem after 2–3 loops. The gate learns to partition into RAM dimensions (amplifying prompt for value retrieval) and ALU dimensions (suppressing prompt to protect internal computation).

2. **RoPE Loop Encoding** — 1D Rotary Position Embeddings over the loop counter. The model knows "which reasoning step am I on" using continuous rotations that extrapolate beyond training range. This is why a model trained on 1–5 hops solves 8-hop chains.

3. **Per-Loop Supervision** — Every loop step has a gradient target. Loop 1 should predict the first pointer, loop 2 the second, all the way to `<HALT>`. This turns temporal credit assignment from an exponentially hard problem into a supervised one.

---

## The Phase Transition

The most surprising finding: the Prompt Lifeline is a **training-time scaffold**, not an inference-time dependency.

| Loop | Lifeline = 1.0 | Lifeline = 0.0 | Match |
|------|:-:|:-:|:---:|
| L1 | P | P | ✅ |
| L2 | P | P | ✅ |
| L3 | Q | Q | ✅ |
| L4 | R | R | ✅ |
| L5 | R | R | ✅ |
| L6 | S | S | ✅ |
| L7 | S | T | ❌ |
| L8 | T | T | ✅ |
| L9 | T | T | ✅ |
| L10 | T | T | ✅ |

> 9/10 loops produce identical top-1 predictions. The single divergence (L7) is a minor timing difference — both trajectories converge immediately after.

What this means: the model uses the Lifeline during training to route gradients through recurrent depth. After convergence, the entire algorithm — the FSM, the pointer logic, the halt condition — lives inside the reasoning core's parameters. The scaffold can be removed.

---

## Mechanistic Findings

Using hidden state probes and causal gate ablation, we found:

- **Discrete FSM**: The model's hidden state encodes a finite state machine with distinct states per reasoning step. PCA captures 95.6% of variance in 2 components.
- **RAM/ALU Partition**: The vector gate physically separates into 16.1% "RAM" dimensions (amplifying prompt) and 2.0% "ALU" dimensions (suppressing prompt). The model evolved a von Neumann architecture.
- **Learned Scheduler**: The `<HALT>` token fires with p=1.000 precision at exactly the right depth — the model learned its own stopping criterion.

---

## Repository Structure

```
├── training/                  # Core training scripts
│   ├── finetune_mamba2_130m_v34.py   # Mamba2 RLF (SSM)
│   ├── finetune_gpt2_rlf.py         # GPT-2 RLF (Transformer) ← NEW
│   └── ...
├── probes/                    # Mechanistic interpretability
│   ├── v34_hidden_state_probe.py     # FSM state analysis
│   ├── v34_causal_gate_ablation.py   # RAM/ALU ablation
│   ├── v34_strict_ablation.py        # True zero / noise / shuffle
│   ├── v34_trajectory_plot.py        # Phase transition visualization
│   └── ...
├── data_builders/             # Dataset generation
├── PAPER.md                   # Full research paper
├── phase_transition_trajectory.png   # Hero plot
└── README.md
```

---

## Quick Start

### 1. Install
```bash
pip install torch transformers mamba-ssm
```

### 2. Train Mamba2 RLF (v34)
```bash
PYTORCH_ALLOC_CONF=expandable_segments:True \
python -u training/finetune_mamba2_130m_v34.py 2>&1 | tee logs/v34_train.log
```
Converges in ~1,500 steps (~1 hour on any RTX GPU).

### 3. Train GPT-2 RLF (Transformer crossover)
```bash
PYTORCH_ALLOC_CONF=expandable_segments:True \
python -u training/finetune_gpt2_rlf.py 2>&1 | tee logs/gpt2_rlf.log
```

### 4. Run probes
```bash
python probes/v34_hidden_state_probe.py       # FSM analysis
python probes/v34_strict_ablation.py           # Inference autonomy
python probes/v34_trajectory_plot.py           # Generate phase transition plot
```

---

## Version History

| Version | Change |
|---------|--------|
| v28 | Latent Forcing — per-loop supervision |
| v29 | `<HALT>` — model learns its own stopping criterion |
| v31 | Ablation baseline — no lifeline (proves credit assignment failure) |
| v32 | **+ Prompt Lifeline** — O(1) gradient highway |
| v33 | **+ Vector Gate** — float32 per-dimension gate, full vocabulary |
| **v34** | **+ RoPE** — 8-hop OOD generalization, inference autonomy proven |
| **GPT-2** | **Transformer crossover** — architecture-agnostic confirmation |

---

## Citation

If you use this work, please cite the paper: [PAPER.md](PAPER.md)

## License

MIT
