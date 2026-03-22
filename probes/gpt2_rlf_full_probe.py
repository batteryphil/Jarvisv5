"""
gpt2_rlf_full_probe.py — Comprehensive inference probe for GPT-2 RLF
=====================================================================
Runs on GPU (training is done). Tests:
  1. In-distribution chains (2-5 hops)
  2. OOD chains (6, 7, 8, 10, 12, 15 hops)
  3. Counterfactual / prior override
  4. KV cache verification (memory stays O(1) per loop)
  5. Lifeline ablation (gate=1.0 vs gate=0.0)
"""
import torch
import torch.nn as nn
import sys
import os

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── Imports ───────────────────────────────────────────────────────────────────
from transformers import AutoTokenizer, GPT2LMHeadModel

tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.add_special_tokens({"additional_special_tokens": ["<THINK>", "<HALT>"]})
HALT_ID = tokenizer.convert_tokens_to_ids("<HALT>")


# ── Model (same architecture as training script) ─────────────────────────────
class LoopRoPE(nn.Module):
    """1D Rotary Position Embedding for loop step encoding."""

    def __init__(self, d_model: int, base: int = 10000):
        """Init: precompute frequency bands."""
        super().__init__()
        self.d_model = d_model
        inv_freq = 1.0 / (base ** (torch.arange(0, d_model, 2).float() / d_model))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x: torch.Tensor, loop_index: int) -> torch.Tensor:
        """Apply RoPE rotation for loop_index to x."""
        n = torch.tensor(float(loop_index), device=x.device)
        freqs = n * self.inv_freq.to(device=x.device, dtype=torch.float32)
        cos_v = torch.stack([freqs.cos(), freqs.cos()], dim=-1).flatten()[:self.d_model].to(dtype=x.dtype)
        sin_v = torch.stack([freqs.sin(), freqs.sin()], dim=-1).flatten()[:self.d_model].to(dtype=x.dtype)
        x1, x2 = x[..., ::2], x[..., 1::2]
        rotated = torch.stack([-x2, x1], dim=-1).flatten(-2)
        return x * cos_v + rotated * sin_v


class RecursiveGPT2_RLF(nn.Module):
    """GPT-2 with Recursive Latent Forcing inference model."""

    MAX_LOOPS: int = 20

    def __init__(self, backbone: GPT2LMHeadModel):
        """Init: freeze GPT-2, build loop engine."""
        super().__init__()
        self.transformer = backbone.transformer
        self.lm_head = backbone.lm_head
        d = backbone.config.n_embd
        for p in self.transformer.parameters():
            p.requires_grad = False
        self.transformer.wte.weight.requires_grad = True
        self.loop_rope = LoopRoPE(d)
        self.loop_norm = nn.LayerNorm(d)
        el = nn.TransformerEncoderLayer(
            d_model=d, nhead=12, dim_feedforward=d * 4,
            dropout=0.0, activation="gelu",
            batch_first=True, norm_first=True)
        self.transformer_core = nn.TransformerEncoder(el, num_layers=2)
        self.lifeline_gate = nn.Parameter(torch.ones(d, dtype=torch.float32))
        self.d_model = d

    def forward(
        self,
        input_ids: torch.Tensor,
        gate_scale: float = 1.0,
        max_loops: int | None = None,
    ) -> tuple:
        """Forward with optional gate_scale ablation."""
        B, T = input_ids.shape
        ml = max_loops or self.MAX_LOOPS

        # Base encoding — GPT-2 single pass
        pos = torch.arange(T, device=input_ids.device).unsqueeze(0)
        x = self.transformer.wte(input_ids) + self.transformer.wpe(pos)
        x = self.transformer.drop(x)
        for block in self.transformer.h:
            out = block(x)
            x = out[0] if isinstance(out, tuple) else out
        x = self.transformer.ln_f(x)
        x_prompt = x.clone().detach()

        mask = torch.triu(
            torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)

        trace: list[tuple] = []
        last_answer = ""
        for i in range(ml):
            gate = self.lifeline_gate.to(x.dtype) * gate_scale
            x = x + gate.unsqueeze(0).unsqueeze(0) * x_prompt
            x = self.loop_rope(x, i)
            x = x + self.transformer_core(x, mask=mask)
            x = self.loop_norm(x)
            lg = self.lm_head(x)
            p = torch.softmax(lg[0, -1, :].float(), dim=-1)
            tid = p.argmax().item()
            tok = tokenizer.decode([tid]).strip()
            conf = round(p[tid].item(), 4)
            trace.append((f"L{i+1}", tok if tid != HALT_ID else "<HALT>", conf))
            if tid == HALT_ID:
                return i + 1, trace, last_answer
            last_answer = tok
        return ml, trace, last_answer


# ── Load checkpoint ───────────────────────────────────────────────────────────
print("Loading GPT-2 base...")
base = GPT2LMHeadModel.from_pretrained("gpt2")
base.resize_token_embeddings(len(tokenizer))
base.lm_head = nn.Linear(base.config.n_embd, len(tokenizer), bias=False)
base.lm_head.weight.data = base.transformer.wte.weight.data.clone()
for p in base.parameters():
    p.requires_grad = False
base.transformer.wte.weight.requires_grad = True
base.lm_head.weight.requires_grad = True

model = RecursiveGPT2_RLF(base).to(DEVICE)
ckpt = torch.load("gpt2_rlf_best.pt", map_location=DEVICE, weights_only=False)
model.load_state_dict(ckpt["model_state_dict"], strict=False)
model.eval()
step = ckpt.get("step", "?")
val_acc = ckpt.get("val_allloop_acc", 0)
print(f"Loaded checkpoint: step {step}, val {val_acc:.1f}%\n")


# ── Helper ────────────────────────────────────────────────────────────────────
def run_test(
    prompt: str,
    expected: str,
    label: str,
    gate_scale: float = 1.0,
    max_loops: int = 20,
) -> bool:
    """Run a single inference test and print results."""
    ids = tokenizer.encode(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        n, trace, answer = model(ids, gate_scale=gate_scale, max_loops=max_loops)
    hit = expected.lower() in answer.lower()
    mark = "✅" if hit else "❌"
    gs = f" [gate={gate_scale}]" if gate_scale != 1.0 else ""
    print(f"\n  {label}{gs}: {mark} (got: {answer!r}, expected: {expected!r})")
    for lbl, tok, conf in trace:
        tag = " ← ✅" if tok.lower() == expected.lower() else (
            " <HALT>" if tok == "<HALT>" else "")
        print(f"    {lbl:5s}  {tok!r:18s}  p={conf:.4f}{tag}")
    return hit


def make_chain(value: str, hops: int) -> str:
    """Generate a variable assignment chain with N hops."""
    if hops <= 26:
        vars_ = [chr(65 + i) for i in range(hops + 1)]
    else:
        vars_ = [f"X{i}" for i in range(1, hops + 2)]
    parts = [f"{vars_[0]} = {value}."]
    for i in range(1, len(vars_)):
        parts.append(f"{vars_[i]} = {vars_[i-1]}.")
    parts.append(f"What is {vars_[-1]}?")
    return " ".join(parts) + "\nAnswer:"


# ══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("  GPT-2 RLF — FULL INFERENCE PROBE")
print(f"  Checkpoint: step {step} | Val acc: {val_acc:.1f}%")
print("=" * 70)


# ── 1. KV Cache Verification ─────────────────────────────────────────────────
print("\n" + "─" * 70)
print("  1. KV CACHE VERIFICATION")
print("     Checking VRAM stays constant across increasing loop counts")
print("─" * 70)

torch.cuda.reset_peak_memory_stats()
torch.cuda.empty_cache()
base_mem = torch.cuda.memory_allocated() / 1e6

# Run a 4-hop chain and measure VRAM at each step
prompt_4 = "A = red. B = A. C = B. D = C. What is D?\nAnswer:"
ids = tokenizer.encode(prompt_4, return_tensors="pt").to(DEVICE)

# Manual forward pass to measure VRAM per loop
with torch.no_grad():
    B, T = ids.shape
    pos = torch.arange(T, device=ids.device).unsqueeze(0)
    x = model.transformer.wte(ids) + model.transformer.wpe(pos)
    x = model.transformer.drop(x)
    for block in model.transformer.h:
        out = block(x)
        x = out[0] if isinstance(out, tuple) else out
    x = model.transformer.ln_f(x)
    x_prompt = x.clone().detach()
    mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)

    after_base = torch.cuda.memory_allocated() / 1e6
    print(f"\n  After GPT-2 base pass:  {after_base:.1f} MB")

    loop_mems = []
    for i in range(10):
        gate = model.lifeline_gate.to(x.dtype)
        x = x + gate.unsqueeze(0).unsqueeze(0) * x_prompt
        x = model.loop_rope(x, i)
        x = x + model.transformer_core(x, mask=mask)
        x = model.loop_norm(x)
        mem = torch.cuda.memory_allocated() / 1e6
        loop_mems.append(mem)

    print(f"  After loop  1:          {loop_mems[0]:.1f} MB")
    print(f"  After loop  5:          {loop_mems[4]:.1f} MB")
    print(f"  After loop 10:          {loop_mems[9]:.1f} MB")
    delta = loop_mems[9] - loop_mems[0]
    print(f"  VRAM growth (L1→L10):   {delta:+.1f} MB")
    if abs(delta) < 5.0:
        print(f"  ✅ O(1) memory per loop confirmed — NO KV cache accumulation")
    else:
        print(f"  ⚠️  VRAM grew by {delta:.1f} MB — possible KV cache leak")


# ── 2. In-Distribution Tests ─────────────────────────────────────────────────
print("\n" + "─" * 70)
print("  2. IN-DISTRIBUTION CHAINS (2-5 hops)")
print("─" * 70)

in_dist = [
    ("A = red. B = A. What is B?\nAnswer:", "red", "2-hop"),
    ("A = cat. B = A. C = B. What is C?\nAnswer:", "cat", "3-hop"),
    ("A = democracy. B = A. C = B. D = C. What is D?\nAnswer:", "democracy", "4-hop"),
    ("A = quantum. B = A. C = B. D = C. E = D. What is E?\nAnswer:", "quantum", "5-hop"),
]
in_hits = sum(run_test(p, e, l) for p, e, l in in_dist)
print(f"\n  In-dist: {in_hits}/{len(in_dist)}")


# ── 3. OOD Length Generalization ──────────────────────────────────────────────
print("\n" + "─" * 70)
print("  3. OOD LENGTH GENERALIZATION (6-15 hops)")
print("     Model trained on 1-5 hops only")
print("─" * 70)

ood_tests = [
    (make_chain("python", 6),    "python",    "6-hop (OOD+1)"),
    (make_chain("neural", 7),    "neural",    "7-hop (OOD+2)"),
    (make_chain("gravity", 8),   "gravity",   "8-hop (OOD+3)"),
    (make_chain("logic", 10),    "logic",     "10-hop (2× train)"),
    (make_chain("cosmic", 12),   "cosmic",    "12-hop (2.4× train)"),
    (make_chain("stellar", 15),  "stellar",   "15-hop (3× train)"),
]
ood_hits = sum(run_test(p, e, l, max_loops=20) for p, e, l in ood_tests)
print(f"\n  OOD: {ood_hits}/{len(ood_tests)}")


# ── 4. Counterfactual / Prior Override ────────────────────────────────────────
print("\n" + "─" * 70)
print("  4. COUNTERFACTUAL (prior override)")
print("─" * 70)

counter = [
    ("Fire is icy cold. What is fire?\nAnswer:", "icy", "Prior override: fire=icy"),
    ("The sky is green. What color is the sky?\nAnswer:", "green", "Prior override: sky=green"),
    ("Water flows upward. How does water flow?\nAnswer:", "upward", "Prior override: water=upward"),
]
cf_hits = sum(run_test(p, e, l) for p, e, l in counter)
print(f"\n  Counterfactual: {cf_hits}/{len(counter)}")


# ── 5. Lifeline Ablation ─────────────────────────────────────────────────────
print("\n" + "─" * 70)
print("  5. LIFELINE ABLATION (gate=1.0 vs gate=0.0)")
print("     Tests whether the model internalized the algorithm")
print("─" * 70)

ablation_prompt = "A = democracy. B = A. C = B. D = C. What is D?\nAnswer:"
run_test(ablation_prompt, "democracy", "4-hop", gate_scale=1.0)
run_test(ablation_prompt, "democracy", "4-hop", gate_scale=0.0)

ablation_prompt2 = make_chain("logic", 6)
run_test(ablation_prompt2, "logic", "6-hop OOD", gate_scale=1.0)
run_test(ablation_prompt2, "logic", "6-hop OOD", gate_scale=0.0)


# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
total = in_hits + ood_hits + cf_hits
total_n = len(in_dist) + len(ood_tests) + len(counter)
print(f"  TOTAL: {total}/{total_n} tests passed")
print(f"  KV Cache: {'O(1) ✅' if abs(delta) < 5.0 else 'GROWING ⚠️'}")
print(f"  Gate μ={model.lifeline_gate.data.mean():.4f} σ={model.lifeline_gate.data.std():.4f}")
print("=" * 70 + "\n")
