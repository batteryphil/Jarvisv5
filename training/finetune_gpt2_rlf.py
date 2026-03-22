"""
finetune_gpt2_rlf.py — Recursive Latent Forcing on GPT-2 (124M)
=================================================================
Crossover experiment: identical RLF training objective on an attention-based
Transformer instead of an SSM. Tests whether the technique is architecture-
agnostic.

Architecture mapping from Mamba2-130M v34:
  - Mamba2 layers 0-5 (frozen)  → GPT-2 layers 0-5 (frozen)
  - Mamba2 layers 6-23 (LoRA)   → GPT-2 layers 6-11 (LoRA on c_attn, c_proj)
  - mamba2_core (Mamba2 block)   → transformer_core (2-layer TransformerEncoder)
  - Prompt Lifeline (float32)    → identical
  - RoPE loop encoding           → identical
  - Per-loop RLF supervision     → identical
  - Training data (v33 chains)   → identical

Both models share d_model=768. The ONLY variable is: SSM vs attention.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from transformers import AutoTokenizer, GPT2LMHeadModel, GPT2Config
import json, random, time, os, re
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.checkpoint import checkpoint as grad_ckpt

DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_ID    = "gpt2"
BASE_SPLIT  = 6
LORA_RANK   = 8
MAX_LOOPS   = 16
SEQ_LEN     = 256
BATCH_SIZE  = 8
ACCUM       = 4
STEPS       = 100_000
LOG_EVERY   = 50
VAL_EVERY   = 500
EARLY_STOP_ACC   = 95.0
EARLY_STOP_COUNT = 3
SAVE_PATH   = "gpt2_rlf.pt"

CORE_N_LAYERS = 2
CORE_N_HEADS  = 12

print(f"\n{'='*70}", flush=True)
print(f"  GPT-2 (124M) — Recursive Latent Forcing Crossover Experiment", flush=True)
print(f"  Testing: does RLF generalize from SSM → Transformer?", flush=True)
print(f"  Device: {DEVICE} | Steps={STEPS} | MAX_LOOPS={MAX_LOOPS}", flush=True)
print(f"{'='*70}\n", flush=True)


# ── Tokenizer ─────────────────────────────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.add_special_tokens({"additional_special_tokens": ["<THINK>", "<HALT>"]})
HALT_ID = tokenizer.convert_tokens_to_ids("<HALT>")
print(f"  Vocab: {len(tokenizer):,} | <HALT>: {HALT_ID}")


# ── 1D RoPE for Loop Index ────────────────────────────────────────────────────
class LoopRoPE(nn.Module):
    """1D Rotary Position Embedding for loop step encoding."""

    def __init__(self, d_model: int, base: int = 10000):
        """Init: precompute frequency bands."""
        super().__init__()
        self.d_model = d_model
        inv_freq = 1.0 / (base ** (torch.arange(0, d_model, 2).float() / d_model))
        self.register_buffer("inv_freq", inv_freq)

    def _get_sincos(self, loop_index: int, device: torch.device, dtype: torch.dtype):
        """Compute cos/sin for a given loop index."""
        n = torch.tensor(float(loop_index), device=device)
        freqs = n * self.inv_freq.to(device=device, dtype=torch.float32)
        cos_v = torch.stack([freqs.cos(), freqs.cos()], dim=-1).flatten()[:self.d_model]
        sin_v = torch.stack([freqs.sin(), freqs.sin()], dim=-1).flatten()[:self.d_model]
        return cos_v.to(dtype=dtype), sin_v.to(dtype=dtype)

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """Rotate pairs: [x1, x2, ...] → [-x2, x1, ...]."""
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        return torch.stack([-x2, x1], dim=-1).flatten(-2)

    def forward(self, x: torch.Tensor, loop_index: int) -> torch.Tensor:
        """Apply RoPE rotation for loop_index to x."""
        cos_v, sin_v = self._get_sincos(loop_index, x.device, x.dtype)
        return x * cos_v + self._rotate_half(x) * sin_v


# ── Chain parsing (identical to v34) ──────────────────────────────────────────
def _parse_chain(text: str) -> list[str] | None:
    """Extract per-loop targets + HALT from chain text."""
    assignments = re.findall(r'([A-Za-z_]\w*)\s*=\s*(\S+?)[\.\n]', text)
    if len(assignments) < 2:
        return None
    val: dict[str, str] = {}
    for var, expr in assignments:
        val[var] = expr
    chain_vars = [assignments[0][0]]
    for var, _ in assignments[1:]:
        chain_vars.append(var)
    targets: list[str] = [chain_vars[i] for i in range(len(chain_vars) - 1)]
    final_var = chain_vars[-1]
    resolved  = val.get(final_var, final_var)
    visited: set[str] = set()
    while resolved in val and resolved not in visited:
        visited.add(resolved)
        resolved = val[resolved]
    targets.append(resolved)
    targets.append("<HALT>")
    return targets if len(targets) >= 3 else None


def _parse_override(sample: dict) -> list[str]:
    """Override: single direct answer then HALT."""
    return [sample["answer"], "<HALT>"]


def find_answer_start(ids: list[int]) -> int:
    """Find first position after 'Answer:' boundary."""
    for boundary in (
        tokenizer.encode("Answer:",  add_special_tokens=False),
        tokenizer.encode(" Answer:", add_special_tokens=False),
        tokenizer.encode("\nAnswer:", add_special_tokens=False),
    ):
        n = len(boundary)
        for i in range(len(ids) - n + 1):
            if ids[i:i + n] == boundary:
                return min(i + n, len(ids) - 1)
    return -1


# ── LoRA for GPT-2 Conv1D ─────────────────────────────────────────────────────
class LoRAConv1D(nn.Module):
    """Low-rank adapter for GPT-2's Conv1D layers.

    GPT-2 Conv1D stores weight as [in_features, out_features] and computes
    x @ weight + bias. This is transposed relative to nn.Linear.
    """

    def __init__(self, conv1d: nn.Module, rank: int = 8, alpha: float = 16.0):
        """Init from existing Conv1D, preserving shape and dtype."""
        super().__init__()
        # Conv1D weight shape: [in_features, out_features]
        d_in, d_out = conv1d.weight.shape
        dtype = conv1d.weight.dtype
        self.register_buffer("base_weight", conv1d.weight.data.clone())
        self.bias = conv1d.bias
        # LoRA: x @ (W + scale * A @ B) + bias
        # A: [d_in, rank], B: [rank, d_out]
        self.lora_A = nn.Parameter(torch.empty(d_in, rank, dtype=dtype))
        self.lora_B = nn.Parameter(torch.zeros(rank, d_out, dtype=dtype))
        self.scale  = alpha / rank
        nn.init.kaiming_uniform_(self.lora_A)

    @property
    def weight(self) -> torch.Tensor:
        """Effective weight including LoRA update."""
        return self.base_weight + self.scale * (self.lora_A @ self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward: x @ weight + bias (Conv1D convention, preserves all dims)."""
        return x @ self.weight + self.bias


# ── Recursive GPT-2 with RLF ─────────────────────────────────────────────────
class RecursiveGPT2_RLF(nn.Module):
    """GPT-2 with Recursive Latent Forcing: Prompt Lifeline + RoPE loops.

    Architecture: GPT-2 runs ALL 12 layers ONCE (frozen base encoder).
    Only the transformer_core (2-layer causal attention) loops, matching
    how mamba2_core loops in the SSM version. No LoRA needed since the
    base GPT-2 layers never re-execute.

    Loop structure:
      GPT-2 all layers (ONCE) → x_prompt snapshot → LOOP(lifeline + RoPE +
      transformer_core + loop_norm → logits)
    """

    MAX_LOOPS: int = MAX_LOOPS

    def __init__(self, backbone: GPT2LMHeadModel):
        """Init: freeze all GPT-2 layers, build loop engine."""
        super().__init__()
        self.transformer = backbone.transformer
        self.lm_head     = backbone.lm_head
        d_model          = backbone.config.n_embd

        # Freeze ALL GPT-2 layers (they run once, no need for LoRA)
        for p in self.transformer.parameters():
            p.requires_grad = False
        # Unfreeze wte so the model can learn <HALT> embedding
        self.transformer.wte.weight.requires_grad = True

        # ── Loop Engine (the ONLY thing that iterates) ─────────────────────
        self.loop_rope = LoopRoPE(d_model)
        self.loop_norm = nn.LayerNorm(d_model)

        # Transformer core: replaces mamba2_core
        # 2-layer causal self-attention block for loop-level reasoning
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=CORE_N_HEADS,
            dim_feedforward=d_model * 4,
            dropout=0.0, activation="gelu",
            batch_first=True, norm_first=True,
        )
        self.transformer_core = nn.TransformerEncoder(
            encoder_layer, num_layers=CORE_N_LAYERS)

        # Zero-init the output projections for near-identity at start
        for layer in self.transformer_core.layers:
            nn.init.zeros_(layer.linear2.weight)
            nn.init.zeros_(layer.linear2.bias)

        # Float32 vector lifeline gate
        self.lifeline_gate = nn.Parameter(
            torch.ones(d_model, dtype=torch.float32))

        self.d_model = d_model

        # Stats
        n_core = sum(p.numel() for p in self.transformer_core.parameters())
        total  = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"  GPT-2 base:      FROZEN (single pass, no LoRA)")
        print(f"  Loop engine:     {n_core:,} (transformer_core, {CORE_N_LAYERS} layers)")
        print(f"  RoPE:            0 params (analytical)")
        print(f"  Lifeline gate:   {d_model:,} floats (float32 vector)")
        print(f"  Total trainable: {total:,}")
        print(f"  Loop encoding:   RoPE (loop_i) — valid for any loop index\n")

    def _lifeline_inject(self, x: torch.Tensor, x_prompt: torch.Tensor) -> torch.Tensor:
        """Per-dimension lifeline injection."""
        gate = self.lifeline_gate.to(x.dtype)
        return x + gate.unsqueeze(0).unsqueeze(0) * x_prompt

    def _make_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Create causal attention mask for transformer_core."""
        return torch.triu(
            torch.ones(seq_len, seq_len, device=device, dtype=torch.bool),
            diagonal=1)

    def forward(
        self,
        input_ids:     torch.Tensor,
        chain_targets: list | None = None,
        ans_starts:    list | None = None,
    ) -> tuple:
        """Forward: GPT-2 single pass → lifeline loop (transformer_core only)."""
        B, T = input_ids.shape

        # ── Base encoding: GPT-2 all 12 layers, ONCE ─────────────────────
        position_ids = torch.arange(T, device=input_ids.device).unsqueeze(0)
        x = self.transformer.wte(input_ids) + self.transformer.wpe(position_ids)
        x = self.transformer.drop(x)

        for block in self.transformer.h:
            out = block(x)
            x = out[0] if isinstance(out, tuple) else out

        # Apply final layer norm from GPT-2
        x = self.transformer.ln_f(x)

        x_prompt = x.clone().detach()   # Prompt Lifeline anchor

        # Causal mask for transformer_core
        causal_mask = self._make_causal_mask(T, x.device)

        # ── Training ──────────────────────────────────────────────────────
        if self.training and chain_targets is not None:
            n_loops = max(len(t) for t in chain_targets)

            step_losses: list[torch.Tensor] = []
            step_accs:   list[torch.Tensor] = []
            halt_accs:   list[float]        = []

            for loop_i in range(n_loops):
                x = self._lifeline_inject(x, x_prompt)
                x = self.loop_rope(x, loop_i)

                # ONLY transformer_core loops
                x = x + self.transformer_core(x, mask=causal_mask)
                x = self.loop_norm(x)

                logits_step = self.lm_head(x)
                vocab_size  = logits_step.shape[-1]

                loop_loss = torch.tensor(0.0, device=x.device, requires_grad=True)
                loop_acc  = torch.tensor(0.0, device=x.device)
                valid = 0

                for b in range(B):
                    as_ = ans_starts[b]
                    if as_ < 1 or as_ >= T:
                        continue
                    btgt   = chain_targets[b]
                    tgt_id = int(btgt[min(loop_i, len(btgt) - 1)])
                    if tgt_id >= vocab_size:
                        continue
                    logits_b = logits_step[b, as_ - 1, :]
                    pred_tok = logits_b.argmax().item()
                    tgt_t    = torch.tensor(tgt_id, device=x.device)
                    loop_loss = loop_loss + F.cross_entropy(
                        logits_b.unsqueeze(0), tgt_t.unsqueeze(0))
                    loop_acc = loop_acc + float(pred_tok == tgt_id)
                    valid   += 1
                    if tgt_id == HALT_ID:
                        halt_accs.append(float(pred_tok == tgt_id))

                if valid > 0:
                    step_losses.append(loop_loss / valid)
                    step_accs.append(loop_acc   / valid)

            avg_loss   = (torch.stack(step_losses).mean()
                          if step_losses else
                          torch.tensor(0.0, device=x.device, requires_grad=True))
            avg_acc    = (torch.stack([a.clone().detach() for a in step_accs]).mean()
                          if step_accs else torch.tensor(0.0))
            ans_accs   = step_accs[:-1] if len(step_accs) > 1 else step_accs
            answer_acc = (torch.stack([a.clone().detach() for a in ans_accs]).mean()
                          if ans_accs else avg_acc)
            halt_acc   = (sum(halt_accs) / len(halt_accs)) if halt_accs else 0.0
            return avg_loss, avg_acc, answer_acc, halt_acc

        # ── Inference ─────────────────────────────────────────────────────
        else:
            trace: list[tuple] = []; last_answer = ""
            for loop_i in range(self.MAX_LOOPS):
                x = self._lifeline_inject(x, x_prompt)
                x = self.loop_rope(x, loop_i)
                x = x + self.transformer_core(x, mask=causal_mask)
                x = self.loop_norm(x)
                lg  = self.lm_head(x)
                p   = torch.softmax(lg[0, -1, :].float(), dim=-1)
                tid = p.argmax().item()
                tok = tokenizer.decode([tid]).strip()
                trace.append((f"L{loop_i+1}", tok, round(p[tid].item(), 4)))
                if tid == HALT_ID:
                    trace[-1] = (f"L{loop_i+1}", "<HALT>", round(p[tid].item(), 4))
                    return loop_i + 1, trace, last_answer
                last_answer = tok
            return self.MAX_LOOPS, trace, last_answer


# ── Data Pipeline (identical to v34) ──────────────────────────────────────────
def load_training_data() -> list[dict]:
    """Load v33 data (single-token chains + reality overrides)."""
    samples: list[dict] = []

    def _prep(text: str, tgt_strs: list[str]) -> dict | None:
        """Prepare a single sample with tokenized targets."""
        enc_ids   = tokenizer.encode(text, add_special_tokens=False)
        ans_start = find_answer_start(enc_ids)
        if ans_start < 1:
            return None
        ans_start = min(ans_start, len(enc_ids) - 1)
        tgt_ids: list[int] = []
        for ts in tgt_strs:
            if ts == "<HALT>":
                tgt_ids.append(HALT_ID); continue
            for pfx in (" ", ""):
                cands = tokenizer.encode(pfx + ts, add_special_tokens=False)
                if cands: tgt_ids.append(cands[0]); break
            else:
                tgt_ids.append(tokenizer.eos_token_id)
        return {
            "text": text, "hops": len(tgt_strs) - 1,
            "chain_targets": tgt_strs, "chain_tgt_ids": tgt_ids,
            "ans_start": ans_start,
        }

    for fname, label in [("system2_logic_v33.json", "v33"), ("system2_logic_v32.json", "v32")]:
        if not os.path.exists(fname):
            continue
        with open(fname) as f:
            data = json.load(f)
        ok = 0
        for item in data:
            ct = _parse_override(item) if item.get("type") == "override" else _parse_chain(item["text"])
            if ct and len(ct) >= 2:
                s = _prep(item["text"], ct)
                if s: samples.append(s); ok += 1
        print(f"  {label}: {ok:,} samples loaded")

    hop_dist = {}
    for s in samples: hop_dist[s["hops"]] = hop_dist.get(s["hops"], 0) + 1
    print(f"  Total: {len(samples):,} | Hop dist: {dict(sorted(hop_dist.items()))}\n")
    return samples


def make_batch(pool: list[dict], seed: int) -> tuple:
    """Sample a padded batch."""
    batch = random.Random(seed).sample(pool, min(BATCH_SIZE, len(pool)))
    enc   = tokenizer(
        [s["text"] for s in batch],
        max_length=SEQ_LEN, truncation=True, padding="max_length", return_tensors="pt"
    )
    return (
        enc["input_ids"].to(DEVICE),
        [s["chain_tgt_ids"] for s in batch],
        [s["ans_start"]     for s in batch],
    )


# ── Load Model ────────────────────────────────────────────────────────────────
print(f"  Loading {MODEL_ID}...", flush=True)
base_model = GPT2LMHeadModel.from_pretrained(MODEL_ID)

# Resize embeddings for <THINK> and <HALT>
base_model.resize_token_embeddings(len(tokenizer))

# Untie lm_head from wte (RLF needs independent head)
base_model.lm_head = nn.Linear(
    base_model.config.n_embd, len(tokenizer), bias=False)
base_model.lm_head.weight.data = base_model.transformer.wte.weight.data.clone()

for p in base_model.parameters():
    p.requires_grad = False
base_model.transformer.wte.weight.requires_grad = True
base_model.lm_head.weight.requires_grad         = True

model = RecursiveGPT2_RLF(base_model).to(DEVICE)

# ── Optimizer ─────────────────────────────────────────────────────────────────
samples = load_training_data()
if not samples:
    raise RuntimeError("No training data. Check system2_logic_v33.json exists.")

random.seed(42)
hop_buckets: dict[int, list] = {}
for s in samples:
    hop_buckets.setdefault(s["hops"], []).append(s)
train_samples: list[dict] = []; val_samples: list[dict] = []
for hop, bucket in sorted(hop_buckets.items()):
    random.shuffle(bucket)
    n_val = max(1, int(len(bucket) * 0.10))
    val_samples.extend(bucket[:n_val])
    train_samples.extend(bucket[n_val:])

print(f"  Train: {len(train_samples):,} | Val: {len(val_samples):,}")
hv = {}
for s in val_samples: hv[s["hops"]] = hv.get(s["hops"], 0) + 1
print(f"  Val hop dist: {dict(sorted(hv.items()))}\n")

# Only two groups: lifeline gate + everything else trainable
g_gate = [model.lifeline_gate]
g_rest = [p for p in model.parameters()
          if p.requires_grad and id(p) != id(model.lifeline_gate)]

optimizer = optim.AdamW([
    {"params": g_gate, "lr": 5e-4, "weight_decay": 0.0},
    {"params": g_rest, "lr": 5e-4, "weight_decay": 0.01},
])
scheduler = CosineAnnealingLR(optimizer, T_max=STEPS // ACCUM, eta_min=1e-6)

print(f"{'─'*70}")
print(f"  GPT-2 RLF | RoPE loop encoding | Vector Lifeline | {STEPS} steps")
print(f"  Same data, same loss, same hyperparameters as Mamba2 v34")
print(f"{'─'*70}\n")


def run_validation(val_pool: list, n_batches: int = 40) -> tuple:
    """Full-vocab validation."""
    vl = vaa = vfa = vha = 0.0; valid = 0
    with torch.no_grad():
        for i in range(n_batches):
            try:
                ids, ct, ans = make_batch(val_pool, seed=99999 + i)
                loss, aa, fa, ha = model(ids, chain_targets=ct, ans_starts=ans)
                if torch.isfinite(loss):
                    vl += loss.item(); vaa += aa.item()*100
                    vfa += fa.item()*100; vha += ha*100; valid += 1
            except Exception:
                pass
    if valid == 0: return float("inf"), 0.0, 0.0, 0.0
    return vl/valid, vaa/valid, vfa/valid, vha/valid


# ── Training Loop ─────────────────────────────────────────────────────────────
model.train(); optimizer.zero_grad()
t0 = time.time()
total_loss = total_aa = total_fa = total_ha = 0.0
early_stop_hits = 0; best_val_acc = 0.0; last_train_aa = 0.0

for step in range(1, STEPS + 1):
    for accum_i in range(ACCUM):
        ids, ct, ans = make_batch(train_samples, seed=step * ACCUM + accum_i)
        loss, aa, fa, ha = model(ids, chain_targets=ct, ans_starts=ans)
        (loss / ACCUM).backward()
        total_loss += loss.item(); total_aa += aa.item()*100
        total_fa   += fa.item()*100; total_ha += ha*100

    torch.nn.utils.clip_grad_norm_(
        [p for p in model.parameters() if p.requires_grad], max_norm=1.0)
    optimizer.step(); scheduler.step(); optimizer.zero_grad()

    if step % LOG_EVERY == 0:
        n = LOG_EVERY * ACCUM
        al = total_loss/n; aa = total_aa/n; fa = total_fa/n; ha = total_ha/n
        last_train_aa = aa; total_loss = total_aa = total_fa = total_ha = 0.0
        vram = torch.cuda.memory_allocated()/1e9 if DEVICE == "cuda" else 0
        tps  = (BATCH_SIZE * ACCUM * SEQ_LEN * LOG_EVERY) / (time.time() - t0); t0 = time.time()
        g    = model.lifeline_gate.data
        print(
            f"  Step {step:5d} | Loss: {al:.4f} | AllLoop: {aa:5.1f}%"
            f" | Answer: {fa:5.1f}% | Halt: {ha:5.1f}%"
            f" | Gate μ={g.mean():.3f} σ={g.std():.4f}"
            f" | LR: {optimizer.param_groups[0]['lr']:.1e}"
            f" | TPS: {int(tps)} | VRAM: {vram:.2f}GB",
            flush=True)

    if step % VAL_EVERY == 0:
        vl, vaa, vfa, vha = run_validation(val_samples)
        gap  = last_train_aa - vaa
        flag = " ⚠️ OVERFIT" if gap > 10.0 else ""
        print(f"\n  ── VAL @ step {step} {'─'*44}", flush=True)
        print(f"  Val AllLoop: {vaa:5.1f}% | Answer: {vfa:5.1f}% | Halt: {vha:5.1f}%  [GPT-2 RLF]", flush=True)
        print(f"  Train-Val gap: {gap:+.1f}pp{flag}", flush=True)

        if vaa > best_val_acc:
            best_val_acc = vaa; best_path = SAVE_PATH.replace(".pt", "_best.pt")
            torch.save({"model_state_dict": model.state_dict(),
                        "step": step, "val_allloop_acc": vaa,
                        "halt_id": HALT_ID, "backbone": "gpt2",
                        "lifeline_gate": model.lifeline_gate.data.clone(),
                        "rope": True, "d_model": model.d_model}, best_path)
            print(f"  🏆 Best val {vaa:.1f}% → {best_path}", flush=True)

        if vaa >= EARLY_STOP_ACC:
            early_stop_hits += 1
            print(f"  ⏱  Early-stop: {early_stop_hits}/{EARLY_STOP_COUNT}  (val={vaa:.1f}%)", flush=True)
            if early_stop_hits >= EARLY_STOP_COUNT:
                print(f"\n  ✅ EARLY STOP @ step {step}. Val={vaa:.1f}%\n", flush=True)
                torch.save({"model_state_dict": model.state_dict(),
                            "step": step, "halt_id": HALT_ID, "backbone": "gpt2",
                            "lifeline_gate": model.lifeline_gate.data.clone(),
                            "rope": True}, SAVE_PATH)
                break
        else:
            early_stop_hits = 0
        print(f"  {'─'*60}\n", flush=True)

    if step % 2000 == 0:
        torch.save({"model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "step": step, "halt_id": HALT_ID, "backbone": "gpt2",
                    "rope": True,
                    "lifeline_gate": model.lifeline_gate.data.clone()},
                   SAVE_PATH.replace(".pt", f"_step{step}.pt"))
        print(f"  💾 step {step} checkpoint", flush=True)
else:
    torch.save({"model_state_dict": model.state_dict(), "step": STEPS,
                "halt_id": HALT_ID, "backbone": "gpt2", "rope": True,
                "lifeline_gate": model.lifeline_gate.data.clone()}, SAVE_PATH)
    print(f"\n✅ GPT-2 RLF complete — {SAVE_PATH}")


# ── OOD Length Test ───────────────────────────────────────────────────────────
model.eval()
print("="*70)
print("  GPT-2 RLF OOD LENGTH TEST — trained 1-5 hops, testing 6/7/8/10 hops")
print("  Comparing directly against Mamba2 v34 results")
print("="*70)

ood_tests = [
    ("A = democracy. B = A. C = B. D = C. What is D?\nAnswer:",              "democracy",  "4-hop (in-dist)"),
    ("A = democracy. B = A. C = B. D = C. E = D. F = E. What is F?\nAnswer:", "democracy","6-hop (OOD+1)"),
    ("A = saxophone. B = A. C = B. D = C. E = D. F = E. G = F. What is G?\nAnswer:", "saxophone","7-hop (OOD+2)"),
    ("P = algorithm. Q = P. R = Q. S = R. T = S. U = T. V = U. W = V. What is W?\nAnswer:", "algorithm","8-hop (OOD+3)"),
    ("X1=parliament. X2=X1. X3=X2. X4=X3. X5=X4. X6=X5. X7=X6. X8=X7. X9=X8. X10=X9. What is X10?\nAnswer:", "parliament","10-hop (2× training max)"),
]

for prompt, expected, label in ood_tests:
    ids_ = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        n, trace, answer = model(ids_)
    hit = "✅ GENERALIZES" if expected.lower() in answer.lower() else f"❌ FAILS → {answer!r}"
    print(f"\n  {label}: {hit}")
    for lbl, tok, prob in trace:
        mark = " ← ✅" if tok.lower() == expected.lower() else (" <HALT>" if tok == "<HALT>" else "")
        print(f"    {lbl:5s}  {tok!r:14s}  p={prob:.4f}{mark}")

print("\n" + "="*70)
print("  If GPT-2 learns the same FSM: RLF is architecture-agnostic")
print("  If GPT-2 fails: the bottleneck is attention vs. recurrent state")
print("="*70 + "\n")
