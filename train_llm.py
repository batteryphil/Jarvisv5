import torch
import torch.optim as optim
import json, os, time, copy, random, glob
from transformers import GPT2Tokenizer
from mamba_llm_diffusion import DiM_LLM, MaskedDiffusionEngine, Config

# DiM-LLM v3.2 - Fine Tune
# FIX-1: Resume from epoch003 EMA (best checkpoint)
# FIX-2: LR 3e-5 + CosineAnnealingLR
# FIX-3: batch_size 4
# FIX-4: DSR interleaving 1:3
# FIX-5: Early stopping patience=3
# FIX-6: Auto-save best model

RESUME_CHECKPOINT = "dim_llm_ema_epoch003.pt"
DSR_FILE          = "synthetic_dsr_data.json"
DSR_RATIO         = 0.25
BATCH_SIZE        = 2   # OOM at 4 with seq_len=256 on 12GB; back to safe value
EPOCHS            = 40
LR                = 3e-5
EARLY_STOP_PAT    = 3
SEQ_LEN           = 256


def build_dsr_chunks(tokenizer, seq_len):
    if not os.path.exists(DSR_FILE):
        print("  [DSR] Not found - disabled.")
        return []
    with open(DSR_FILE, "r") as f:
        dsr_list = json.load(f)
    sep = " " + tokenizer.eos_token + " "
    raw = sep.join(dsr_list)
    ids = torch.tensor(tokenizer.encode(raw, add_special_tokens=False), dtype=torch.long)
    chunks = [ids[i:i+seq_len] for i in range(0, len(ids)-seq_len, seq_len)]
    random.shuffle(chunks)
    print(f"  [DSR] {len(chunks)} chunks from {len(dsr_list)} examples.")
    return chunks


def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"DiM-LLM v3.2 Fine-Tune on {device}")

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.add_special_tokens({"mask_token": "[MASK]"})

    with open("train_data.json", "r") as f:
        train_list = json.load(f)
    with open("val_data.json", "r") as f:
        val_list = json.load(f)

    sep = " " + tokenizer.eos_token + " "
    train_ids = torch.tensor(tokenizer.encode(sep.join(train_list), add_special_tokens=False), dtype=torch.long)
    val_ids   = torch.tensor(tokenizer.encode(sep.join(val_list),   add_special_tokens=False), dtype=torch.long)
    dsr_chunks = build_dsr_chunks(tokenizer, SEQ_LEN)

    config = Config(vocab_size=len(tokenizer), d_model=1024, n_layers=11, seq_len=SEQ_LEN)
    model     = DiM_LLM(config).to(device)
    ema_model = copy.deepcopy(model).to(device)
    for p in ema_model.parameters():
        p.requires_grad = False

    engine = MaskedDiffusionEngine(model, config, device=device, ema_decay=0.999)
    engine.ema_model = ema_model

    if os.path.exists(RESUME_CHECKPOINT):
        print(f"  -> Resuming from {RESUME_CHECKPOINT}")
        state = torch.load(RESUME_CHECKPOINT, map_location=device)
        model.load_state_dict(state)
        ema_model.load_state_dict(state)
    else:
        print("  -> No checkpoint, starting fresh.")

    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    num_chunks = (len(train_ids) - 1) // SEQ_LEN
    print(f"  -> {num_chunks} train chunks | batch={BATCH_SIZE} | lr={LR} | patience={EARLY_STOP_PAT}")

    stats = {"train_loss": [], "val_loss": [], "step": 0, "salads": [], "tps": 0}
    best_val = float("inf")
    patience_counter = 0

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0
        indices = list(range(num_chunks))
        random.shuffle(indices)
        dsr_cursor = 0

        for i in range(0, num_chunks, BATCH_SIZE):
            t0 = time.time()

            if dsr_chunks and random.random() < DSR_RATIO:
                batch_list = [dsr_chunks[(dsr_cursor + k) % len(dsr_chunks)] for k in range(BATCH_SIZE)]
                dsr_cursor += BATCH_SIZE
            else:
                batch_indices = indices[i:i+BATCH_SIZE]
                batch_list = [train_ids[idx*SEQ_LEN:(idx+1)*SEQ_LEN] for idx in batch_indices]

            if not batch_list:
                continue
            batch_tokens = torch.stack(batch_list).to(device)

            optimizer.zero_grad()
            loss = engine.forward_process(batch_tokens)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()
            engine.update_ema()

            elapsed   = max(time.time() - t0, 1e-6)
            batch_tps = (batch_tokens.shape[0] * SEQ_LEN) / elapsed
            stats["tps"] = 0.9 * stats.get("tps", batch_tps) + 0.1 * batch_tps
            epoch_loss  += loss.item()
            stats["step"] += 1

            if stats["step"] % 25 == 0:
                with open("training_stats.json", "w") as fj:
                    json.dump(stats, fj)

        scheduler.step()

        # Validation
        model.eval()
        val_accum     = 0.0
        num_val_chunks = (len(val_ids) - 1) // SEQ_LEN
        max_val        = min(num_val_chunks, 100)
        with torch.no_grad():
            for v in range(0, max_val, BATCH_SIZE):
                vb = [val_ids[j*SEQ_LEN:(j+1)*SEQ_LEN] for j in range(v, min(v+BATCH_SIZE, num_val_chunks))]
                if vb:
                    val_accum += engine.forward_process(torch.stack(vb).to(device)).item()

        avg_train = epoch_loss / max(1, num_chunks // BATCH_SIZE)
        avg_val   = val_accum  / max(1, max_val   // BATCH_SIZE)
        stats["train_loss"].append(avg_train)
        stats["val_loss"].append(avg_val)
        cur_lr = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch+1}/{EPOCHS} | Train={avg_train:.4f} Val={avg_val:.4f} LR={cur_lr:.2e} TPS={stats['tps']:.1f}")

        # Save rotating checkpoints
        ckpt     = f"dim_llm_epoch{epoch+1:03d}.pt"
        ema_ckpt = f"dim_llm_ema_epoch{epoch+1:03d}.pt"
        torch.save(model.state_dict(),     ckpt)
        torch.save(ema_model.state_dict(), ema_ckpt)
        torch.save(model.state_dict(),     "dim_llm_checkpoint.pt")
        torch.save(ema_model.state_dict(), "dim_llm_ema_checkpoint.pt")
        for pat, keep in [("dim_llm_epoch*.pt", ckpt), ("dim_llm_ema_epoch*.pt", ema_ckpt)]:
            for old in sorted(glob.glob(pat))[:-3]:
                if old != keep:
                    os.remove(old)

        # Early stopping
        if avg_val < best_val:
            best_val = avg_val
            patience_counter = 0
            torch.save(ema_model.state_dict(), "dim_llm_ema_best.pt")
            print(f"  ** Best val={best_val:.4f} saved -> dim_llm_ema_best.pt")
        else:
            patience_counter += 1
            print(f"  -- No val improvement {patience_counter}/{EARLY_STOP_PAT}")
            if patience_counter >= EARLY_STOP_PAT:
                print("Early stopping triggered. Best: dim_llm_ema_best.pt")
                break

        # Word salad
        print("--- Word Salad ---")
        samples    = engine.sample(n_samples=3, steps=32, temperature=0.3)
        salad_list = []
        for j, g in enumerate(samples):
            dec = tokenizer.decode(g.tolist(), skip_special_tokens=False)
            print(f"  [{j}] {dec[:140]}")
            salad_list.append({"prompt": f"Sample {j}", "response": dec})
        stats["salads"].append(salad_list)
        print("------------------")

        with open("training_stats.json", "w") as fj:
            json.dump(stats, fj)

    print("Training complete.")


if __name__ == "__main__":
    train()
