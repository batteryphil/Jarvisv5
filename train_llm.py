import torch
import torch.optim as optim
import json
import os
import time
from transformers import GPT2Tokenizer
from mamba_llm_diffusion import DiM_LLM, MaskedDiffusionEngine, Config

# System Upgrade: DiM-LLM v3 with GPT-2 BPE Tokenizer

def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training DiM-LLM v3 (Masked Diffusion + GPT-2) on {device}...")
    
    # 1. Initialize GPT-2 Tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    # Add explicit mask token
    tokenizer.add_special_tokens({'mask_token': '[MASK]'})
    
    # 2. Load Dataset
    if not os.path.exists("train_data.json"):
        print("Error: train_data.json not found. Run generate_dataset.py first.")
        return

    # Load raw text and concatenate for BPE chunking
    with open("train_data.json", "r") as f:
        train_list = json.load(f)
    with open("val_data.json", "r") as f:
        val_list = json.load(f)
        
    full_train_text = " <|endoftext|> ".join(train_list)
    full_val_text = " <|endoftext|> ".join(val_list)
    
    # Encode entire stream
    train_ids = torch.tensor(tokenizer.encode(full_train_text, add_special_tokens=False), dtype=torch.long)
    val_ids = torch.tensor(tokenizer.encode(full_val_text, add_special_tokens=False), dtype=torch.long)
    
    # Model Setup
    config = Config(
        vocab_size=len(tokenizer), # 50258
        d_model=1024,
        n_layers=11,
        seq_len=256
    )

    model = DiM_LLM(config).to(device)
    
    # EMA Model Initialization (Shadow Weights)
    import copy
    ema_model = copy.deepcopy(model).to(device)
    for p in ema_model.parameters():
        p.requires_grad = False
        
    engine = MaskedDiffusionEngine(model, config, device=device, ema_decay=0.999)
    engine.ema_model = ema_model
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)


    # 3. Training Loop
    stats = {"train_loss": [], "val_loss": [], "step": 0, "salads": [], "tps": 0}
    
    batch_size = 2 # Reduced batch size for 200M model and seq_len=256 on 12GB VRAM
    epochs = 40

    
    print(f"Starting training (Vocab: {config.vocab_size}, Params: {sum(p.numel() for p in model.parameters()):,})...")

    for epoch in range(epochs):
        epoch_loss = 0
        model.train()
        
        # Chunking the stream into seq_len segments
        num_chunks = (len(train_ids) - 1) // config.seq_len
        indices = list(range(num_chunks))
        import random
        random.shuffle(indices)
        
        for i in range(0, num_chunks, batch_size):
            start_time = time.time()
            
            # Create batch
            batch_indices = indices[i:i+batch_size]
            batch_tokens = []
            for idx in batch_indices:
                chunk = train_ids[idx * config.seq_len : (idx + 1) * config.seq_len]
                batch_tokens.append(chunk)
            
            if not batch_tokens: continue
            
            batch_tokens = torch.stack(batch_tokens).to(device)
            
            # Masked Diffusion Train Step
            optimizer.zero_grad()
            loss = engine.forward_process(batch_tokens)
            loss.backward()
            optimizer.step()
            engine.update_ema()

            
            end_time = time.time()
            
            # TPS
            batch_tps = (batch_tokens.shape[0] * config.seq_len) / (end_time - start_time)
            stats["tps"] = 0.9 * stats.get("tps", batch_tps) + 0.1 * batch_tps
            
            epoch_loss += loss.item()
            stats["step"] += 1

            if stats["step"] % 25 == 0:
                with open("training_stats.json", "w") as f:
                    json.dump(stats, f)
            
        # Validation
        model.eval()
        val_loss_accum = 0
        num_val_chunks = (len(val_ids) - 1) // config.seq_len
        with torch.no_grad():
            for v_idx in range(0, min(num_val_chunks, 20), batch_size): # Limit val steps for speed
                v_batch = []
                for j in range(v_idx, min(v_idx + batch_size, num_val_chunks)):
                    v_batch.append(val_ids[j * config.seq_len : (j + 1) * config.seq_len])
                if not v_batch: continue
                v_batch = torch.stack(v_batch).to(device)
                v_loss = engine.forward_process(v_batch)
                val_loss_accum += v_loss.item()

        avg_train = epoch_loss / max(1, (num_chunks/batch_size))
        avg_val = val_loss_accum / max(1, (min(num_val_chunks, 20)/batch_size))
        stats["train_loss"].append(avg_train)
        stats["val_loss"].append(avg_val)
        
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train:.4f} - Val Loss: {avg_val:.4f} - TPS: {stats['tps']:.1f}")
        # Save both main and EMA checkpoints
        torch.save(model.state_dict(), "dim_llm_checkpoint.pt")
        torch.save(ema_model.state_dict(), "dim_llm_ema_checkpoint.pt")

        
        # Generation
        print("\n--- Epoch Word Salad ---")
        gen_tokens = engine.sample(n_samples=3, steps=32)
        epoch_salads = []
        for j in range(len(gen_tokens)):
            # Use GPT-2 decoder
            decoded = tokenizer.decode(gen_tokens[j].tolist(), skip_special_tokens=False)
            print(f"  [Sample {j}] -> {decoded}")
            epoch_salads.append({"prompt": f"Sample {j}", "response": decoded})
        
        stats["salads"].append(epoch_salads)
        print("------------------------\n")

        with open("training_stats.json", "w") as f:
            json.dump(stats, f)

    print("Training complete.")

if __name__ == "__main__":
    train()
