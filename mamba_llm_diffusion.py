import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from tqdm import tqdm
from mamba_diffusion import MambaBlock

@dataclass
class Config:
    vocab_size: int = 50258 # GPT-2 (50257) + [MASK]
    d_model: int = 1024
    n_layers: int = 11
    seq_len: int = 256


class DiM_LLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.vocab_size = config.vocab_size
        self.d_model = config.d_model
        
        # Token Embedder
        self.token_embed = nn.Embedding(config.vocab_size, config.d_model)
        
        # Time Embedding (Continuous t_norm [0, 1])
        self.time_mlp = nn.Sequential(
            nn.Linear(1, config.d_model),
            nn.SiLU(),
            nn.Linear(config.d_model, config.d_model)
        )
        
        # Mamba Backbone
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "mamba": MambaBlock(config.d_model),
                "norm": nn.LayerNorm(config.d_model)
            })
            for _ in range(config.n_layers)
        ])
        
        self.final_norm = nn.LayerNorm(config.d_model)
        # Standard language modeling head
        self.output_proj = nn.Linear(config.d_model, config.vocab_size)

    def forward(self, input_ids, t_norm):
        """
        input_ids: (B, L)
        t_norm: (B,) - Diffusion timestep [0, 1]
        """
        x = self.token_embed(input_ids) # (B, L, D)
        t_emb = self.time_mlp(t_norm.unsqueeze(-1).float()).unsqueeze(1) # (B, 1, D)

        h = x
        for layer in self.layers:
            # Inject time embedding at every layer (DiT-style per-layer conditioning)
            h = layer["norm"](h + t_emb)
            h = h + layer["mamba"](h)
            t_emb = t_emb  # pass through (structured for future AdaLN upgrade)
            
        h = self.final_norm(h)
        return self.output_proj(h)

class MaskedDiffusionEngine:
    def __init__(self, model, config, device="cuda", ema_decay=0.999):
        self.model = model.to(device)
        self.vocab_size = config.vocab_size
        self.mask_id = config.vocab_size - 1
        self.device = device
        self.seq_len = config.seq_len
        
        # EMA Setup
        self.ema_decay = ema_decay
        self.ema_model = None # Will be set in train_llm.py if used

    def update_ema(self):
        if self.ema_model is None:
            return
        with torch.no_grad():
            for p_ema, p_model in zip(self.ema_model.parameters(), self.model.parameters()):
                p_ema.data.mul_(self.ema_decay).add_(p_model.data, alpha=1 - self.ema_decay)

    def get_mask_ratio(self, t_norm):
        return torch.cos(t_norm * math.pi / 2)

    def forward_process(self, input_ids):
        B, L = input_ids.shape
        t_norm = torch.rand(B, device=self.device)
        mask_ratio = self.get_mask_ratio(t_norm)
        
        rand_tensor = torch.rand(B, L, device=self.device)
        mask_bool = rand_tensor < mask_ratio.unsqueeze(-1)
        
        masked_inputs = input_ids.clone()
        masked_inputs[mask_bool] = self.mask_id
        
        logits = self.model(masked_inputs, t_norm)
        
        loss = F.cross_entropy(
            logits.view(-1, self.vocab_size), 
            input_ids.view(-1), 
            reduction='none'
        ).view(B, L)
        
        mask_float = mask_bool.float()
        loss = (loss * mask_float).sum() / (mask_float.sum() + 1e-8)
        
        return loss

    @torch.no_grad()
    def sample(self, n_samples=1, steps=32, prompt_ids=None, temperature=0.3):
        # temperature=0.3: Conservative sampling to prevent high-entropy repetition loops
        # Use EMA model for sampling if available
        sampling_model = self.ema_model if self.ema_model is not None else self.model
        sampling_model.eval()
        
        # Initialize with all [MASK]
        current_ids = torch.full((n_samples, self.seq_len), self.mask_id, dtype=torch.long, device=self.device)
        
        # If prompt is provided, place it at the start and fix it
        prompt_len = 0
        if prompt_ids is not None:
            prompt_len = min(prompt_ids.shape[1], self.seq_len)
            current_ids[:, :prompt_len] = prompt_ids[:, :prompt_len]
        
        mask_indices = torch.ones((n_samples, self.seq_len), dtype=torch.bool, device=self.device)
        mask_indices[:, :prompt_len] = False # Prompt tokens are not masked

        for step in tqdm(range(steps), desc="Unmasking Sequence"):
            t_scalar = 1.0 - (step / steps)
            t_norm = torch.full((n_samples,), t_scalar, device=self.device)
            
            # Predict
            logits = sampling_model(current_ids, t_norm)
            
            # Stochastic Sampling with Temperature
            probs = F.softmax(logits / temperature, dim=-1) # (B, L, V)
            B, L, V = probs.shape
            
            # Sample using Multinomial
            flat_probs = probs.reshape(-1, V)
            sampled_ids = torch.multinomial(flat_probs, 1).reshape(B, L)
            
            # Calculate confidence (use max prob from softmax for selection confidence)
            max_probs, _ = torch.max(probs, dim=-1)
            
            # Identify which tokens are currently [MASK] AND NOT part of the prompt
            is_masked = (current_ids == self.mask_id) & mask_indices
            
            ratio_to_unmask = self.get_mask_ratio(torch.tensor(t_scalar - (1.0/steps)))
            maskable_len = self.seq_len - prompt_len
            tokens_to_keep = int((1.0 - ratio_to_unmask.item()) * maskable_len)
            
            # Only consider confidence for currently masked tokens
            confidence = max_probs.masked_fill(~is_masked, -1.0) 
            _, indices_to_unmask = torch.topk(confidence, tokens_to_keep, dim=-1)
            
            # Fill sampled IDs into the selected indices
            current_ids.scatter_(1, indices_to_unmask, sampled_ids.gather(1, indices_to_unmask))

        return current_ids



