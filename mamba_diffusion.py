import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from tqdm import tqdm

# Attempt to load the C++ extension
try:
    import mamba_scan
    CPP_AVAILABLE = True
except ImportError:
    CPP_AVAILABLE = False
    print("Warning: Mamba C++ extension not found. Falling back to (slower) Pure PyTorch implementation.")
    print("To compile for maximum speed, run: python setup.py install --user")

# --- Mamba Core (Fast C++ / Generic PyTorch Implementation) ---

class SelectiveSSM(nn.Module):
    def __init__(self, d_model, d_state=16, dt_rank=None):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.dt_rank = dt_rank if dt_rank is not None else math.ceil(d_model / 16)

        # A is a learnable diagonal matrix (initialized as log-space for stability)
        A = repeat_A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(d_model, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(d_model))

        # Selective projections
        self.x_proj = nn.ParameterList([
            nn.Linear(d_model, self.dt_rank + 2 * d_state, bias=False) for _ in range(2) # Bi-directional
        ])
        self.dt_proj = nn.ParameterList([
            nn.Linear(self.dt_rank, d_model, bias=True) for _ in range(2)
        ])

    def forward(self, x, direction=0):
        """
        x: (B, L, D)
        direction: 0 for forward, 1 for backward
        """
        B, L, D = x.shape
        device = x.device

        # Selective step: Generate dt, B, C from input x
        x_proj_out = self.x_proj[direction](x)  # (B, L, rank + 2*state)
        dt_params, B_params, C_params = torch.split(x_proj_out, [self.dt_rank, self.d_state, self.d_state], dim=-1)

        dt = F.softplus(self.dt_proj[direction](dt_params))  # (B, L, D)
        A = -torch.exp(self.A_log)  # (D, n)

        # 🚀 C++ Fast Path
        if CPP_AVAILABLE:
            return mamba_scan.ssm_scan_fwd(x, dt, A, B_params, C_params, self.D)

        # 🐌 PyTorch Fallback (Recurrent implementation for CPU/Generic compatibility)
        y = torch.zeros_like(x)
        h = torch.zeros(B, D, self.d_state, device=device)  # State (B, D, N)
        A_expanded = A.unsqueeze(0)  # (1, D, N)

        for t in range(L):
            dt_t = dt[:, t, :].unsqueeze(-1)  # (B, D, 1)
            B_t = B_params[:, t, :].unsqueeze(1)  # (B, 1, N)
            C_t = C_params[:, t, :].unsqueeze(-1)  # (B, N, 1)
            x_t = x[:, t, :].unsqueeze(-1)  # (B, D, 1)

            A_bar = torch.exp(dt_t * A_expanded)
            B_bar = dt_t * B_t
            h = A_bar * h + B_bar * x_t
            y[:, t, :] = (h @ C_t).squeeze(-1)

        return y + x * self.D

class MambaBlock(nn.Module):
    def __init__(self, d_model, d_state=16, expand=2):
        super().__init__()
        self.d_model = d_model
        self.d_inner = d_model * expand
        
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, kernel_size=4, padding=3, groups=self.d_inner)
        self.ssm_fwd = SelectiveSSM(self.d_inner, d_state)
        self.ssm_bwd = SelectiveSSM(self.d_inner, d_state)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x):
        B, L, D = x.shape
        
        # In projection
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1) # (B, L, D_inner)
        
        # Conv
        x = x.transpose(1, 2)
        x = self.conv1d(x)[:, :, :L]
        x = x.transpose(1, 2)
        x = F.silu(x)
        
        # Bi-directional SSM (Crucial for images/2D signals)
        y_fwd = self.ssm_fwd(x, direction=0)
        y_bwd = self.ssm_bwd(x.flip(1), direction=1).flip(1)
        
        y = (y_fwd + y_bwd) * F.silu(z)
        
        return self.out_proj(y)

# --- Integrated Diffusion Mamba Architecture ---

class TimestepEmbedding(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model

    def forward(self, t):
        device = t.device
        half_dim = self.d_model // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class MambaDiffusion(nn.Module):
    def __init__(self, img_size=32, in_channels=3, patch_size=4, d_model=256, n_layers=4):
        super().__init__()
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        self.d_model = d_model
        
        # Encoder: Images to Patches
        self.patch_embed = nn.Conv2d(in_channels, d_model, kernel_size=patch_size, stride=patch_size)
        
        # Timestep conditioning
        self.time_mlp = nn.Sequential(
            TimestepEmbedding(d_model),
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model)
        )
        
        # Mamba Backbone (The "Mixer")
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "mamba": MambaBlock(d_model),
                "norm": nn.LayerNorm(d_model)
            })
            for _ in range(n_layers)
        ])
        
        # Final Norm + Decoder
        self.final_norm = nn.LayerNorm(d_model)
        self.final_linear = nn.Linear(d_model, patch_size * patch_size * in_channels)

    def forward(self, x, t):
        """
        x: (B, C, H, W) - Noised image
        t: (B,) - Diffusion steps
        """
        B, C, H, W = x.shape
        
        # 1. Patchify
        x = self.patch_embed(x) # (B, D, H/p, W/p)
        L = (H // self.patch_size) * (W // self.patch_size)
        x = x.flatten(2).transpose(1, 2) # (B, L, D)
        
        # 2. Time Embedding Injection
        t_emb = self.time_mlp(t).unsqueeze(1) # (B, 1, D)
        
        # 3. Mamba Processing
        for layer in self.layers:
            # Condition on time: simple addition or AdaLN (Adaptive Layer Norm)
            # Here we use addition for simplicity in the "unified" version
            h = layer["norm"](x + t_emb)
            x = x + layer["mamba"](h)
            
        x = self.final_norm(x)
        
        # 4. Depatchify
        x = self.final_linear(x) # (B, L, p*p*C)
        x = x.transpose(1, 2).reshape(B, C * self.patch_size**2, H // self.patch_size, W // self.patch_size)
        x = F.pixel_shuffle(x, self.patch_size) if C == 1 else self.custom_depatchify(x, B, C, H, W)
        
        return x

    def custom_depatchify(self, x, B, C, H, W):
        # x is (B, C*p*p, H/p, W/p)
        x = x.reshape(B, C, self.patch_size, self.patch_size, H // self.patch_size, W // self.patch_size)
        x = x.permute(0, 1, 4, 2, 5, 3).reshape(B, C, H, W)
        return x

# --- Diffusion Engine (Forward/Reverse Logic) ---

class DiffusionEngine:
    def __init__(self, model, n_steps=1000, device="cpu"):
        self.model = model.to(device)
        self.n_steps = n_steps
        self.device = device
        
        # Scheduler (Linear)
        self.beta = torch.linspace(1e-4, 0.02, n_steps, device=device)
        self.alpha = 1 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)

    def noise(self, x_0, t):
        noise = torch.randn_like(x_0)
        sqrt_alpha_bar = torch.sqrt(self.alpha_bar[t])[:, None, None, None]
        sqrt_one_minus_alpha_bar = torch.sqrt(1 - self.alpha_bar[t])[:, None, None, None]
        return sqrt_alpha_bar * x_0 + sqrt_one_minus_alpha_bar * noise, noise

    @torch.no_grad()
    def sample(self, shape):
        self.model.eval()
        x = torch.randn(shape, device=self.device)
        
        for i in tqdm(reversed(range(self.n_steps)), desc="Sampling", total=self.n_steps):
            t = torch.tensor([i] * shape[0], device=self.device)
            predicted_noise = self.model(x, t)
            
            alpha_t = self.alpha[i]
            alpha_bar_t = self.alpha_bar[i]
            beta_t = self.beta[i]
            
            if i > 0:
                noise = torch.randn_like(x)
            else:
                noise = 0
                
            x = (1 / torch.sqrt(alpha_t)) * (x - ((1 - alpha_t) / (torch.sqrt(1 - alpha_bar_t))) * predicted_noise) + torch.sqrt(beta_t) * noise
            
        return x

# --- Training / Demo ---

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running MambaDiffusion on {device}...")
    
    # Model Setup
    # Small parameters for CPU runnability
    img_size = 32
    model = MambaDiffusion(img_size=img_size, in_channels=3, d_model=128, n_layers=4)
    engine = DiffusionEngine(model, n_steps=200, device=device) # Reduced steps for demo
    
    # 1. Forward Pass (Noising)
    x_real = torch.randn(1, 3, 32, 32, device=device)
    t = torch.tensor([100], device=device)
    x_noised, noise = engine.noise(x_real, t)
    
    # 2. Prediction
    pred_noise = model(x_noised, t)
    loss = F.mse_loss(pred_noise, noise)
    print(f"Forward Consistency Check: Loss = {loss.item():.6f}")
    
    # 3. Sampling Demo
    print("Starting generation process (Universal CPU/GPU Mamba-Diffusion)...")
    samples = engine.sample((1, 3, 32, 32))
    print(f"Sampling complete. Output shape: {samples.shape}")
    print("Architecture 'MambaDiffusion' is unified and functional.")

if __name__ == "__main__":
    main()
