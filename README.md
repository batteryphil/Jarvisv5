# 🌀 MambaDiffusion (DiM)

A unified architecture that merges **Selective State-Space Models (SSM)** from Mamba with the **Reverse Diffusion Process**.

Usually, Diffusion models rely on Transformers (Self-Attention) which have $O(L^2)$ complexity. MambaDiffusion replaces these with Mamba blocks, achieving $O(L)$ linear scaling while maintaining the powerful generative capabilities of Diffusion.

## 🚀 Key Features

- **Unified Backbone**: The denoising model is a pure Mamba-based architecture. Instead of attending to all pixels simultaneously, it scans the image sequence forward and backward to "diffuse" information.
- **Bi-Directional Scanning**: Specifically adapted for 2D images, the SSM scans in both directions to ensure spatial consistency.
- **Universal Hardware Support**: This implementation is written in **Pure PyTorch**. Unlike standard Mamba implementations that require NVIDIA CUDA and custom kernels, this version runs natively on:
  - 💻 **CPU** (Intel/AMD/Apple Silicon)
  - 🎮 **GPU** (NVIDIA/AMD)
- **Patch-based Processing**: Treat images as sequences of patches, mapping perfectly to Mamba's sequential nature.

## 🧠 Theoretical Integration

The Reverse Diffusion step $X_{t-1} = \Phi(X_t, t)$ is modeled as a Selective State Space:
$$
h'(t) = \mathbf{A}(x_t, t) h(t) + \mathbf{B}(x_t, t) x(t) \\
y(t) = \mathbf{C}(x_t, t) h(t)
$$
The "Selective" nature of Mamba allows the model to decide which parts of the image structure to keep during the denoising process, acting as a dynamic filter for the Gaussian noise.

## 🛠️ Performance

On a standard machine, the sampling process for a 32x32 image takes approximately:

- **GPU**: ~0.08s per diffusion step.
- **CPU**: ~0.5s per diffusion step (depending on core count).

## Usage

```python
from mamba_diffusion import MambaDiffusion, DiffusionEngine

# Initialize the integrated model
model = MambaDiffusion(img_size=32, d_model=256)
engine = DiffusionEngine(model, device="cuda")

# Generate a sample
sample = engine.sample((1, 3, 32, 32))
```
