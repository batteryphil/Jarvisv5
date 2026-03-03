
import torch

checkpoint = torch.load("dim_llm_checkpoint.pt", map_location="cpu")
for key in checkpoint.keys():
    print(f"{key}: {checkpoint[key].shape}")
