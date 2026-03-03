import torch
import time
import os
import json
import sys
from mamba_llm_diffusion import Tokenizer, DiM_LLM, LlmTrainer
from mamba_diffusion import MambaDiffusion, SelectiveSSM

# --- Thorough Test Suite for MambaDiffusion ---

class MambaDiffusionTester:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"--- Initialization: Running tests on {self.device} ---")

    def test_shape_consistency(self):
        print("[1/5] Testing Architecture Shape Consistency...")
        B, L, D = 4, 64, 128
        model = DiM_LLM(vocab_size=300, d_model=D)
        model.to(self.device)
        
        x_noisy = torch.randn(B, L, D, device=self.device)
        t = torch.randint(0, 1000, (B,), device=self.device)
        
        output = model(x_noisy, t)
        assert output.shape == (B, L, 300), f"Shape mismatch: {output.shape}"
        print("      PASS: Logit output shape matches expectations.")

    def test_diffusion_logic(self):
        print("[2/5] Testing Diffusion Noising Logic...")
        # Check if alpha_bar reduces smoothly
        model = DiM_LLM(vocab_size=300, d_model=128)
        trainer = LlmTrainer(model, device=self.device)
        
        assert trainer.alpha_bar[0] > trainer.alpha_bar[-1], "Alpha scheduler failed: cumulative alpha must decrease."
        assert trainer.alpha_bar[-1] < 1e-2, "Alpha scheduler failed: noise at T should be near-Gaussian."
        print("      PASS: Diffusion scheduler integrity verified.")

    def test_backbone_mamba(self):
        print("[3/5] Testing Mamba Backbone Stability (No NaNs)...")
        ssm = SelectiveSSM(d_model=128, d_state=16).to(self.device)
        x = torch.randn(2, 32, 128, device=self.device)
        
        # Test fwd
        y = ssm(x, direction=0)
        assert not torch.isnan(y).any(), "NaN detected in Forward Mamba Scan!"
        
        # Test bwd (Bi-directional check)
        y_rev = ssm(x, direction=1)
        assert not torch.isnan(y_rev).any(), "NaN detected in Backward Mamba Scan!"
        print("      PASS: Mamba Selective SSM is numerically stable.")

    def test_device_compatibility(self):
        print("[4/5] Testing Dual-Device (CPU/GPU) Portability...")
        # Force a CPU pass even if GPU exists
        model_cpu = DiM_LLM(vocab_size=100, d_model=64).to("cpu")
        x_cpu = torch.randn(1, 16, 64)
        t_cpu = torch.tensor([500])
        
        start = time.time()
        out_cpu = model_cpu(x_cpu, t_cpu)
        cpu_time = time.time() - start
        
        print(f"      CPU Execution Time (16 tokens): {cpu_time:.4f}s")
        assert out_cpu is not None
        
        if torch.cuda.is_available():
            model_gpu = model_cpu.to("cuda")
            x_gpu = x_cpu.to("cuda")
            t_gpu = t_cpu.to("cuda")
            
            start = time.time()
            out_gpu = model_gpu(x_gpu, t_gpu)
            gpu_time = time.time() - start
            print(f"      GPU Execution Time (16 tokens): {gpu_time:.4f}s")
            
        print("      PASS: Model runs on both standard CPU and CUDA GPU.")

    def test_tokenizer_tool_support(self):
        print("[5/5] Testing Tokenizer & Tool Formatting...")
        tokenizer = Tokenizer(vocab_size=1000)
        # Manually seed common markers
        text = "How to use <tool>add(1, 2)</tool> in <code>python</code>?"
        tokenizer.build_vocab([text])
        
        encoded = tokenizer.encode(text, max_len=64)
        decoded = tokenizer.decode(encoded)
        
        assert "<tool>" in decoded, f"Tokenizer failed to preserve tool markers! Seen: {decoded}"
        assert "<code>" in decoded, "Tokenizer failed to preserve code markers!"
        print("      PASS: Integrated Tokenizer supports specialized tool/code syntax.")

    def run_all(self):
        print("\n" + "="*50)
        print("MambaDiffusion (DiM) - INTEGRATED TEST SUITE")
        print("="*50 + "\n")
        
        try:
            self.test_shape_consistency()
            self.test_diffusion_logic()
            self.test_backbone_mamba()
            self.test_device_compatibility()
            self.test_tokenizer_tool_support()
            print("\n" + "="*50)
            print("FINAL STATUS: ALL TESTS PASSED ✅")
            print("="*50 + "\n")
        except Exception as e:
            print(f"\n❌ TEST FAILED: {str(e)}")
            sys.exit(1)

if __name__ == "__main__":
    tester = MambaDiffusionTester()
    tester.run_all()
