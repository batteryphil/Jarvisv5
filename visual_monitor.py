import json
import time
import os
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.gridspec as gridspec
import textwrap

# --- Advanced Visual Training Monitor (DiM-LLM v3) ---
# Cyberpunk Dark Theme with enhanced Word Salad visualization

def animate(i, fig, ax_loss, ax_stats, ax_salad):
    stats_file = "training_stats.json"
    if not os.path.exists(stats_file):
        return

    try:
        with open(stats_file, "r") as f:
            stats = json.load(f)
    except (json.JSONDecodeError, PermissionError):
        return

    train_loss = stats.get("train_loss", [])
    val_loss = stats.get("val_loss", [])
    salads = stats.get("salads", [])
    tps = stats.get("tps", 0)
    current_step = stats.get("step", 0)
    
    # --- 1. Loss Panel (Left) ---
    ax_loss.clear()
    if train_loss:
        epochs = range(1, len(train_loss) + 1)
        ax_loss.plot(epochs, train_loss, label='Train Loss', color='#00FFCC', linewidth=3, marker='o', markersize=6, alpha=0.8)
        if val_loss:
            ax_loss.plot(epochs, val_loss, label='Val Loss', color='#FF007F', linewidth=3, marker='x', markersize=6, alpha=0.8)
        
        # Fill area under Train Loss
        ax_loss.fill_between(epochs, train_loss, color='#00FFCC', alpha=0.1)
        
        ax_loss.set_xlabel('Epoch', color='#888888', fontsize=10)
        ax_loss.set_ylabel('Cross-Entropy Loss', color='#888888', fontsize=10)
        ax_loss.legend(facecolor='#1A1A1A', edgecolor='#333333', labelcolor='white', fontsize=9)
    else:
        ax_loss.text(0.5, 0.5, "COLLECTING FIRST DATA POINTS...", 
                     transform=ax_loss.transAxes, ha='center', color='#00FFCC', fontsize=12, fontweight='bold')

    ax_loss.set_title('LEARNING CURVE', color='white', loc='left', pad=15, fontsize=12, fontweight='bold')
    ax_loss.grid(True, linestyle=':', alpha=0.2, color='#FFFFFF')
    ax_loss.tick_params(colors='#888888')

    # --- 2. Stats Panel (Right) ---
    ax_stats.clear()
    ax_stats.axis('off')
    
    # Calculate more info
    epoch_num = (len(train_loss)) if train_loss else 0
    
    stats_text = (
        f"SYNETHETIC INTELLIGENCE MONITOR\n"
        f"--------------------------------\n"
        f"ARCHITECTURE : DiM-LLM v3\n"
        f"CORE ENGINE  : Masked Diffusion\n"
        f"TOKENIZER    : GPT-2 BPE\n\n"
        f"CURRENT STEP : {current_step:,}\n"
        f"EPOCH        : {epoch_num}\n"
        f"THROUGHPUT   : {tps:.1f} TPS\n"
        f"VRAM STATE   : OPTIMIZED\n"
    )
    
    ax_stats.text(0.05, 0.95, stats_text, transform=ax_stats.transAxes, 
                 fontsize=11, color='#00FFCC', verticalalignment='top', family='monospace')

    # --- 3. Word Salad Panel (Bottom) ---
    ax_salad.clear()
    ax_salad.axis('off')
    
    salad_title = f"> LATEST NEURAL RECONSTRUCTION (STEP {current_step})"
    ax_salad.text(0.0, 1.05, salad_title, transform=ax_salad.transAxes, 
                  color='#FF007F', fontsize=11, fontweight='bold', verticalalignment='top')
    
    if salads:
        latest = salads[-1]
        full_text = ""
        # Show last 2-3 samples completely instead of 5 truncated ones
        for idx, s in enumerate(latest[:3]):
            prompt = s.get('prompt', f'Sample {idx}')
            response = s.get('response', '')
            
            # Remove [MASK] characters for cleaner look or highlight them?
            # User wants to see ALL. We will wrap it.
            wrapped_resp = textwrap.fill(response, width=120)
            full_text += f"\nPROMPT: {prompt}\n"
            full_text += f"RESULT: {wrapped_resp}\n"
            full_text += "-" * 120 + "\n"
        
        ax_salad.text(0.0, 0.95, full_text, transform=ax_salad.transAxes, 
                     fontsize=9, color='white', verticalalignment='top', family='monospace',
                     alpha=0.9)
    else:
        ax_salad.text(0.5, 0.5, "AWAITING EPOCH ZERO GENERATION...", 
                     transform=ax_salad.transAxes, ha='center', color='#555555', fontsize=12)

def visual_monitor():
    # Set dark background globally
    plt.rcParams['figure.facecolor'] = '#0A0A0A'
    plt.rcParams['axes.facecolor'] = '#111111'
    plt.rcParams['savefig.facecolor'] = '#0A0A0A'
    
    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1.2], width_ratios=[2, 1])
    
    ax_loss = fig.add_subplot(gs[0, 0])
    ax_stats = fig.add_subplot(gs[0, 1])
    ax_salad = fig.add_subplot(gs[1, :]) # Span across bottom
    
    plt.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.05, hspace=0.3)
    
    ani = FuncAnimation(fig, animate, fargs=(fig, ax_loss, ax_stats, ax_salad), interval=5000, cache_frame_data=False)
    
    plt.suptitle("JARVIS RECONSTRUCTION UNIT | STAGE 3", color='#FF007F', fontsize=16, fontweight='bold', y=0.97)
    
    print("Launching Redesigned Visual Monitor...")
    plt.show()

if __name__ == "__main__":
    visual_monitor()
