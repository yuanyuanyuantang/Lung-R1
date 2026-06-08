"""
Training metrics logger and real-time plotting for GRPO.

Records training metrics during GRPO training and provides
post-training visualization of reward, loss, KL, and completion curves.
"""

import os
import json
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless servers
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
from transformers import TrainerCallback


# ═══════════════════════════════════════════════════════════════
# CJK Font Setup
# ═══════════════════════════════════════════════════════════════

def _setup_font():
    """Configure matplotlib to use a CJK-capable font for Chinese text."""
    cjk_candidates = [
        "Noto Sans CJK JP", "Noto Serif CJK JP",
        "WenQuanYi Micro Hei", "WenQuanYi Zen Hei",
        "SimHei", "Microsoft YaHei", "PingFang SC",
        "Noto Sans CJK SC", "Noto Sans CJK TC",
    ]
    available = {f.name for f in fm.fontManager.ttflist}
    for font_name in cjk_candidates:
        if font_name in available:
            plt.rcParams["font.family"] = font_name
            return
    # Fallback: try to find any CJK font
    for f in fm.fontManager.ttflist:
        if any(k in f.name.lower() for k in ["cjk", "noto sans cjk", "wenquan"]):
            plt.rcParams["font.family"] = f.name
            return


# ═══════════════════════════════════════════════════════════════
# Callback: record metrics to JSONL during training
# ═══════════════════════════════════════════════════════════════

class MetricsLoggerCallback(TrainerCallback):
    """Records training metrics to a JSONL file at each logging step."""

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.metrics_path = os.path.join(output_dir, "training_metrics.jsonl")
        self.records = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        record = {
            "step": state.global_step,
            "epoch": round(state.epoch, 4) if state.epoch else 0,
        }
        # Extract relevant metrics
        for key in [
            "loss", "grad_norm", "learning_rate",
            "reward", "reward_std",
            "kl",
            "completions/mean_length",
            "completions/clipped_ratio",
            "frac_reward_zero_std",
            "clip_ratio/low_mean",
            "clip_ratio/high_mean",
        ]:
            if key in logs:
                val = logs[key]
                if hasattr(val, "item"):
                    val = val.item()
                record[key] = round(val, 6) if isinstance(val, float) else val

        # Also try to capture reward function component values
        for key in logs:
            if "rewards/reward_func" in key:
                val = logs[key]
                if hasattr(val, "item"):
                    val = val.item()
                record[key] = round(val, 6) if isinstance(val, float) else val

        self.records.append(record)
        # Append to file
        with open(self.metrics_path, "a") as f:
            f.write(json.dumps(record) + "\n")

    def get_records(self):
        return self.records


# ═══════════════════════════════════════════════════════════════
# Plotting functions
# ═══════════════════════════════════════════════════════════════

def load_metrics(metrics_path: str):
    """Load metrics from JSONL file."""
    records = []
    if not os.path.exists(metrics_path):
        print(f"[Plot] No metrics file found at {metrics_path}")
        return records
    with open(metrics_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def plot_training_curves(metrics_path: str, output_dir: str = None):
    """Generate publication-quality training curve plots.

    Creates a single figure with 6 subplots:
      1. Total Reward
      2. Loss
      3. KL Divergence
      4. Completion Length
      5. Reward Std
      6. Clipped Ratio / Zero-Std Fraction
    """
    records = load_metrics(metrics_path)
    if not records:
        print("[Plot] No metrics to plot")
        return

    if output_dir is None:
        output_dir = os.path.dirname(metrics_path)

    steps = [r["step"] for r in records]
    epochs = [r["epoch"] for r in records]

    # ── Extract metrics ──
    def safe_get(records, key, default=0.0):
        values = []
        for r in records:
            v = r.get(key, default)
            values.append(v if v is not None else default)
        return values

    reward = safe_get(records, "reward")
    reward_std = safe_get(records, "reward_std")
    loss = safe_get(records, "loss")
    kl = safe_get(records, "kl")
    grad_norm = safe_get(records, "grad_norm")
    comp_len = safe_get(records, "completions/mean_length")
    clipped_ratio = safe_get(records, "completions/clipped_ratio")
    frac_zero_std = safe_get(records, "frac_reward_zero_std")
    lr = safe_get(records, "learning_rate")

    # ── Create figure ──
    # Use CJK font for Chinese text rendering
    _setup_font()
    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "figure.dpi": 150,
    })

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle("Lung-R1 KG-Guided GRPO Training Curves", fontsize=14, fontweight="bold")

    colors = plt.cm.tab10(np.linspace(0, 1, 10))

    # 1. Total Reward
    ax = axes[0, 0]
    ax.plot(steps, reward, color=colors[0], linewidth=1.5, marker="o", markersize=3, label="Mean Reward")
    ax.fill_between(steps,
                    [r - s for r, s in zip(reward, reward_std)],
                    [r + s for r, s in zip(reward, reward_std)],
                    alpha=0.2, color=colors[0])
    ax.set_xlabel("Step")
    ax.set_ylabel("Reward")
    ax.set_title("Total Reward (R = lambda1*R_dx + lambda2*R_graph + lambda3*R_path)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 2. Loss
    ax = axes[0, 1]
    ax.plot(steps, loss, color=colors[1], linewidth=1.5, marker="s", markersize=3)
    ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title("GRPO Policy Loss")
    ax.grid(True, alpha=0.3)

    # 3. KL Divergence
    ax = axes[0, 2]
    ax.plot(steps, kl, color=colors[2], linewidth=1.5, marker="^", markersize=3)
    ax.set_xlabel("Step")
    ax.set_ylabel("KL Divergence")
    ax.set_title("KL Divergence (from Reference Model)")
    ax.grid(True, alpha=0.3)

    # 4. Completion Length
    ax = axes[1, 0]
    ax.plot(steps, comp_len, color=colors[3], linewidth=1.5, marker="d", markersize=3)
    ax.set_xlabel("Step")
    ax.set_ylabel("Tokens")
    ax.set_title("Mean Completion Length")
    ax.grid(True, alpha=0.3)

    # 5. Reward Std & Grad Norm
    ax = axes[1, 1]
    ax2 = ax.twinx()
    line1 = ax.plot(steps, reward_std, color=colors[4], linewidth=1.5, marker="v", markersize=3, label="Reward Std")
    line2 = ax2.plot(steps, grad_norm, color=colors[5], linewidth=1.0, linestyle="--", marker="x", markersize=3, label="Grad Norm")
    ax.set_xlabel("Step")
    ax.set_ylabel("Reward Std", color=colors[4])
    ax2.set_ylabel("Grad Norm", color=colors[5])
    ax.set_title("Reward Std & Gradient Norm")
    ax.tick_params(axis="y", labelcolor=colors[4])
    ax2.tick_params(axis="y", labelcolor=colors[5])
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, fontsize=8)
    ax.grid(True, alpha=0.3)

    # 6. Clipped Ratio & Zero-Std Fraction
    ax = axes[1, 2]
    ax.plot(steps, clipped_ratio, color=colors[6], linewidth=1.5, marker="p", markersize=3, label="Clipped Ratio")
    ax.plot(steps, frac_zero_std, color=colors[7], linewidth=1.5, marker="*", markersize=3, label="Zero-Std Fraction")
    ax.set_xlabel("Step")
    ax.set_ylabel("Fraction")
    ax.set_title("Clipped Completions & Zero-Std Groups")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(output_dir, "training_curves.png")
    fig.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"[Plot] Saved training curves to {save_path}")

    # ── Also plot reward components if available ──
    _plot_reward_components(records, output_dir)


def _plot_reward_components(records, output_dir):
    """Plot individual reward components if the data is available."""
    # Check if we have per-component data
    has_components = any("rewards/reward_func/mean" in r for r in records)
    if not has_components:
        return

    steps = [r["step"] for r in records]
    fig, ax = plt.subplots(figsize=(10, 5))

    for key_prefix, label, color in [
        ("rewards/reward_func/mean", "Mean Reward", "blue"),
        ("rewards/reward_func/std", "Reward Std", "red"),
    ]:
        values = [r.get(key_prefix, 0) for r in records]
        if any(v != 0 for v in values):
            ax.plot(steps, values, label=label, color=color, linewidth=1.5, marker="o", markersize=3)

    ax.set_xlabel("Step")
    ax.set_ylabel("Reward")
    ax.set_title("Reward Function Output Over Training")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(output_dir, "reward_components.png")
    fig.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"[Plot] Saved reward components to {save_path}")


# ═══════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Plot training curves from metrics log")
    parser.add_argument("--metrics", type=str, required=True,
                        help="Path to training_metrics.jsonl file")
    parser.add_argument("--output", type=str, default=None,
                        help="Output directory for plots (default: same as metrics file)")
    args = parser.parse_args()

    plot_training_curves(args.metrics, args.output)
