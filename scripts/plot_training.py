"""
Generate training trajectory plots from Fair-RLVR batch_logs.json.

Usage:
    python scripts/plot_training.py --log results/fair_rlvr/logs/batch_logs.json
    python scripts/plot_training.py --log results/fair_rlvr/logs/batch_logs.json --out results/fair_rlvr/figures
"""

import json
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


def smooth(values, window=10):
    """Simple moving average."""
    result = []
    for i, v in enumerate(values):
        lo = max(0, i - window // 2)
        hi = min(len(values), i + window // 2 + 1)
        result.append(sum(values[lo:hi]) / (hi - lo))
    return result


def load_logs(path):
    with open(path) as f:
        return json.load(f)


def plot_trajectory(logs, out_dir: Path, smooth_window: int = 15):
    out_dir.mkdir(parents=True, exist_ok=True)

    steps            = [e["step"] for e in logs]
    accuracy         = [e["accuracy"] for e in logs]
    stereo_rate      = [e["stereotype_pick_rate_ambig"] for e in logs]
    abstention       = [e["abstention_rate"] for e in logs]
    r_fairness       = [e["avg_r_fairness"] for e in logs]
    r_total          = [e["avg_r_total"] for e in logs]
    p_structural     = [e["avg_p_structural"] for e in logs]
    format_fail      = [e.get("format_failure_rate", 0) for e in logs]

    # ── Figure 1: Main fairness metrics ─────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(9, 6), tight_layout=True)
    fig.suptitle("Fair-RLVR Training Trajectory", fontsize=13, fontweight="bold")

    panels = [
        (axes[0, 0], accuracy,     smooth(accuracy, smooth_window),
         "Batch Accuracy", "Accuracy", "steelblue"),
        (axes[0, 1], stereo_rate,  smooth(stereo_rate, smooth_window),
         "Stereotype Pick Rate (Ambig)", "Rate", "crimson"),
        (axes[1, 0], abstention,   smooth(abstention, smooth_window),
         "Abstention Rate", "Rate", "darkorange"),
        (axes[1, 1], r_total,      smooth(r_total, smooth_window),
         "Mean Total Reward", "Reward", "seagreen"),
    ]

    for ax, raw, sm, title, ylabel, color in panels:
        ax.plot(steps, raw, color=color, alpha=0.25, linewidth=0.8)
        ax.plot(steps, sm,  color=color, linewidth=2.0, label="smoothed")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Training Step")
        ax.set_ylabel(ylabel)
        ax.set_xlim(left=0)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
        ax.grid(True, alpha=0.3)

    path1 = out_dir / "training_trajectory.pdf"
    fig.savefig(path1, dpi=200, bbox_inches="tight")
    fig.savefig(path1.with_suffix(".png"), dpi=200, bbox_inches="tight")
    print(f"Saved: {path1}")
    plt.close(fig)

    # ── Figure 2: Reward components ─────────────────────────────────────────
    fig2, axes2 = plt.subplots(1, 3, figsize=(11, 3.5), tight_layout=True)
    fig2.suptitle("Reward Component Breakdown", fontsize=13, fontweight="bold")

    comp_panels = [
        (axes2[0], r_fairness,   smooth(r_fairness, smooth_window),
         "R_fairness", "steelblue"),
        (axes2[1], p_structural, smooth(p_structural, smooth_window),
         "P_structural (penalty)", "firebrick"),
        (axes2[2], format_fail,  smooth(format_fail, smooth_window),
         "Format Failure Rate", "darkorange"),
    ]

    for ax, raw, sm, title, color in comp_panels:
        ax.plot(steps, raw, color=color, alpha=0.25, linewidth=0.8)
        ax.plot(steps, sm,  color=color, linewidth=2.0)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Training Step")
        ax.set_xlim(left=0)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
        ax.grid(True, alpha=0.3)

    path2 = out_dir / "reward_components.pdf"
    fig2.savefig(path2, dpi=200, bbox_inches="tight")
    fig2.savefig(path2.with_suffix(".png"), dpi=200, bbox_inches="tight")
    print(f"Saved: {path2}")
    plt.close(fig2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", default="results/fair_rlvr/logs/batch_logs.json",
                        help="Path to batch_logs.json")
    parser.add_argument("--out", default="results/fair_rlvr/figures",
                        help="Output directory for figures")
    parser.add_argument("--smooth", type=int, default=15,
                        help="Smoothing window size (steps)")
    args = parser.parse_args()

    logs = load_logs(args.log)
    print(f"Loaded {len(logs)} steps from {args.log}")
    plot_trajectory(logs, Path(args.out), smooth_window=args.smooth)


if __name__ == "__main__":
    main()
