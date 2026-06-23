"""
Generate Fair-RLVR architecture diagram -- side-by-side layout.

Run:  python scripts/generate_architecture_fig.py
Out:  RLVR.png  (replaces the old tall single-column figure)

Verified values (2026-06-20):
  - LoRA r=32, alpha=32  (fair_rlvr/config.json & lambda_sweep.yaml)
  - alpha_consistency=0.0 in ALL actual runs (not 0.25 as stated in some docs)
  - fair_rlvr: 1,000 steps;  lambda_sweep: 750 steps
  - eps_L=0.20, eps_H=0.28, KL coeff=0.01, G=4, seed=42
  - BBQ 58,492 total (52,643 train / 5,849 eval), 9 categories
  - Group-fairness metrics (DPD/EOD/DIR/RB) dropped from paper;
    ASE and three-level CoT faithfulness added instead.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

# ── Palette ──────────────────────────────────────────────────────────────────
C_PANEL_L = "#EEF3FB"    # pale blue  -- left column panels
C_PANEL_R = "#EEFAEE"    # pale green -- right column panels
C_REWARD  = "#FEF8E7"    # pale amber -- composite reward header
C_EVAL    = "#F4F0FA"    # pale lavender -- evaluation header
C_SUB     = "#FAFAFA"    # white-ish  -- sub-boxes
C_WRAP_L  = "#DDE8F6"    # left outer wrapper
C_WRAP_R  = "#DCF0DC"    # right outer wrapper
C_EDGE    = "#606060"
C_ARR     = "#383838"
C_TITLE   = "#111133"
C_BODY    = "#2a2a2a"
C_HINT    = "#666680"

# ── Canvas ────────────────────────────────────────────────────────────────────
FIG_W, FIG_H = 13.4, 7.8
fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
ax.set_xlim(0, FIG_W)
ax.set_ylim(0, FIG_H)
ax.axis("off")
fig.patch.set_facecolor("white")
ax.set_facecolor("white")


# ── Helpers ───────────────────────────────────────────────────────────────────

def rbox(x, y, w, h, fill=C_SUB, edge=C_EDGE, lw=0.9, zorder=2):
    p = FancyBboxPatch((x, y), w, h,
                        boxstyle="round,pad=0.06",
                        linewidth=lw, edgecolor=edge, facecolor=fill, zorder=zorder)
    ax.add_patch(p)


def txt(x, y, text, fs=7.8, fw="normal", color=C_BODY,
        ha="center", va="center", zorder=3):
    ax.text(x, y, text, ha=ha, va=va, fontsize=fs, fontweight=fw,
            color=color, zorder=zorder, clip_on=False)


def sub_box(x, y, w, h, rows, edge="#CCCCCC", lw=0.55, fs=7.5):
    """White sub-box with stacked text rows; first row is bold."""
    rbox(x, y, w, h, fill=C_SUB, edge=edge, lw=lw, zorder=3)
    n = len(rows)
    gap = min(h / (n + 0.25), 0.20)
    top = y + h / 2 + (n - 1) / 2 * gap
    for i, row in enumerate(rows):
        fw = "bold" if i == 0 else "normal"
        col = C_TITLE if i == 0 else C_BODY
        txt(x + w / 2, top - i * gap, row, fs=fs, fw=fw, color=col, zorder=4)


def titled_box(x, y, w, h, title, body_rows,
               fill=C_PANEL_L, edge=C_EDGE, lw=1.0,
               title_fs=8.5, body_fs=7.1):
    """Panel with bold title and body rows, auto-spaced."""
    rbox(x, y, w, h, fill=fill, edge=edge, lw=lw)
    all_rows = [title] + body_rows
    n = len(all_rows)
    gap = min(h / (n + 0.3), 0.22)
    top = y + h / 2 + (n - 1) / 2 * gap
    for i, row in enumerate(all_rows):
        fs = title_fs if i == 0 else body_fs
        fw = "bold" if i == 0 else "normal"
        col = C_TITLE if i == 0 else C_BODY
        txt(x + w / 2, top - i * gap, row, fs=fs, fw=fw, color=col)


def arr(x1, y1, x2, y2, rad=0.0, lw=1.3, color=C_ARR):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color=color, lw=lw,
                                connectionstyle=f"arc3,rad={rad}"),
                zorder=5)


# ====================================================================
#  LEFT COLUMN   x: 0.25 -> 5.85
# ====================================================================
LX, LW = 0.25, 5.60

# Outer wrapper
rbox(LX - 0.12, 0.18, LW + 0.24, 7.44, fill=C_WRAP_L, edge="#AABBCC", lw=0.5, zorder=1)

# -- 1. INPUTS (y: 5.90 -> 7.10) --
IY, IH = 5.90, 1.20
rbox(LX, IY, LW, IH, fill=C_PANEL_L, edge=C_EDGE, lw=1.0)
txt(LX + LW / 2, IY + IH - 0.14, "Inputs", fs=9.0, fw="bold", color=C_TITLE)

sub_iw = (LW - 0.38) / 2
sub_box(LX + 0.14, IY + 0.10, sub_iw, 0.82,
        ["Qwen2.5-3B-Instruct", "+LoRA  r=32,  alpha=32,  bf16"])
sub_box(LX + 0.14 + sub_iw + 0.10, IY + 0.10, sub_iw, 0.82,
        ["BBQ Dataset", "58,492 QA  x  9 categories"])

arr(LX + LW / 2, IY, LX + LW / 2, IY - 0.35)

# -- 2. TEMPLATE-FAMILY SPLIT (y: 5.00 -> 5.65) --
TY, TH = 5.00, 0.68
titled_box(LX, TY, LW, TH,
           "Template-Family Split",
           ["90% Train  x  10% Eval  x  seed=42",
            "group by (category, question_index)  ->  no near-duplicate leakage"],
           fill=C_PANEL_L, title_fs=8.5, body_fs=7.1)

arr(LX + LW / 2, TY, LX + LW / 2, TY - 0.36)

# -- 3. GRPO SAMPLING (y: 4.02 -> 4.64) --
SY, SH = 4.02, 0.64
titled_box(LX, SY, LW, SH,
           "GRPO Sampling   G=4",
           ["FamilyGroupedSampler -- siblings kept adjacent",
            "Output:  <think> reasoning </think> <answer> (a/b/c) </answer>"],
           fill=C_PANEL_L, title_fs=8.5, body_fs=7.1)

arr(LX + LW / 2, SY, LX + LW / 2, SY - 0.32)

# -- 4. GRPO UPDATE (y: 0.30 -> 3.72) --
UY, UH = 0.30, 3.44
rbox(LX, UY, LW, UH, fill=C_PANEL_L, edge=C_EDGE, lw=1.0)
txt(LX + LW / 2, UY + UH - 0.16, "GRPO Update", fs=9.0, fw="bold", color=C_TITLE)

sgw = (LW - 0.38) / 2
sgh = 1.35
gx = LX + 0.14
# top row
sub_box(gx,             UY + 1.82, sgw, sgh,
        ["Normalize", r"$A_i = (R_i - \mu)\;/\;\sigma$"])
sub_box(gx + sgw + 0.10, UY + 1.82, sgw, sgh,
        ["DAPO Clip", r"$\varepsilon_L$=0.20   $\varepsilon_H$=0.28"])
# bottom row
sub_box(gx,             UY + 0.32, sgw, sgh,
        ["KL Penalty", "coeff = 0.01"])
sub_box(gx + sgw + 0.10, UY + 0.32, sgw, sgh,
        ["Update Weights"])

# Back-loop arrow (GRPO Update left edge -> GRPO Sampling left edge)
ax.annotate("", xy=(LX, SY + SH * 0.40),
            xytext=(LX, UY + UH * 0.50),
            arrowprops=dict(arrowstyle="->", color=C_ARR, lw=1.3,
                            connectionstyle="arc3,rad=0"), zorder=5)
txt(LX - 0.08, (SY + SH * 0.40 + UY + UH * 0.50) / 2,
    "next\nstep", fs=6.2, color=C_HINT, ha="right")


# ====================================================================
#  RIGHT COLUMN   x: 6.10 -> 12.85
# ====================================================================
RX, RW = 6.10, 6.75

# Outer wrapper
rbox(RX - 0.12, 0.18, RW + 0.24, 7.44, fill=C_WRAP_R, edge="#AACCAA", lw=0.5, zorder=1)

# -- 5. COMPOSITE REWARD (y: 4.38 -> 7.10) --
CRY, CRH = 4.38, 2.72
rbox(RX, CRY, RW, CRH, fill=C_REWARD, edge=C_EDGE, lw=1.0)
txt(RX + RW / 2, CRY + CRH - 0.16,
    "Composite Reward", fs=9.0, fw="bold", color=C_TITLE)
txt(RX + RW / 2, CRY + CRH - 0.42,
    r"$R = \lambda \cdot R^{f} + \alpha \cdot R^{c} - P^{s}$",
    fs=9.0, fw="bold", color="#222244")
txt(RX + RW / 2, CRY + CRH - 0.68,
    r"$\lambda$=0.5,  $\alpha$=0    ($\alpha$=0.25 activates $R^c$ consistency bonus)",
    fs=7.2, color=C_HINT)

# Three reward sub-boxes
srw = (RW - 0.40) / 3 - 0.067
srh = 1.58
for i, rows in enumerate([
    [r"$R^f$ Fairness",     "BBQ label match", "+1.0  /  0.0"],
    [r"$R^c$ Consistency",  "Sibling answer match", "+1.0  /  0.0"],
    [r"$P^s$ Structural",   "4 violations x -0.3", "max penalty  -1.2"],
]):
    sub_box(RX + 0.14 + i * (srw + 0.10), CRY + 0.58, srw, srh, rows)

arr(RX + RW / 2, CRY, RX + RW / 2, CRY - 0.36)

# -- 6. EVALUATION (y: 0.30 -> 4.02) --
EY, EH = 0.30, 3.72
rbox(RX, EY, RW, EH, fill=C_EVAL, edge=C_EDGE, lw=1.0)
txt(RX + RW / 2, EY + EH - 0.16, "Evaluation", fs=9.0, fw="bold", color=C_TITLE)

eval_rows = [
    ["BBQ Accuracy", "Ambiguous (BBQ-A)  |  Disambiguated (BBQ-D)"],
    ["Bias Score  &  ASE",
     "stereotype errors / total (down)   |   (1-BBQ-A) x Bias (down)"],
    ["CoT Faithfulness (3-level)",
     "real CoT  |  permuted CoT  |  null CoT   ->  causal sensitivity"],
    ["OOD Generalization",
     "WinoBias  |  StereoSet  |  Intersectional BBQ (race x gender, race x SES)"],
]
e_h  = 0.70
e_gap = 0.09
total_e = len(eval_rows) * e_h + (len(eval_rows) - 1) * e_gap
e_start = EY + (EH - total_e - 0.16) / 2 + 0.05

for i, rows in enumerate(eval_rows):
    ey_i = e_start + (len(eval_rows) - 1 - i) * (e_h + e_gap)
    sub_box(RX + 0.18, ey_i, RW - 0.36, e_h, rows)


# ====================================================================
#  CROSS ARROWS  (left <-> right)
# ====================================================================
# GRPO Sampling -> Composite Reward
arr(LX + LW, SY + SH / 2,
    RX, CRY + CRH * 0.62,
    rad=-0.20)
txt(5.97, 5.12, "completions ->", fs=6.8, color=C_HINT)

# Composite Reward -> GRPO Update
arr(RX, CRY + 0.55,
    LX + LW, UY + UH * 0.48,
    rad=0.30)
txt(5.97, 2.82, "<- reward signals", fs=6.8, color=C_HINT)


# ── Save ──────────────────────────────────────────────────────────────────────
plt.tight_layout(pad=0)
fig.savefig("RLVR.png", dpi=160, bbox_inches="tight", facecolor="white")
print("Saved RLVR.png")
