# -*- coding: utf-8 -*-
"""
图 0 · 分析策略总览（第 3 章方法图）
    figs2/fig_M1_flow.png

规范：图内无标题无图注、图内文字全英文、宽 174 mm（6.85 in）通栏、
dpi=600、白底、实线=主路径 / 虚线=对照路径。

初版三处版式缺陷（勿回退）：
  ① 标题与首行正文重叠 —— 框高 0.115 太小、正文按"居中"排版，三行正文顶到标题。
     → 改正文自标题下方按 va="top" 向下排，框高加到 0.135。
  ② RQ1 与 RQ2 框边缘相贴（pad 外扩 0.008 后仅剩 0.02 间隙）→ 三列改为 0.03 / 0.36 / 0.69，宽 0.28。
  ③ "553 → RQ1 复现"那条大弧虚线穿过两条实线与一个文字标签，图面糊成一团
     → 删掉该曲线，改为在 RQ1 框内直接写 "replicated in the 553 (rank only)"。
"""
import sys as _sys
if hasattr(_sys.stdout, "reconfigure"):
    _sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

HERE = os.path.dirname(os.path.abspath(__file__))
FIGDIR = os.path.join(HERE, "figs2")
os.makedirs(FIGDIR, exist_ok=True)
OUT = os.path.join(FIGDIR, "fig_M1_flow.png")
os.makedirs("figs2", exist_ok=True)

C_MAIN = "#B3453A"
C_SEC = "#2E6DA4"
C_GREY = "#3A3A37"
C_FILL = "#FCFCFB"

fig, ax = plt.subplots(figsize=(6.85, 5.05))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")


def box(x, y, w, h, edge, title, lines, fs_body=7.4, fs_title=8.5, gap=1.5):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.007,rounding_size=0.018",
        linewidth=1.15, edgecolor=edge, facecolor=C_FILL,
        transform=ax.transAxes, zorder=2))
    ax.text(x + w / 2, y + h - 0.022, title.strip(), ha="center", va="top",
            fontsize=fs_title, color=edge, fontweight="bold",
            transform=ax.transAxes, zorder=3)
    ax.text(x + w / 2, y + h - 0.062, "\n".join(lines), ha="center", va="top",
            fontsize=fs_body, color=C_GREY, linespacing=gap,
            transform=ax.transAxes, zorder=3)


def arrow(p1, p2, color, dashed=False, rad=0.0):
    ax.add_patch(FancyArrowPatch(
        p1, p2, transform=ax.transAxes, zorder=1,
        arrowstyle="-|>", mutation_scale=8.5,
        linewidth=1.05, color=color,
        linestyle=(0, (3.6, 2.2)) if dashed else "-",
        connectionstyle=f"arc3,rad={rad}",
        shrinkA=0, shrinkB=0))


# ---------- 第一层：原始批次 ----------
Y0, H0 = 0.790, 0.175
box(0.03, Y0, 0.44, H0, C_MAIN, "Main sample (May-Jun)",
    ["N = 1,026",
     "779 paper (screened before entry)",
     "247 online (314 cleaned in this study)"])
box(0.53, Y0, 0.44, H0, C_SEC, "Second batch (Sep)",
    ["N = 553",
     "paper only, engineering-oriented",
     "no Year 4; composition differs by design"])

# ---------- 第二层：数据集 ----------
Y1, H1 = 0.520, 0.175
box(0.03, Y1, 0.44, H1, C_MAIN, "Dataset A",
    ["1,026 rows", "used for RQ1 and RQ2 (main)"])
box(0.53, Y1, 0.44, H1, C_SEC, "Dataset B (pooled)",
    ["1,579 rows", "pooling admissible (Chow test)",
     "batch dummy kept in every model"])

arrow((0.25, Y0), (0.25, Y1 + H1), C_MAIN)
arrow((0.75, Y0), (0.75, Y1 + H1), C_SEC)

# ---------- 第三层：三个研究问题 ----------
Y2, H2 = 0.150, 0.290
W2 = 0.28
box(0.03, Y2, W2, H2, C_MAIN, "RQ1  Profile",
    ["descriptives + Wilson CI",
     "alpha / omega; Kendall's W",
     "EFA: polychoric, parallel",
     "analysis, promax",
     "Q7 dual format",
     "replicated in the 553 (rank only)"],
    fs_body=6.9, gap=1.60)
box(0.36, Y2, W2, H2, C_MAIN, "RQ2  Grade gradient",
    ["confounding: Cramer's V",
     "Jonckheere trend (perm.)",
     "shape diagnostics",
     "ordered logit + Brant",
     "multinomial logit + AME",
     "coarse vs adjusted estimates"],
    fs_body=6.9, gap=1.60)
box(0.69, Y2, W2, H2, C_SEC, "RQ3  Stress to choice",
    ["ordered logit per employer",
     "ipsative check",
     "factor-contrast Wald",
     "AME on nominal choice",
     "bootstrapped indirect link",
     "E-value"],
    fs_body=6.9, gap=1.60)

arrow((0.25, Y1), (0.17, Y2 + H2), C_MAIN, rad=0.08)
arrow((0.25, Y1), (0.48, Y2 + H2), C_MAIN, rad=-0.08)
arrow((0.75, Y1), (0.83, Y2 + H2), C_SEC, rad=-0.08)
arrow((0.585, Y1), (0.52, Y2 + H2), C_SEC, dashed=True, rad=0.02)

# ---------- 底部：全篇统一口径 ----------
ax.text(0.50, 0.092,
        "Whole-text conventions: Benjamini-Hochberg across outcomes  |  HC3 robust SE  |  "
        "z-standardisation within each dataset  |\n"
        "reference groups specified explicitly  |  all outcomes reported, including null ones",
        fontsize=6.7, color=C_GREY, ha="center", va="center",
        transform=ax.transAxes, linespacing=1.5)
ax.text(0.50, 0.032,
        "All estimates are cross-sectional associations; no causal claim is made.",
        fontsize=6.9, color=C_GREY, ha="center", va="center",
        transform=ax.transAxes, style="italic")

fig.savefig(OUT, dpi=600, bbox_inches="tight", facecolor="white")
print("saved", OUT)
