# -*- coding: utf-8 -*-
"""
RQ1 合并口径 · 五张对比图（重制版）
==================================
step28_rq1_pooled.py 负责统计与报告；本脚本只负责出图，便于反复目视迭代。

⚠ 初版五张图全部有版式缺陷，本版逐张修正（教训写在各段注释里）：
  图 P1  四面板共用一套图例 → 颜色含义不同会误导；改各面板独立图例。
  图 P2  Δ 标签贴在整panel最右端，与数据行脱节；图例压第一行；
         改上下两栏 + 轴外右侧对齐 + 图例并入副标题。
  图 P3  五个指标量纲不同（1.35–3.85）却共用一个 x 轴，
         「CI 重叠」这个核心信息在图上完全看不出；改 2×3 小面板，各面板独立 x 轴。
  图 P4  图例压第一行；三 panel 轴范围不同导致跨组误读；改统一 0–0.55 + 右侧标数值。
  图 P5  左图 1026 与 1579 的柱子画在**同一 x 位置**（后者覆盖前者）；
         图例未区分颜色含义；改为 4 根柱分别偏移 + 双图例。

用法：E:/Python/python.exe step28b_plots_rq1.py
"""
import os
import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "论文二_输出")
FIG = os.path.join(BASE, "figs2")
XL = os.path.join(OUT, "RQ1_合并口径结果.xlsx")

Z95 = float(norm.ppf(0.975))
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 9

C_A, C_B = "#B4573C", "#185FA5"
GREY = "#8C8C89"


def log(m):
    print(f"  {m}", flush=True)


d1 = pd.read_excel(os.path.join(BASE, "论文二数据集_1026.xlsx"))
d2 = pd.read_excel(os.path.join(BASE, "论文二数据集_1579.xlsx"))
d3 = d2.iloc[1026:].reset_index(drop=True)
SETS = [("1026", d1), ("1579", d2), ("553", d3)]

sh = pd.read_excel(XL, sheet_name=None)
t_q4 = sh["Q4题项对比"]
t_q10 = sh["Q10题项对比"]
t_core = sh["核心指标对比"]
t_mr = sh["多重响应对比"]
t_q7 = sh["Q7双口径对比"]
# ⚠ Excel 读回来 "1026"/"1579"/"553" 会被 pandas 转成 int64，
#   之后 t_q7["集"] == "1026" 恒为空 → bar 一根都画不出来（初版就栽在这）。
t_q7["集"] = t_q7["集"].astype(str)
p11 = sh["Q11七项对比"]

Q11_ITEMS = [f"Q11_{i}" for i in range(1, 8)]
Q11_EN = {"Q11_1": "Salary expectations", "Q11_2": "Location & working hours",
          "Q11_3": "Workload", "Q11_4": "Job-content interest",
          "Q11_5": "Employer size & reputation", "Q11_6": "Career prospects",
          "Q11_7": "Work atmosphere"}
INTENT_ORDER = ["深造", "直接就业", "求稳体制", "方向未定"]
INTENT_EN = {"深造": "Further study", "直接就业": "Direct employment",
             "求稳体制": "Public-sector stability", "方向未定": "Undecided"}

# ======================================================================
# 图 P1 · 样本构成（分组柱，类别名落在 x 轴上）
# ----------------------------------------------------------------------
# ⚠ 两版踩坑记录：
#   初版  四面板共用一套图例 → 颜色在四个面板里含义完全不同，读者会把
#         Major 的蓝条当成 Year 1；
#   第二版 给每个面板加独立图例（bbox 放到面板下方）→ 条带图把整幅宽度占满，
#         图例被下一行坐标轴的白色底patch盖住，最终只有 Grade 一个图例可见，
#         另外三个面板**完全没有类别标签**（代码不报错，必须看图才发现）。
#   本版  改为分组柱：类别名直接作为 x 轴刻度，不再依赖图例表达类别；
#         唯一的图例只表达「三组样本」这一层，而这一层在四个面板里含义一致。
# ======================================================================
GRADE_EN = {"y1": "Year 1", "y2": "Year 2", "y3": "Year 3", "y4": "Year 4",
            "grad": "Postgrad"}
MAJOR_EN = {"eng": "Engineering", "sci": "Sciences", "econ": "Econ & mgmt",
            "hum": "Humanities", "oth": "Other"}
_PAN = [("grade", ["y1", "y2", "y3", "y4", "grad"], GRADE_EN, "(a)  Grade"),
        ("major", ["eng", "sci", "econ", "hum", "oth"], MAJOR_EN, "(b)  Major"),
        ("sex", ["male", "female"], {"male": "Male", "female": "Female"}, "(c)  Sex"),
        ("channel", ["paper", "net"], {"paper": "Paper", "net": "Online"},
         "(d)  Channel")]
_C_SET = [C_A, C_B, "#8C8C89"]          # 1026 / 1579 / 553

fig, axes = plt.subplots(2, 2, figsize=(6.85, 5.9), sharey=True)
for ax, (col, lvs, en, title) in zip(axes.ravel(), _PAN):
    xx = np.arange(len(lvs))
    _w = 0.26
    for j, (_, dd) in enumerate(SETS):
        vals = np.array([float((dd[col] == lv).mean()) * 100 for lv in lvs])
        _xs = xx + (j - 1) * _w
        ax.bar(_xs, vals, width=_w * 0.92, color=_C_SET[j],
               edgecolor="white", linewidth=0.5, zorder=2)
        for xi, v in zip(_xs, vals):
            ax.text(xi, v + 1.2, f"{v:.0f}", ha="center", va="bottom",
                    fontsize=5.6, color="#3A3A37", rotation=90, zorder=3)
    ax.set_xticks(xx)
    ax.set_xticklabels([en[l] for l in lvs], fontsize=7.0, rotation=22, ha="right")
    ax.set_ylim(0, 108)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_title(title, fontsize=9.2, loc="left", pad=4)
    ax.tick_params(labelsize=7.6)
    ax.grid(axis="y", ls=":", color="#DCDCD8", lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
axes[0, 0].set_ylabel("Share of sample (%)", fontsize=8.2)
axes[1, 0].set_ylabel("Share of sample (%)", fontsize=8.2)
_hl = [mpatches.Patch(color=_C_SET[0], label="1026  main sample"),
       mpatches.Patch(color=_C_SET[1], label="1579  pooled sample"),
       mpatches.Patch(color=_C_SET[2], label="553  added batch (only)")]
fig.legend(handles=_hl, loc="lower center", ncol=3, fontsize=7.8, frameon=False,
           bbox_to_anchor=(0.5, -0.005), handlelength=1.1, columnspacing=1.6,
           handletextpad=0.5)
fig.subplots_adjust(hspace=0.60, wspace=0.13, top=0.965, bottom=0.155,
                    left=0.095, right=0.985)
fig.savefig(os.path.join(FIG, "fig_RQ1_pooled_composition.png"),
            dpi=720, bbox_inches="tight", facecolor="white")
plt.close(fig)
log("图 P1 构成 ✓")

# ======================================================================
# 图 P2 · 排序哑铃（上下两栏，Δ 放轴外右侧）
# ======================================================================
fig, axes = plt.subplots(2, 1, figsize=(6.85, 6.9))
for ax, (t, title) in zip(axes, [(t_q4, "(a)  Q4 stress sources (nine items)"),
                                 (t_q10, "(b)  Q10 employer-type preferences (eight items)")]):
    t = t.sort_values("均值_1579", ascending=True).reset_index(drop=True)
    yy = np.arange(len(t))
    ax.plot([t["均值_1026"], t["均值_1579"]], [yy, yy],
            color="#C4C4C0", lw=1.2, zorder=1, solid_capstyle="round")
    ax.scatter(t["均值_1026"], yy, s=52, facecolor="white", edgecolor=C_A,
               linewidth=1.6, zorder=2)
    ax.scatter(t["均值_1579"], yy, s=30, color=C_B, zorder=3)
    ax.set_yticks(yy)
    ax.set_yticklabels(list(t["English"]), fontsize=8.4)
    _lo = float(min(t["均值_1026"].min(), t["均值_1579"].min()))
    _hi = float(max(t["均值_1026"].max(), t["均值_1579"].max()))
    _pad = max((_hi - _lo) * 0.18, 0.04)
    ax.set_xlim(_lo - _pad, _hi + _pad)
    ax.set_xlabel("Mean (1–5 scale)", fontsize=8.4)
    ax.set_title(title, fontsize=9.8, loc="left", pad=4)
    ax.tick_params(labelsize=8)
    ax.grid(axis="x", ls=":", color="#DCDCD8", lw=0.7)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    # Δ 与名次变化贴在该行右侧（轴外），不再贴整 panel 最右端
    for i, r in t.iterrows():
        _rank = "" if r["Δ排名"] == 0 else f"  [{r['排名_1026']}→{r['排名_1579']}]"
        ax.text(1.015, i, f"{r['Δ均值']:+.3f}{_rank}", va="center", ha="left",
                fontsize=7.6, color="#4A4A47",
                transform=ax.get_yaxis_transform())
handles = [plt.Line2D([], [], marker="o", ls="", mfc="white", mec=C_A, mew=1.6,
                      ms=8, label="1026 (main sample)"),
           plt.Line2D([], [], marker="o", ls="", mfc=C_B, mec=C_B, ms=5.5,
                      label="1579 (pooled)")]
fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=8.4,
           frameon=False, bbox_to_anchor=(0.48, -0.012))
fig.subplots_adjust(hspace=0.42, top=0.975, bottom=0.085, left=0.235, right=0.735)
fig.savefig(os.path.join(FIG, "fig_RQ1_pooled_rank.png"),
            dpi=720, bbox_inches="tight", facecolor="white")
plt.close(fig)
log("图 P2 排序 ✓")

# ======================================================================
# 图 P3 · 核心指标（2×3 小面板，各自独立 x 轴）
# ======================================================================
fig, axes = plt.subplots(2, 3, figsize=(6.85, 4.3))
for ax, (_, r) in zip(axes.ravel(), t_core.iterrows()):
    a = pd.to_numeric(d1[r["指标"]], errors="coerce").dropna()
    b = pd.to_numeric(d2[r["指标"]], errors="coerce").dropna()
    ma, mb = a.mean(), b.mean()
    ea = Z95 * a.std() / np.sqrt(len(a))
    eb = Z95 * b.std() / np.sqrt(len(b))
    ax.plot([ma - ea, ma + ea], [1, 1], color=C_A, lw=2.0, solid_capstyle="butt")
    ax.plot([mb - eb, mb + eb], [0, 0], color=C_B, lw=2.0, solid_capstyle="butt")
    ax.scatter([ma], [1], s=34, color=C_A, zorder=3)
    ax.scatter([mb], [0], s=34, color=C_B, zorder=3)
    _lo = min(ma - ea, mb - eb)
    _hi = max(ma + ea, mb + eb)
    _pad = max((_hi - _lo) * 0.55, 0.02)
    ax.set_xlim(_lo - _pad, _hi + _pad)
    ax.set_ylim(-0.75, 1.75)
    ax.set_yticks([1, 0])
    ax.set_yticklabels(["1026", "1579"], fontsize=8)
    ax.set_xticks([_lo, (_lo + _hi) / 2, _hi])
    ax.set_xticklabels([f"{v:.2f}" for v in (_lo, (_lo + _hi) / 2, _hi)], fontsize=7.4)
    ax.set_title(f"{r['English']}\nd = {r['Cohen_d']:+.3f} ({r['量级']})",
                 fontsize=8.4, pad=4)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(labelsize=7.6)
    ax.grid(axis="x", ls=":", color="#DCDCD8", lw=0.6)
    ax.set_axisbelow(True)
ax = axes.ravel()[-1]
ax.axis("off")
# 小面板高度有限（约 1.3 in），说明文字必须拆成短块，每块不超 2 行；
# 初版写成一整段 5 行，直接占掉一半高度并与下面几块重叠。
ax.text(0.0, 0.93, "All |d| < 0.05", fontsize=8.2, color=C_A,
        fontweight="bold", transform=ax.transAxes, va="top")
# ⚠ 四块说明之间必须留出可见空隙：三块两行文本若 y 只差 0.20，第二行会紧贴
#   下一块的第一行，读者会把「in every index」和「Bars = 95% CI」读成一句。
for _yy, _txt in [(0.78, ["negligible shift", "in every index"]),
                  (0.50, ["Bars = 95% CI", "Dots = mean"]),
                  (0.22, ["Own x-axis per panel:", "indices differ in scale"])]:
    ax.text(0.0, _yy, chr(10).join(_txt), fontsize=7.2, color="#3A3A37",
            transform=ax.transAxes, va="top", linespacing=1.45)
fig.subplots_adjust(hspace=0.62, wspace=0.38, top=0.925, bottom=0.10,
                    left=0.075, right=0.975)
fig.savefig(os.path.join(FIG, "fig_RQ1_pooled_indices.png"),
            dpi=720, bbox_inches="tight", facecolor="white")
plt.close(fig)
log("图 P3 核心指标 ✓")

# ======================================================================
# 图 P4 · 多重响应（统一 x 轴 + 右侧标数值）
# ======================================================================
fig, axes = plt.subplots(1, 3, figsize=(6.85, 5.4))
_TT = {"Q6 优势": "Q6 perceived advantages", "Q8 反应": "Q8 stress reactions",
       "Q9 应对": "Q9 coping strategies"}
for ax, lbl in zip(axes, ["Q6 优势", "Q8 反应", "Q9 应对"]):
    g = t_mr[t_mr["题组"] == lbl].sort_values("率_1579", ascending=True).reset_index(drop=True)
    yy = np.arange(len(g))
    ax.plot([g["率_1026"], g["率_1579"]], [yy, yy],
            color="#C4C4C0", lw=1.2, zorder=1, solid_capstyle="round")
    ax.scatter(g["率_1026"], yy, s=46, facecolor="white", edgecolor=C_A,
               linewidth=1.5, zorder=2)
    ax.scatter(g["率_1579"], yy, s=26, color=C_B, zorder=3)
    ax.set_yticks(yy)
    ax.set_yticklabels(list(g["English"]), fontsize=7.4)
    # ⚠ 上限必须覆盖最高项（Q6「价值观正确」达 77%）；初版设 0.62 把它裁到轴外，
    #   图上只剩数值标签、看不到点。
    ax.set_xlim(0, 0.88)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8])
    ax.set_xticklabels(["0", "20", "40", "60", "80"])
    ax.set_xlabel("Selection rate (%)", fontsize=8.0)
    ax.set_title(_TT[lbl], fontsize=9.0, loc="left", pad=4)
    ax.tick_params(labelsize=7.4)
    ax.grid(axis="x", ls=":", color="#DCDCD8", lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    # 数值标在点正上方（轴外固定 x 会串到相邻 panel 的标签区）
    for i, r in g.iterrows():
        ax.annotate(f"{r['率_1579']*100:.1f}", xy=(r["率_1579"], i),
                    xytext=(0, 6.5), textcoords="offset points",
                    ha="center", va="bottom", fontsize=6.4, color="#5A5A57")
handles = [plt.Line2D([], [], marker="o", ls="", mfc="white", mec=C_A, mew=1.5,
                      ms=7.5, label="1026 (main)"),
           plt.Line2D([], [], marker="o", ls="", mfc=C_B, mec=C_B, ms=5,
                      label="1579 (pooled)")]
fig.legend(handles=handles, loc="upper right", ncol=2, fontsize=8.2,
           frameon=False, bbox_to_anchor=(0.995, 0.982))
fig.subplots_adjust(top=0.885, bottom=0.10, left=0.205, right=0.985, wspace=1.05)
fig.savefig(os.path.join(FIG, "fig_RQ1_pooled_multiresp.png"),
            dpi=720, bbox_inches="tight", facecolor="white")
plt.close(fig)
log("图 P4 多重响应 ✓")

# ======================================================================
# 图 P5 · Q7 双口径 + Q11
# ======================================================================
fig, axes = plt.subplots(1, 2, figsize=(6.85, 4.3))

ax = axes[0]
g = t_q7[t_q7["集"] != "553"]
# ⚠ 类别名第四版才定（前三版全部重叠）：
#   ① rotation=16 + 全称 → 相邻标签叠字；
#   ② 两行全称 → "Public-sector / stability" 宽 0.63 in > 每组可用 0.60 in，仍叠；
#   ③ 两行短名 + 6.1 pt → "employment" 与 "Public" 贴边。
#   根因：左面板只有约 2.4 in 宽，四类每组 0.60 in，靠压宽度无解。
#   最终：**单行短名 + 倾斜 30°**，让标签沿纵向错开。
_labs = ["Further study", "Direct employment", "Public sector", "Undecided"]
xx = np.arange(len(_labs))
_w = 0.19
# ⚠ 四根柱必须各占一个位置：初版把 1026 与 1579 画在同一 x，后者覆盖前者
# 四种柱：实心=单选、白底斜线=多选；砖红=1026、蓝=1579
_OFF = [(("1026", "单选"), -1.5, C_A, C_A, None),
        (("1026", "多选"), -0.5, C_A, "white", "////"),
        (("1579", "单选"), 0.5, C_B, C_B, None),
        (("1579", "多选"), 1.5, C_B, "white", "////")]
for (nm, kind), off, edge, fill, hatch in _OFF:
    sub = g[g["集"] == nm].set_index("类别").reindex(INTENT_ORDER)
    colname = "单选_%" if kind == "单选" else "多选_%"
    ax.bar(xx + off * _w, sub[colname], width=_w, facecolor=fill,
           edgecolor=edge, linewidth=1.2, hatch=hatch)
ax.set_xticks(xx)
ax.set_xticklabels(_labs, fontsize=6.6, rotation=30, ha="right",
                   rotation_mode="anchor")
ax.set_ylabel("Share of respondents (%)", fontsize=8.2)
ax.set_title("(a)  Q7: single choice vs multiple response", fontsize=9.4,
             loc="left", pad=4)
ax.set_ylim(0, 68)
ax.tick_params(labelsize=8)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
_h1 = [mpatches.Patch(color=C_A), mpatches.Patch(color=C_B)]
_h2 = [mpatches.Patch(facecolor="#DDDDDD", edgecolor="#5F5E5A", label="Single"),
       mpatches.Patch(facecolor="#DDDDDD", edgecolor="#5F5E5A", hatch="///",
                      label="Multiple")]
_lg1 = ax.legend(_h1, ["1026", "1579"], fontsize=7.6, frameon=False,
                 loc="upper right", title="Sample", title_fontsize=7.6)
ax.add_artist(_lg1)
ax.legend(_h2, ["Single choice", "Multiple response"], fontsize=7.6, frameon=False,
          loc="upper center", bbox_to_anchor=(0.52, -0.215), ncol=2)

ax = axes[1]
g = p11.sort_values("均值_1579", ascending=True).reset_index(drop=True)
yy = np.arange(len(g))
ax.plot([g["均值_1026"], g["均值_1579"]], [yy, yy],
        color="#C4C4C0", lw=1.2, zorder=1, solid_capstyle="round")
ax.scatter(g["均值_1026"], yy, s=48, facecolor="white", edgecolor=C_A,
           linewidth=1.5, zorder=2)
ax.scatter(g["均值_1579"], yy, s=26, color=C_B, zorder=3)
ax.set_yticks(yy)
ax.set_yticklabels(list(g["English"]), fontsize=7.8)
_lo, _hi = float(g["均值_1026"].min()), float(g["均值_1026"].max())
_pad = (_hi - _lo) * 0.16
ax.set_xlim(_lo - _pad, _hi + _pad)
ax.set_xlabel("Mean (0–2)", fontsize=8.2)
ax.set_title("(b)  Q11: expectations affected by pressure", fontsize=9.4,
             loc="left", pad=4)
ax.tick_params(labelsize=8)
ax.grid(axis="x", ls=":", color="#DCDCD8", lw=0.6)
ax.set_axisbelow(True)
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
for i, r in g.iterrows():
    ax.text(1.02, i, f"{r['Δ均值']:+.3f}", va="center", ha="left", fontsize=7.2,
            color="#4A4A47", transform=ax.get_yaxis_transform())
fig.subplots_adjust(top=0.945, bottom=0.215, left=0.085, right=0.915, wspace=0.95)
fig.savefig(os.path.join(FIG, "fig_RQ1_pooled_intent.png"),
            dpi=720, bbox_inches="tight", facecolor="white")
plt.close(fig)
log("图 P5 Q7/Q11 ✓")
print("五张图重制完成。")
