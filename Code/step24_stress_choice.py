# -*- coding: utf-8 -*-
r"""
Step 12–14 · 研究问题三（前半）：压力源 → 就业选择
======================================================================
输入：论文二数据集_1579.xlsx（Step 1 产出；含 Step 6 的 F1 / F2）
输出：论文二_输出/Step12_14_压力与选择结果.xlsx
      figs2/fig_RQ3_forest.png    图 9   每 1 SD 压力的调整 OR 森林图
      figs2/fig_RQ3_factor.png    图 11  因子 × 偏好的 OR 热图 + 差异检验

⚠ 卡点 7 口径（硬性，全文执行）
  · 样本 1579；协变量含 batch；只用 channel + batch，禁止三分类「来源」、禁止 batch×channel
  · 连续自变量按 1579 集内 SD 标准化（数据集里的 z_* 与 F* 均已按 1579 内算好，直接取用）
  · Q11 先 replace(99, nan)（本脚本的 Q11 披露段用到）
  · 稳健标准误 HC3

⚠ 措辞红线：associated with / 不得写「导致」「effect」「changes over time」
"""
import sys as _sys
if hasattr(_sys.stdout, "reconfigure"):
    _sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import os
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy import stats
from statsmodels.miscmodels.ordinal_model import OrderedModel

warnings.filterwarnings("ignore")

# ======================================================================
# 0. 全局设定
# ======================================================================
BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "论文二_输出")
FIG = os.path.join(BASE, "figs2")
for d in (OUT, FIG):
    os.makedirs(d, exist_ok=True)

F1579 = os.path.join(BASE, "论文二数据集_1579.xlsx")
XLSX = os.path.join(OUT, "Step12_14_压力与选择结果.xlsx")

plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["pdf.fonttype"] = 42
DPI = 600
SEED = 42
Z975 = 1.959963984540054

# 九个结局：八个 Q10 分题 + 一个复合偏好
# ⚠ Q10_体制内偏好 ≡ Q10_1（实测 np.allclose 为真），故不单列，避免同一结局出两行
OUTCOMES = [
    ("Q10_1",             "国企 / 事业单位", "State-owned / public inst.",  "ord5"),
    ("Q10_2",             "外资企业",        "Foreign-funded enterprise",   "ord5"),
    ("Q10_3",             "民营企业（大型）", "Large private enterprise",    "ord5"),
    ("Q10_4",             "民营企业（中小型）", "Small / medium private",    "ord5"),
    ("Q10_5",             "自主创业",        "Entrepreneurship",            "ord5"),
    ("Q10_6",             "自由职业",        "Freelance work",              "ord5"),
    ("Q10_7",             "新业态从业",      "New-form employment",         "ord5"),
    ("Q10_8",             "基层就业",        "Grassroots employment",       "ord5"),
    ("Q10_灵活新就业偏好", "灵活 / 新就业偏好（复合）", "Flexible / new-form pref.", "cont"),
]
GRADE_ORDER = ["y1", "y2", "y3", "y4", "grad"]
MAJ_ORDER = ["eng", "sci", "econ", "hum", "oth"]
Z_STRESS = "z_压力指数"
ZF1, ZF2 = "F1", "F2"          # 已是 1579 集内 z 化的单元加权得分

def log(msg=""):
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)

# ======================================================================
# 1. 统计工具
# ======================================================================
def _grab_from_step23():
    """从 step23_grade_trend.py 提取已做过 Monte Carlo 校准的公用实现。

    为什么用 exec 而不是复制：Brant 检验刚修过两个静默 bug（切点模型缺截距、
    协方差误用 block_diag），JT 与 Brant 的校准脚本也直接从主脚本取用。
    复制一份等于埋一颗"两处实现可能漂移"的雷。
    """
    src_path = os.path.join(BASE, "step23_grade_trend.py")
    with open(src_path, "r", encoding="utf-8") as f:
        src = f.read()

    def _slice(start_marker, end_marker):
        i = src.find(start_marker)
        if i < 0:
            raise RuntimeError(f"step23 里找不到 {start_marker}")
        j = src.find(end_marker, i)
        if j < 0:
            j = len(src)
        return src[i:j]

    blocks = [
        _slice("def bh_adjust(", "def cramers_v("),
        _slice("def brant_test(", "def _fmt_vals("),
        _slice("def _fmt_vals(", "def collapse_sparse("),
        _slice("def collapse_sparse(", "def mean_ci("),
    ]
    ns = {"np": np, "pd": pd, "sm": sm, "stats": stats}
    exec(compile("\n".join(blocks), "from_step23", "exec"), ns)
    for fn in ("bh_adjust", "brant_test", "collapse_sparse"):
        if fn not in ns:
            raise RuntimeError(f"未能从 step23 提取 {fn}")
    return ns["bh_adjust"], ns["brant_test"], ns["collapse_sparse"]

bh_adjust, brant_test, collapse_sparse = _grab_from_step23()

def design(d, zcols, q6=True, q8=True):
    """设计矩阵：z 自变量在前，协变量在后。

    协变量按卡点 7：性别(参照 male) + 年级(参照 y1) + 专业(参照 eng)
                    + channel(参照 paper) + batch + Q6_选中数 + Q8_选中数
    ⚠ 只用 channel 与 batch 两列：net ⊂ batch0，两者不共线；
      但 net×batch1 是结构零，故禁止拟合 batch×channel。
    """
    cols = list(zcols) if isinstance(zcols, (list, tuple)) else [zcols]
    X = pd.DataFrame(index=d.index)
    for c in cols:
        X[c] = d[c].values.astype(float)
    X["female"] = (d["sex"].values == "female").astype(float)
    for g in GRADE_ORDER[1:]:
        X[f"grade_{g}"] = (d["grade"].values == g).astype(float)
    for m in MAJ_ORDER[1:]:
        X[f"major_{m}"] = (d["major"].values == m).astype(float)
    X["net"] = (d["channel"].values == "net").astype(float)
    X["batch2"] = (d["batch"].values == 1).astype(float)
    if q6:
        X["Q6_选中数"] = d["Q6_选中数"].values.astype(float)
    if q8:
        X["Q8_选中数"] = d["Q8_选中数"].values.astype(float)
    return X

def vif(X):
    """逐列 VIF = 1/(1−R²_j)，R² 来自该列对其余列的 OLS。"""
    out = {}
    X = X.astype(float)
    for c in X.columns:
        yv = X[c].values
        others = X.drop(columns=[c])
        if others.shape[1] == 0 or np.std(yv) == 0:
            out[c] = np.nan
            continue
        A = sm.add_constant(others.values.astype(float))
        r2 = sm.OLS(yv, A).fit().rsquared
        out[c] = np.inf if r2 >= 1 - 1e-12 else 1.0 / (1.0 - r2)
    return out

def zhang_yu(orr, p0):
    """Zhang & Yu (1998)：OR → RR。RR = OR / (1 − p0 + p0·OR)
    p0 = 未暴露组的结局发生率。结局不罕见时必须先转，否则 E-value 被高估。"""
    if orr is None or not np.isfinite(orr) or orr <= 0 or not np.isfinite(p0):
        return np.nan
    denom = 1.0 - p0 + p0 * orr
    return np.nan if denom <= 0 else orr / denom

def e_value(rr):
    """VanderWeele & Ding (2017)：E-value = RR + sqrt(RR·(RR−1))，适用于 RR>1。
    RR<1 时先取倒数（调用方负责把 CI 端也取倒数）。"""
    if rr is None or not np.isfinite(rr) or rr <= 0:
        return np.nan
    if rr < 1:
        rr = 1.0 / rr
    if rr <= 1.0:
        return 1.0
    return rr + np.sqrt(rr * (rr - 1.0))

def e_value_pair(est, lo, hi):
    """返回 (点估计 E-value, 置信限侧 E-value)。置信限取靠近 1 的那一端。"""
    if est >= 1:
        return e_value(est), e_value(lo)
    return e_value(1.0 / est), e_value(1.0 / hi)

def ord_fit(y_codes, X):
    """OrderedModel 拟合。成功返回 res，失败返回 None。

    ⚠⚠ 必须传 **DataFrame**，不能传 ndarray：
      1. statsmodels 的 OrderedModel 参数排列是 **[自变量……, 阈值……]** ——
         自变量在前，阈值（名为 "0/1"、"1/2"……）在后。
         **不是**很多人以为的「阈值在前」。
      2. 传 ndarray 就只能按位置取系数；若按「阈值在前」数数，取到的是阈值区 ——
         **不报错、方向反、CI 宽十倍**，典型静默错误（本脚本初版就踩了，
         表现为 Step 13 的 OR 与 Step 12 的 OLS 方向相反且 0/9 显著）。
      3. 传 DataFrame 后 `res.model.exog_names` 带列名，按名字定位可从根本上免疫。
    """
    for mth, kw in (("bfgs", {}), ("nm", {"maxiter": 4000})):
        try:
            res = OrderedModel(np.asarray(y_codes), X, distr="logit").fit(
                method=mth, disp=0, **kw)
        except Exception:
            continue
        b = np.asarray(res.params)
        if np.all(np.isfinite(b)):
            return res
    return None

def ord_pos(res, name):
    """在参数向量里定位自变量。用 exog_names 查名，不靠数数。"""
    names = list(res.model.exog_names)
    if name not in names:
        raise KeyError(f"{name} 不在 exog_names 里：{names}")
    return names.index(name)

def ord_coef(res, name):
    i = ord_pos(res, name)
    b = float(np.asarray(res.params)[i])
    se = float(np.sqrt(np.asarray(res.cov_params())[i, i]))
    p = float(np.asarray(res.pvalues)[i])
    return b, se, p

def wald_diff(res, ni, nj):
    """Wald 检验 H0: β_i − β_j = 0。Var = V_ii + V_jj − 2·V_ij。
    这是 Step 14 的核心检验 —— 不是分别报两个系数再目测。"""
    i, j = ord_pos(res, ni), ord_pos(res, nj)
    V = np.asarray(res.cov_params())
    b = np.asarray(res.params)
    d = float(b[i] - b[j])
    var = V[i, i] + V[j, j] - 2.0 * V[i, j]
    if not np.isfinite(var) or var <= 0:
        return np.nan, np.nan, np.nan
    z = d / np.sqrt(var)
    return d, z, float(2.0 * stats.norm.sf(abs(z)))

def to_codes(y, kind):
    """结局 → 有序编码 0..J−1。ord5 直接用；cont（13 值）按五分位分箱。
    返回 (codes, 说明, 原始值数组)。"""
    y = np.asarray(y, dtype=float)
    if kind == "ord5":
        # 与 step23 一致的处置：样本过少的类别并入相邻类（保序）。
        # 不做这一步，Brant 检验会因某切点两侧 n<15 直接返回 None（如外资企业「非常讨厌」仅 7 人）。
        codes, note, groups = collapse_sparse(y, min_n=40)
        return codes, ("类别合并：" + "、".join(note) if note else "5 级直接入模"), y
    q = pd.qcut(pd.Series(y), 5, duplicates="drop")
    codes = pd.Series(q).cat.codes.values.astype(int)
    n_bin = len(np.unique(codes))
    return codes, f"按 1579 集内五分位分箱为 {n_bin} 档（准连续，单一刻度需要）", y

def f2(x, nd=3):
    return "—" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{nd}f}"

def stars(p):
    if p is None or not np.isfinite(p):
        return ""
    return "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else ""

# ======================================================================
# 2. 数据
# ======================================================================
log("载入 1579 数据集 …")
d = pd.read_excel(F1579)
log(f"  形状 {d.shape[0]} × {d.shape[1]}")

# 口径自检（这几条必须打印出来，错了整批结果都不可信）
assert d["batch"].value_counts().to_dict() == {0: 1026, 1: 553}, "batch 构成不对"
assert np.isclose(d[Z_STRESS].std(ddof=1), 1.0), "压力 z 未按 1579 集内标准化"
assert np.isclose(d[ZF1].std(ddof=1), 1.0) and np.isclose(d[ZF2].std(ddof=1), 1.0), \
    "F1/F2 未按 1579 集内标准化"
_ct = pd.crosstab(d["channel"], d["batch"])
log(f"  channel × batch 交叉表（net×batch1 必须为 0，故不可拟合交互）:\n{_ct}")
assert _ct.loc["net", 1] == 0, "net×batch1 应为结构零"

# Q11（仅披露段用）
q11 = d["Q11_指数A_剔除不确定"].replace(99, np.nan)
log(f"  Q11 可用 n = {int(q11.notna().sum())}（已 replace(99, nan)）")

# ======================================================================
# 3. Step 12 · OLS 基线
# ======================================================================
log("[Step 12] OLS 基线：九个结局 × 每 1 SD 压力 …")
rows12 = []
for col, cn, en, kind in OUTCOMES:
    y = d[col].values.astype(float)
    X = design(d, Z_STRESS)
    r = sm.OLS(y, sm.add_constant(X)).fit(cov_type="HC3")
    b = float(r.params[Z_STRESS]); se = float(r.bse[Z_STRESS])
    p = float(r.pvalues[Z_STRESS])
    rows12.append(dict(结局=cn, 英文=en, 类型=kind, n=int(r.nobs),
                       beta=b, se=se, lo=b - Z975 * se, hi=b + Z975 * se, p=p,
                       R2=float(r.rsquared)))
t12 = pd.DataFrame(rows12)
t12["p_BH"] = bh_adjust(t12["p"].values)
t12 = t12.sort_values("p")
log(f"  BH 后显著 {int((t12['p_BH'] < .05).sum())}/9")

# ======================================================================
# 4. Step 13 · 有序 logit 主模型 + Brant + E-value
# ======================================================================
log("[Step 13] 有序 logit 主模型 …")
rows13, rows_brant, rows_ev = [], [], []
codes_cache = {}
for col, cn, en, kind in OUTCOMES:
    codes, note, yraw = to_codes(d[col].values.astype(float), kind)
    codes_cache[col] = (codes, note, yraw)
    X = design(d, Z_STRESS)
    res = ord_fit(codes, X)
    if res is None:
        log(f"  ⚠ {cn} 拟合失败")
        continue
    b, se, p = ord_coef(res, Z_STRESS)
    orr, lo, hi = np.exp(b), np.exp(b - Z975 * se), np.exp(b + Z975 * se)
    J = len(np.unique(codes))

    # 方向核对：不靠记忆，用预测概率实算
    Xp = X.copy()
    Xp[Z_STRESS] = float(X[Z_STRESS].max())
    p_hi = np.asarray(res.predict(exog=Xp)).mean(axis=0)
    Xp[Z_STRESS] = float(X[Z_STRESS].min())
    p_lo = np.asarray(res.predict(exog=Xp)).mean(axis=0)
    dir_ok = bool(p_hi[-1] > p_lo[-1])

    # Brant
    bt = brant_test(codes, X.values.astype(float), list(X.columns))
    bt_all = bt["overall"] if bt else None
    bt_z = bt["per"].get(Z_STRESS) if bt and Z_STRESS in bt["per"] else None

    # ---- E-value：两条轨 ----
    # 轨 A（近似，仅作对照）：直接把有序 OR 当 RR 用。结局不罕见时会高估 E-value。
    ev_a, ev_a_ci = e_value_pair(orr, lo, hi)
    # 轨 B（主报）：结局二分（≥4 视为「喜欢」）拟二项 logit，实测 p0 后 Zhang–Yu 转 RR。
    ybin = (yraw >= 4).astype(int)
    ev_b = ev_b_ci = p0 = orr_bin = rr_b = np.nan
    pos_rate = float(np.mean(ybin))
    if 5 <= ybin.sum() <= len(ybin) - 5:
        try:
            rb = sm.Logit(ybin, sm.add_constant(X)).fit(disp=0, maxiter=300)
            bbin = float(rb.params[Z_STRESS]); sbin = float(rb.bse[Z_STRESS])
            orr_bin = np.exp(bbin)
            lo_b, hi_b = np.exp(bbin - Z975 * sbin), np.exp(bbin + Z975 * sbin)
            X0 = sm.add_constant(X).copy()
            X0[Z_STRESS] = 0.0
            p0 = float(np.mean(rb.predict(X0)))
            rr_b = zhang_yu(orr_bin, p0)
            ev_b, ev_b_ci = e_value_pair(rr_b, zhang_yu(lo_b, p0), zhang_yu(hi_b, p0))
        except Exception:
            pass

    rows13.append(dict(结局=cn, 英文=en, 类型=kind, 类别数=J, 编码说明=note,
                       n=int(len(codes)), OR=orr, CI_lo=lo, CI_hi=hi, p=p,
                       Brant_W=(bt_all[0] if bt_all else np.nan),
                       Brant_df=(bt_all[1] if bt_all else np.nan),
                       Brant_p=(bt_all[2] if bt_all else np.nan),
                       Brant_p_z=(bt_z[2] if bt_z else np.nan),
                       方向核对=("OR>1 ⇔ 向更高等级" if dir_ok else "⚠ 方向未通过核对"),
                       阳性率=pos_rate))
    rows_ev.append(dict(结局=cn, 英文=en,
                        OR_有序=orr, E轨道A_直接=ev_a, E轨道A_CI限=ev_a_ci,
                        OR_二分=orr_bin, p0=p0, RR_转后=rr_b,
                        E轨道B_主报=ev_b, E轨道B_CI限=ev_b_ci))
    if bt:
        for nm, v in bt["per"].items():
            rows_brant.append(dict(结局=cn, 变量=nm, W=v[0], df=v[1], p=v[2]))

t13 = pd.DataFrame(rows13)
t13["p_BH"] = bh_adjust(t13["p"].values)
t13 = t13.sort_values("p")
t_ev = pd.DataFrame(rows_ev)
t_brant = pd.DataFrame(rows_brant) if rows_brant else pd.DataFrame()
if len(t_brant):
    t_brant["p_BH"] = bh_adjust(t_brant["p"].values)
log(f"  BH 后显著 {int((t13['p_BH'] < .05).sum())}/9；"
    f"Brant 违反 {int((t13['Brant_p'] < .05).sum())}/9")

# ---- Q11 的强制披露（Step 13 必做，不做会被视为隐瞒）----
log("[Step 13] Q11 批次 × 压力交互披露 …")
rows_int = []
for col, cn, en, kind in OUTCOMES + [("__Q11__", "期望受影响指数", "Expectation impact", "cont")]:
    if col == "__Q11__":
        yv = q11.values.astype(float)
        ok = ~np.isnan(yv)
        sub = d.loc[ok].copy()
        yv = yv[ok]
    else:
        sub = d
        yv = d[col].values.astype(float)
    X = design(sub, Z_STRESS)
    X["batch2×压力"] = X["batch2"].values * X[Z_STRESS].values
    r = sm.OLS(yv, sm.add_constant(X)).fit(cov_type="HC3")
    b = float(r.params["batch2×压力"]); se = float(r.bse["batch2×压力"])
    rows_int.append(dict(结局=cn, 英文=en, n=int(r.nobs), theta=b, se=se,
                         lo=b - Z975 * se, hi=b + Z975 * se,
                         p=float(r.pvalues["batch2×压力"])))
t_int = pd.DataFrame(rows_int)
t_int["p_BH"] = bh_adjust(t_int["p"].values)

# ======================================================================
# 4b. Step 13b · ipsative 中心化：判别「整体抬升」还是「真实分化」
# ======================================================================
# 为什么必须做：Step 13 九个结局的 OR **全部 >1**。第一反应不该是「压力让人更想工作」，
# 而应是怀疑量表使用差异（acquiescence / scale usage）——压力大的人对所有选项都打高分。
# 判据：对每个人把 Q10 八题减去他自己的平均分，剩下的才是「相对更偏好哪一类」。
# ⚠ 数学性质：中心化后各结局系数之和恒为 0，所以必然有正有负 ——
#    看的是**分化模式**，不是单个结局是否显著。
log("[Step 13b] ipsative 中心化（个体内去均值）敏感性 …")
Q10_ITEMS = [f"Q10_{i}" for i in range(1, 9)]
_ips_mu = d[Q10_ITEMS].mean(axis=1)
rows_ips = []
for col, cn, en, kind in OUTCOMES:
    if col in Q10_ITEMS:
        yv = d[col].values.astype(float) - _ips_mu.values
    else:
        yv = d[["Q10_5", "Q10_6", "Q10_7"]].sub(_ips_mu, axis=0).mean(axis=1).values
    X_ips = design(d, Z_STRESS)
    r_ips = sm.OLS(yv, sm.add_constant(X_ips)).fit(cov_type="HC3")
    bi = float(r_ips.params[Z_STRESS]); sei = float(r_ips.bse[Z_STRESS])
    braw = t12.loc[t12["结局"] == cn, "beta"]
    rows_ips.append(dict(结局=cn, 英文=en,
                         beta_中心化=bi, se=sei,
                         lo=bi - Z975 * sei, hi=bi + Z975 * sei,
                         p=float(r_ips.pvalues[Z_STRESS]),
                         beta_原始=float(braw.iloc[0]) if len(braw) else np.nan))
t_ips = pd.DataFrame(rows_ips)
t_ips["p_BH"] = bh_adjust(t_ips["p"].values)
t_ips = t_ips.sort_values("beta_中心化", ascending=False)
_npos = int((t_ips["beta_中心化"] > 0).sum())
_ips8 = t_ips[t_ips["结局"].isin([o[1] for o in OUTCOMES if o[0] in Q10_ITEMS])]
log(f"  中心化后正值 {_npos}/{len(t_ips)} 个；八个单题系数之和 = "
    f"{_ips8['beta_中心化'].sum():+.6f}（数学上应恒为 0）")
_q11 = t_int[t_int["结局"] == "期望受影响指数"].iloc[0]
log(f"  Q11 交互 θ = {_q11['theta']:.3f} [{_q11['lo']:.3f}, {_q11['hi']:.3f}], "
    f"p_BH = {_q11['p_BH']:.3f}（手册记载 −0.060 / [−0.114, −0.007] / .138）")

# ======================================================================
# 5. Step 14 · 维度拆解（F1 / F2 同时入模 + 系数差异 Wald）
# ======================================================================
log("[Step 14] 维度拆解 …")
Xf = design(d, [ZF1, ZF2])
_v = vif(Xf)
log(f"  VIF: F1 = {_v[ZF1]:.3f}, F2 = {_v[ZF2]:.3f}"
    f"（手册预判偏高；实测得分相关 r = {d[ZF1].corr(d[ZF2]):.3f}）")
VIF_MAX = float(np.nanmax([_v[ZF1], _v[ZF2]]))
USE_BOTH = VIF_MAX <= 5.0

rows14 = []
for col, cn, en, kind in OUTCOMES:
    codes, note, yraw = codes_cache[col]
    res = ord_fit(codes, Xf)
    if res is None:
        continue
    b1, se1, p1 = ord_coef(res, ZF1)
    b2, se2, p2 = ord_coef(res, ZF2)
    dd, zz, pp = wald_diff(res, ZF1, ZF2)
    # 单因子模型（VIF 超标时的预案，同时入档备用）
    r1 = ord_fit(codes, design(d, ZF1))
    r2 = ord_fit(codes, design(d, ZF2))
    e1 = np.exp(ord_coef(r1, ZF1)[0]) if r1 is not None else np.nan
    e2 = np.exp(ord_coef(r2, ZF2)[0]) if r2 is not None else np.nan
    rows14.append(dict(结局=cn, 英文=en, n=int(len(codes)),
                       OR_F1=np.exp(b1), F1_lo=np.exp(b1 - Z975 * se1),
                       F1_hi=np.exp(b1 + Z975 * se1), p_F1=p1,
                       OR_F2=np.exp(b2), F2_lo=np.exp(b2 - Z975 * se2),
                       F2_hi=np.exp(b2 + Z975 * se2), p_F2=p2,
                       差_logit=dd, 差_OR=np.exp(dd), Wald_z=zz, p_差异=pp,
                       OR_F1_单入=e1, OR_F2_单入=e2))
t14 = pd.DataFrame(rows14)
t14["p_F1_BH"] = bh_adjust(t14["p_F1"].values)
t14["p_F2_BH"] = bh_adjust(t14["p_F2"].values)
t14["p_差异_BH"] = bh_adjust(t14["p_差异"].values)
t14 = t14.sort_values("p_差异")
log(f"  差异检验 BH 后显著 {int((t14['p_差异_BH'] < .05).sum())}/9")

# 总指数 vs 因子的并列对照
t14b = t13[["结局", "OR", "CI_lo", "CI_hi", "p_BH"]].rename(
    columns={"OR": "OR_总指数", "CI_lo": "总指数_lo", "CI_hi": "总指数_hi",
             "p_BH": "p_总指数_BH"}).merge(
    t14[["结局", "OR_F1", "OR_F2", "p_差异_BH"]], on="结局", how="outer")

t_vif = pd.DataFrame({"变量": list(_v.keys()), "VIF": list(_v.values())}).sort_values(
    "VIF", ascending=False)

# ======================================================================
# 6. 图
# ======================================================================
log("[图 9] 每 1 SD 压力的调整 OR 森林图 …")
g = t13.sort_values("OR")
fig, ax = plt.subplots(figsize=(6.85, 5.2))
yy = np.arange(len(g))
for i, (_, r) in enumerate(g.iterrows()):
    ax.plot([r["CI_lo"], r["CI_hi"]], [i, i], color="#185FA5", lw=1.6, zorder=2)
    ax.scatter([r["OR"]], [i], s=42, color="#185FA5", zorder=3)
    tag = "BH n.s." if r["p_BH"] >= .05 else (
        "BH p<.001" if r["p_BH"] < .001 else f"BH p={r['p_BH']:.3f}")
    ax.text(r["CI_hi"] * 1.02, i, f"{r['OR']:.3f}  {tag}", va="center",
            fontsize=8.5, color="#444441")
# ⚠ 图内文字一律英文（Springer 规范），且字体是 Arial —— 中文会渲染成缺字形方框
ax.set_yticks(yy)
ax.set_yticklabels(list(g["英文"]), fontsize=9.5)
ax.axvline(1.0, color="#888780", lw=1, ls="--", zorder=1)
ax.set_xscale("log")
_tk = [0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7]
_tk = [t for t in _tk if float(g["CI_lo"].min()) * 0.9 <= t <= float(g["CI_hi"].max()) * 1.35]
ax.set_xticks(_tk)
ax.set_xticklabels([f"{t:.1f}" for t in _tk])
ax.minorticks_off()
ax.set_xlim(float(g["CI_lo"].min()) * 0.88, float(g["CI_hi"].max()) * 1.42)
ax.set_xlabel("Adjusted OR per 1 SD of stressor index (95% CI, log scale)",
              fontsize=9.5)
ax.tick_params(axis="x", labelsize=9)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig_RQ3_forest.png"), dpi=DPI, facecolor="white")
plt.close(fig)

log("[图 11] 因子 × 偏好 OR 热图 + 差异检验标注 …")
g2 = t14.sort_values("OR_F1", ascending=False)
M = np.column_stack([g2["OR_F1"].values, g2["OR_F2"].values])
fig, axes = plt.subplots(1, 2, figsize=(6.85, 5.0),
                         gridspec_kw={"width_ratios": [1.0, 1.35]})
ax = axes[0]
_hr = float(np.abs(M - 1).max()) * 1.03         # 色阶对称于 1，且随实测范围自适应
im = ax.imshow(M, cmap="RdBu_r", vmin=1 - _hr, vmax=1 + _hr, aspect="auto")
# x 轴只标 F1 / F2 —— 两列太窄，全名会互相压字；全名交给图注（期刊图注本就要写）
ax.set_xticks([0, 1])
ax.set_xticklabels(["F1", "F2"], fontsize=10)
ax.set_yticks(np.arange(len(g2)))
ax.set_yticklabels(list(g2["英文"]), fontsize=8.5)
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        ax.text(j, i, f"{M[i, j]:.3f}", ha="center", va="center", fontsize=8.5,
                color="white" if abs(M[i, j] - 1) > _hr * 0.58 else "#2C2C2A")
# 图内不放标题（Springer 禁止），面板说明改由 colorbar 标签与 (a)(b) 承担
cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.05)
cb.set_label("OR per 1 SD", fontsize=8)
cb.ax.tick_params(labelsize=7.5)
ax.set_title("(a)", fontsize=10, pad=6, loc="left")

ax = axes[1]
yy = np.arange(len(g2))
# ⚠ 必须倒置 y 轴：左侧 imshow 默认 origin="upper"（第 0 行画在**顶部**），
#   而 barh 的 y=0 在**底部**。少了这一行，(b) 的九行会与 (a) 完全镜像错位
#   —— 表现为「Grassroots 明明 F1>F2 却画在负半轴」。初版就踩了这个坑。
ax.invert_yaxis()
dlog = g2["差_logit"].values
# 配色与 (a) 一致：正值红、负值蓝（RdBu_r 的那一端）
ax.barh(yy, dlog, color=["#A32D2D" if v > 0 else "#185FA5" for v in dlog], height=0.62)
_pad = 0.02 * float(np.abs(dlog).max())
for i, (v, pb) in enumerate(zip(dlog, g2["p_差异_BH"])):
    tag = "**" if pb < .01 else "*" if pb < .05 else "n.s."
    ax.text(v + (_pad if v >= 0 else -_pad), i, f"{v:+.3f} {tag}",
            va="center", ha="left" if v >= 0 else "right", fontsize=8,
            color="#444441")
# (b) 的结局顺序与 (a) 完全相同，不再重复 y 轴标签（省出的空间留给数值标注）
ax.set_yticks(yy); ax.set_yticklabels([])
ax.axvline(0, color="#5F5E5A", lw=1)
ax.set_xlim(float(dlog.min()) * 2.9, float(dlog.max()) * 2.9)
ax.set_xlabel("log-OR difference  ($\\beta_{F1}-\\beta_{F2}$)", fontsize=9.5)
ax.set_title("(b)", fontsize=10, pad=6, loc="left")
ax.spines[["top", "right"]].set_visible(False)
ax.tick_params(axis="x", labelsize=8.5)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig_RQ3_factor.png"), dpi=DPI, facecolor="white")
plt.close(fig)

# ======================================================================
# 7. 报告
# ======================================================================
log("写出报告 …")

for _, r in t12.iterrows():
    pass
_n12 = int((t12["p_BH"] < .05).sum())
_sig12 = t12[t12["p_BH"] < .05]
if _n12 == 0:
    _dir12 = "BH 校正后无一显著。"
elif (_sig12["beta"] > 0).all():
    _dir12 = ("显著项方向**一致为正** —— 压力越大，各项就业选择的评分越高。"
              "李克特语义是「喜欢程度」，因此正号应解读为「更倾向于」，不是「压力更大」。")
elif (_sig12["beta"] < 0).all():
    _dir12 = "显著项方向**一致为负** —— 压力越大，各项就业选择的评分越低。"
else:
    _dir12 = "⚠ **显著项方向不一致**，须逐结局解释，不能合成一句话。"

for _, r in t13.iterrows():
    pass

_viol = t13[t13["Brant_p"] < .05]
if len(_viol):
    _sub = t_brant[t_brant["结局"].isin(_viol["结局"])].sort_values("p")
    for _, r in _sub.head(12).iterrows():
        pass
    _wild = t_brant[t_brant["W"] > 200]
    if len(_wild):
        for _, r in _wild.iterrows():
            vn = str(r["变量"])
            if vn.startswith("grade_"):
                nn = int((d["grade"].values == vn[6:]).sum())
            elif vn.startswith("major_"):
                nn = int((d["major"].values == vn[6:]).sum())
            elif vn == "female":
                nn = int((d["sex"].values == "female").sum())
            elif vn == "net":
                nn = int((d["channel"].values == "net").sum())
            elif vn == "batch2":
                nn = int((d["batch"].values == 1).sum())
            else:
                nn = int(d[vn].sum()) if vn in d.columns else -1
    _zbad = t13[t13["Brant_p_z"] < .05]
else:
    pass

for _, r in t_ev.iterrows():
    pr = t13[t13["结局"] == r["结局"]]["阳性率"]
_evmax = float(t_ev["E轨道B_主报"].max())
_evmin = float(t_ev["E轨道B_主报"].min())

for _, r in t_int.sort_values("p").iterrows():
    pass

for _, r in t_ips.iterrows():
    pass
_nsig_ips = int((t_ips["p_BH"] < .05).sum())
_sig_names = "、".join(t_ips[t_ips["p_BH"] < .05]["结局"]) if _nsig_ips else "无"
_ipos = t_ips[t_ips["beta_中心化"] > 0]
_ineg = t_ips[t_ips["beta_中心化"] < 0].sort_values("beta_中心化")
if len(_ipos) and len(_ineg):
    pass

for _, r in t_vif.head(6).iterrows():
    pass
for _, r in t14.iterrows():
    pass
_nd = int((t14["p_差异_BH"] < .05).sum())
for _, r in t14[t14["p_差异_BH"] < .05].iterrows():
    _strong = "F1" if r["差_logit"] > 0 else "F2"
    _weak = "F2" if r["差_logit"] > 0 else "F1"
for _, r in t14b.iterrows():
    pass

with pd.ExcelWriter(XLSX, engine="openpyxl") as w:
    t12.to_excel(w, sheet_name="表12_OLS基线", index=False)
    t13.to_excel(w, sheet_name="表13_有序logit主模型", index=False)
    t_brant.to_excel(w, sheet_name="表13_Brant逐变量", index=False)
    t_ev.to_excel(w, sheet_name="表13_Evalue", index=False)
    t_int.to_excel(w, sheet_name="表13_批次交互披露", index=False)
    t_ips.to_excel(w, sheet_name="表13_ipsative中心化", index=False)
    t14.to_excel(w, sheet_name="表15_因子双入模", index=False)
    t14b.to_excel(w, sheet_name="表15_总指数vs因子", index=False)
    t_vif.to_excel(w, sheet_name="表15_VIF", index=False)

log(f"✅ 完成：{XLSX}")
log(f"✅ 完成：{FIG}/fig_RQ3_forest.png, fig_RQ3_factor.png")
