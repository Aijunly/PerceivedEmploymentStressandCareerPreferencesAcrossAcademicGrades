# -*- coding: utf-8 -*-
r"""
Step 15–17 · 研究问题三（后半）：意向分类、探索性中介、调节效应
======================================================================
输入：论文二数据集_1579.xlsx（Step 1 产出；含 Step 6 的 F1 / F2）
输出：论文二_输出/Step15_17_意向与机制结果.xlsx
      figs2/fig_RQ3_ame.png          图 10  意向4类 AME 点图
      figs2/fig_RQ3_path.png         图 12  探索性中介路径图 + 间接关联
      figs2/fig_RQ3_moderation.png   图 13  分层 OR（显著调节项）

⚠ 卡点 7 口径（硬性，全文执行）
  · 样本 1579；协变量含 batch；只用 channel + batch，禁止三分类「来源」、禁止 batch×channel
  · 连续自变量按 1579 集内 SD 标准化（数据集里的 z_* 与 F* 均已按 1579 内算好，直接取用）
  · Q11 先 replace(99, nan)

⚠ 措辞红线：associated with / 不得写「导致」「effect」「changes over time」「developmental」
  Step 16 必须是 **associational mediation**（横断面不能确立因果路径）
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
import matplotlib.patches as mpatches
import statsmodels.api as sm
from scipy import stats
from statsmodels.miscmodels.ordinal_model import OrderedModel
from statsmodels.stats.mediation import Mediation

warnings.filterwarnings("ignore")

# ======================================================================
# 0. 全局设定
# ======================================================================
BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "论文二_输出")
FIG = os.path.join(BASE, "figs2")
for _d in (OUT, FIG):
    os.makedirs(_d, exist_ok=True)

F1579 = os.path.join(BASE, "论文二数据集_1579.xlsx")
XLSX = os.path.join(OUT, "Step15_17_意向与机制结果.xlsx")

plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["pdf.fonttype"] = 42
DPI = 600
SEED = 42
Z975 = 1.959963984540054
# 正式跑是 5000 / 2000；调试时用环境变量调小，省下每次 ~5 分钟的等待
N_REP = int(os.environ.get("STEP25_NREP", 5000))         # Step 16 自举次数（手册指定 5000）
N_REP_CHK = int(os.environ.get("STEP25_NREP_CHK", 2000))  # 与 statsmodels.Mediation 对照

# 九个结局（与 step24 完全一致）
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
ZF1, ZF2 = "F1", "F2"
M_COL = "Q11M"                    # 中介变量：期望受影响指数（99 已置 nan）

def log(msg=""):
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)

# ======================================================================
# 1. 复用已校准的实现（不复制，防漂移）
# ======================================================================
def _slice(src, start_marker, end_marker, default_end=None):
    i = src.find(start_marker)
    if i < 0:
        raise RuntimeError(f"找不到 {start_marker}")
    j = src.find(end_marker, i) if end_marker else -1
    if j < 0:
        j = len(src) if default_end is None else src.find(default_end, i)
    if j < 0:
        j = len(src)
    return src[i:j]

with open(os.path.join(BASE, "step23_grade_trend.py"), "r", encoding="utf-8") as f:
    _s23 = f.read()
with open(os.path.join(BASE, "step24_stress_choice.py"), "r", encoding="utf-8") as f:
    _s24 = f.read()

_ns = {"np": np, "pd": pd, "sm": sm, "stats": stats,
       "GRADE_ORDER": GRADE_ORDER, "MAJ_ORDER": MAJ_ORDER}
_blocks = [
    _slice(_s23, "def bh_adjust(", "def cramers_v("),
    _slice(_s23, "def _fmt_vals(", "def collapse_sparse("),      # collapse_sparse 依赖它
    _slice(_s23, "def collapse_sparse(", "def mean_ci("),
    _slice(_s24, "def design(", "# ======================================================================\n# 2. 数据"),
]
exec(compile("\n".join(_blocks), "from_prev_steps", "exec"), _ns)
bh_adjust = _ns["bh_adjust"]
collapse_sparse = _ns["collapse_sparse"]
design = _ns["design"]
f2 = _ns["f2"] if "f2" in _ns else None

def f2(x, nd=3):
    return "—" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{nd}f}"

def stars(p):
    if p is None or not np.isfinite(p):
        return ""
    return "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else ""

# ======================================================================
# 2. 数据 + 口径自检
# ======================================================================
log("载入 1579 数据集 …")
d = pd.read_excel(F1579)
log(f"  形状 {d.shape[0]} × {d.shape[1]}")

assert d["batch"].value_counts().to_dict() == {0: 1026, 1: 553}, "batch 构成不对"
assert np.isclose(d[Z_STRESS].std(ddof=1), 1.0), "压力 z 未按 1579 集内标准化"
assert np.isclose(d[ZF1].std(ddof=1), 1.0) and np.isclose(d[ZF2].std(ddof=1), 1.0), \
    "F1/F2 未按 1579 集内标准化"
_ct = pd.crosstab(d["channel"], d["batch"])
assert _ct.loc["net", 1] == 0, "net×batch1 应为结构零"

d[M_COL] = d["Q11_指数A_剔除不确定"].replace(99, np.nan)
log(f"  Q11 中介变量可用 n = {int(d[M_COL].notna().sum())}（99 已置 nan）")
log(f"  corr(z_压力指数, Q11M) = {d[Z_STRESS].corr(d[M_COL]):.4f}")

# ======================================================================
# 3. Step 15 · 多项 logistic：意向 4 类（表 14、图 10）
# ======================================================================
# ⚠⚠ MNLogit **没有 base= 参数**（传了会被 kwargs 静默吞掉），参照组固定是编码后
#    y=0 的那一类。批 3 已经在这里栽过一次（CATS 写反导致整张 RRR 相对「深造」）。
#    这里把「直接就业」显式放在 0，并在拟合后做一次数值复核。
INTENT_MAP = {"直接就业": 0, "深造": 1, "求稳体制": 2, "方向未定": 3}
CATS_DISP = ["深造", "求稳体制", "方向未定"]      # 展示顺序（不含参照组）
CATS_EN = {"深造": "Further study", "求稳体制": "Public-sector stability",
           "方向未定": "Undecided", "直接就业": "Direct employment (ref.)"}
_int_raw = dict(d["意向4类"].value_counts())
log(f"[Step 15] 意向4类分布：{_int_raw}")

def fit_mnlogit(sub, zcols):
    """MNLogit 拟合。注意：多项 logit **需要常数项**（与 OrderedModel 相反）。"""
    X = sm.add_constant(design(sub, zcols))
    y = sub["意向4类"].map(INTENT_MAP).astype(int).values
    return sm.MNLogit(y, X).fit(disp=0, maxiter=200), X, y

def _check_reference(res, X):
    """数值复核参照组是不是 y=0（直接就业）。

    手算 softmax：log p_k/p_0 = X·β_k（k=0 时 β_0 ≡ 0）。若假设错，
    手算的 P 与 res.predict 必然对不上。这是唯一能抓住「参照组错位」的办法。
    """
    ex = list(res.model.exog_names)
    B = np.asarray(res.params)                      # (P, J-1)
    Xv = np.asarray(X, dtype=float)
    eta = np.column_stack([np.zeros(len(Xv)), Xv @ B])
    eta -= eta.max(axis=1, keepdims=True)
    P = np.exp(eta)
    P /= P.sum(axis=1, keepdims=True)
    return float(np.abs(P - np.asarray(res.predict(exog=Xv))).max())

def mn_coef_ci(res, name, k):
    """第 k 个非参照类（k=0 → 深造）的系数。params 维度是 (变量 × 类别)。"""
    i = list(res.model.exog_names).index(name)
    b = float(np.asarray(res.params)[i, k])
    se = float(np.asarray(res.bse)[i, k])
    return b, np.exp(b), np.exp(b - Z975 * se), np.exp(b + Z975 * se), \
        float(np.asarray(res.pvalues)[i, k])

def wald_mn(res, n1, n2, k):
    """同一结局下两个自变量的系数差异 Wald 检验（跨行Comparing across equations）。
    ⚠ 两个系数来自**同一模型的两个自变量行**，Var = V_11 + V_22 − 2·V_12
      （协方差矩阵在 MNLogit 里按 (变量, 类别) 展平，索引 = i*(J-1)+k）。"""
    V = np.asarray(res.cov_params())
    J1 = np.asarray(res.params).shape[1]
    i1 = list(res.model.exog_names).index(n1) * J1 + k
    i2 = list(res.model.exog_names).index(n2) * J1 + k
    b = np.asarray(res.params).reshape(-1)
    dd = float(b[i1] - b[i2])
    var = V[i1, i1] + V[i2, i2] - 2.0 * V[i1, i2]
    if not np.isfinite(var) or var <= 0:
        return np.nan, np.nan, np.nan
    z = dd / np.sqrt(var)
    return dd, z, float(2.0 * stats.norm.sf(abs(z)))

def ame_analytic(res, name, X, J):
    """AME 解析式：∂p_k/∂x = p_k·(β_k − Σ_j p_j β_j)，参照组 β_0 ≡ 0。

    ⚠ **极易写错的一行**：Σ_j p_j β_j 里的 β 是**所有类别**的系数向量，
      必须在循环外一次填好。若写成「每个 k 都新建 b 并只填 b[k]」，
      加权和会塌缩成 p_k·β_k → 参照组那一类必然算出 0（本脚本初版就犯了这个错，
      靠与 get_margeff 的对照才发现）。
    """
    i = list(res.model.exog_names).index(name)
    B = np.asarray(res.params)                       # (P, J-1)
    Xv = np.asarray(X, dtype=float)
    eta = np.column_stack([np.zeros(len(Xv)), Xv @ B])
    eta -= eta.max(axis=1, keepdims=True)
    P = np.exp(eta)
    P /= P.sum(axis=1, keepdims=True)
    b = np.zeros(J)
    for k in range(1, J):
        b[k] = B[i, k - 1]
    wsum = P @ b                                     # Σ_j p_ij β_j（逐观测）
    return np.array([float(np.mean(P[:, k] * (b[k] - wsum))) for k in range(J)])

log("[Step 15] 多项 logistic：意向4类（参照组 = 直接就业）…")
sub_i = d.dropna(subset=["意向4类"]).copy()
t15_rows, t15_ame_rows, refcheck = [], [], []

SPECS = [("总指数", [Z_STRESS]), ("双因子", [ZF1, ZF2])]
for specname, zc in SPECS:
    res, X, y = fit_mnlogit(sub_i, zc)
    rc = _check_reference(res, X)
    refcheck.append((specname, rc))
    log(f"  {specname}: 参照组复核 max|ΔP| = {rc:.2e}"
        f"{'  ✅' if rc < 1e-8 else '  ⚠ 参照组可能错位'}")
    for c in zc:
        for k, cat in enumerate(CATS_DISP):
            b, rrr, lo, hi, p = mn_coef_ci(res, c, k)
            t15_rows.append(dict(自变量组=specname, 自变量=c, 类别=cat, 英文=CATS_EN[cat],
                                 b=b, RRR=rrr, CI_lo=lo, CI_hi=hi, p=p))
    # ---- AME ----
    # get_margeff 的 summary_frame 行序是 [y=0 块][y=1 块]……，每块长度 = 自变量个数。
    # ⚠ 这个行序在 statsmodels 版本间可能变，所以用「按变量名匹配」而不是按位置，
    #   拿不到就跳过（宁缺勿错）。同时用解析式 cross-check 点估计。
    me = res.get_margeff(at="overall", method="dydx")
    _mf = me.summary_frame().reset_index()
    # ⚠ summary_frame 的第一列叫 "endog"，变量名在第二列 "exog"（已实测）。
    #   若将来版本改了列名，这里会一眼看出来而不是静默产出空表。
    if "exog" not in _mf.columns:
        raise RuntimeError(f"summary_frame 列名变了：{list(_mf.columns)}")
    # ⚠ 每块行数 = 自变量个数，但 get_margeff **默认剔除常数项**，所以不能用
    #   len(exog_names)（会多 1 → 块边界整体错位，最后一类必然取不到）。
    #   稳妥做法：用总行数 ÷ 类别数，并校验整除。
    if len(_mf) % 4:
        raise RuntimeError(f"summary_frame 行数 {len(_mf)} 不是类别数 4 的整数倍")
    _nm = len(_mf) // 4
    for c in zc:
        an = ame_analytic(res, c, X, 4)
        for k, cat in enumerate(["直接就业"] + CATS_DISP):
            blk = _mf.iloc[k * _nm:(k + 1) * _nm]
            rw = blk[blk["exog"].astype(str).str.strip() == c]
            if len(rw) == 0:
                raise RuntimeError(f"AME 表里找不到变量 {c}（结局块 {k}）")
            dy = float(rw["dy/dx"].iloc[0]); se = float(rw["Std. Err."].iloc[0])
            t15_ame_rows.append(dict(自变量组=specname, 自变量=c, 类别=cat,
                                     英文=CATS_EN[cat],
                                     AME=dy, se=se, lo=dy - Z975 * se, hi=dy + Z975 * se,
                                     p=float(rw["Pr(>|z|)"].iloc[0]),
                                     AME_解析=float(an[k])))

t15 = pd.DataFrame(t15_rows)
# BH 校正：按自变量组分别校正（组内测试数 = 组自变量数 × 3 类）
t15["p_BH"] = np.nan
for gname, gi in t15.groupby("自变量组"):
    t15.loc[gi.index, "p_BH"] = bh_adjust(gi["p"].values)

t15_ame = pd.DataFrame(t15_ame_rows)
_maxame = float(np.abs(t15_ame["AME"] - t15_ame["AME_解析"]).max()) if len(t15_ame) else np.nan
log(f"  AME 解析式 vs get_margeff 最大差 = {_maxame:.2e}"
    f"{'  ✅ 一致' if np.isfinite(_maxame) and _maxame < 1e-8 else '  ⚠ 需排查'}")

# ---- 维度差异检验（仅双因子模型，回答 Step 14 的同一问题）----
res_b, X_b, y_b = fit_mnlogit(sub_i, [ZF1, ZF2])
diff_rows = []
for k, cat in enumerate(CATS_DISP):
    dd, zz, pp = wald_mn(res_b, ZF1, ZF2, k)
    diff_rows.append(dict(类别=cat, 英文=CATS_EN[cat], 差_logit=dd, z=zz, p=pp))
t15_diff = pd.DataFrame(diff_rows)
t15_diff["p_BH"] = bh_adjust(t15_diff["p"].values)
log(f"  双因子差异检验 BH 后显著 {int((t15_diff['p_BH'] < .05).sum())}/3")

# ---- IIA 敏感性：剔除「方向未定」后重估（Δlog 判据同批 3）----
sub_ex = sub_i[sub_i["意向4类"] != "方向未定"].copy()
res_e, X_e, y_e = fit_mnlogit(sub_ex, [Z_STRESS])
res_main, X_main, _ = fit_mnlogit(sub_i, [Z_STRESS])
_rows_sens = []
for cat in ["深造", "求稳体制"]:
    kk = CATS_DISP.index(cat)
    b_m, rrr_m, lo_m, hi_m, p_m = mn_coef_ci(res_main, Z_STRESS, kk)
    se_m = (np.log(hi_m) - np.log(lo_m)) / (2 * Z975)
    b_e, rrr_e, lo_e, hi_e, p_e = mn_coef_ci(res_e, Z_STRESS, kk)
    se_e = (np.log(hi_e) - np.log(lo_e)) / (2 * Z975)
    # ⚠ 判定串不能用 str.contains("敏感") 去捞 —— "不敏感" 里也含 "敏感"（本脚本初版
    #   就漏了这条，把两条明确写着「不敏感」的行判成了敏感）。改用 ⚠ 前缀。
    dl = b_e - b_m
    _rows_sens.append(dict(类别=cat, RRR_四分类=rrr_m, SE_四分类=se_m,
                           RRR_剔除未定=rrr_e, SE_剔除未定=se_e,
                           Δ_log=dl, 半个SE=0.5 * se_m,
                           判定=("⚠ 敏感" if abs(dl) >= 0.5 * se_m else "不敏感")))
t15_sens = pd.DataFrame(_rows_sens)

# ======================================================================
# 4. Step 16 · 探索性中介（表 16、图 12）
# ======================================================================
COVS16 = ["female", "grade_y2", "grade_y3", "grade_y4", "grade_grad",
          "major_sci", "major_econ", "major_hum", "major_oth",
          "net", "batch2", "Q6_选中数", "Q8_选中数"]

# 预构造协变量列（保持与卡点 7 完全一致的哑变量定义）
_dm = design(d, Z_STRESS)
for c in COVS16:
    d[c] = _dm[c].values

def _wls_beta(X, y, c):
    """加权最小二乘：行权重 c（自举时用重数向量，等价于有放回抽样的解）。"""
    XtWX = X.T @ (c[:, None] * X)
    XtWy = X.T @ (c * y)
    return np.linalg.solve(XtWX, XtWy)

def boot_mediation(df, ycol, mcol, xcol, covs, n_rep=N_REP, seed=SEED):
    """线性模型下的路径分析自举：indirect = a·b, direct = c′, total = c。

    为什么自己写而不是直接用 stats 的 Mediation：
      ① OrderedModel 作结局模型时 Mediation 会重建公式并报
         "There should not be a constant in the model"（patsy 加截距，
          OrderedModel 拒绝），有序口径走不通；
      ② 九个结局 × 5000 次用 Mediation 逐个跑要约 40 分钟；这里按行的**重数向量**
         把每次自举化成一次加权最小二乘（X'WX 与 X'Wy），总耗时降到 1 分钟内；
      ③ 口径更透明 —— 本法与 Mediation 的一致性会做数值对照（见下文）。
    ⚠ 有放回抽样 = 给每行一个整数权重，X'WX = Σ c_i x_i x_iᵀ —— 这是等价变形，
      不是近似。
    """
    cols = [ycol, mcol, xcol] + covs
    s = df[cols].dropna()
    y = s[ycol].values.astype(float)
    m = s[mcol].values.astype(float)
    x = s[xcol].values.astype(float)
    C = np.column_stack([s[c].values.astype(float) for c in covs]) if covs \
        else np.zeros((len(s), 0))
    one = np.ones(len(s))
    Xm = np.column_stack([one, x, C])            # m ~ x + cov
    Xo = np.column_stack([one, x, m, C])         # y ~ x + m + cov
    Xt = np.column_stack([one, x, C])            # y ~ x + cov
    n = len(s)

    c1 = np.ones(n)
    a0 = float(_wls_beta(Xm, m, c1)[1])
    bo = _wls_beta(Xo, y, c1); b0 = float(bo[2]); cp0 = float(bo[1])
    c0 = float(_wls_beta(Xt, y, c1)[1])

    rng = np.random.default_rng(seed)
    ind = np.empty(n_rep); dirv = np.empty(n_rep); tot = np.empty(n_rep)
    for t in range(n_rep):
        cc = np.bincount(rng.integers(0, n, n), minlength=n).astype(float)
        a = float(_wls_beta(Xm, m, cc)[1])
        bb = _wls_beta(Xo, y, cc)
        ind[t] = a * float(bb[2])
        dirv[t] = float(bb[1])
        tot[t] = float(_wls_beta(Xt, y, cc)[1])
    ident = float(np.abs(ind - (tot - dirv)).max())     # 线性模型恒等式 c = c′ + a·b

    def _pc(v, est):
        qq = np.nanpercentile(v, [2.5, 97.5])
        pv = 2.0 * min(float(np.mean(v <= 0)), float(np.mean(v >= 0)))
        return est, qq[0], qq[1], max(pv, 1.0 / n_rep)

    i_est, i_lo, i_hi, i_p = _pc(ind, a0 * b0)
    d_est, d_lo, d_hi, d_p = _pc(dirv, cp0)
    t_est, t_lo, t_hi, t_p = _pc(tot, c0)
    pm = ind / tot
    p_est, p_lo, p_hi, _ = _pc(pm[np.isfinite(pm)], (a0 * b0) / c0 if abs(c0) > 1e-9 else np.nan)
    return dict(n=n, a=a0, b=b0,
                indirect=i_est, ind_lo=i_lo, ind_hi=i_hi, ind_p=i_p,
                direct=d_est, dir_lo=d_lo, dir_hi=d_hi, dir_p=d_p,
                total=t_est, tot_lo=t_lo, tot_hi=t_hi, tot_p=t_p,
                prop=p_est, prop_lo=p_lo, prop_hi=p_hi,
                ident_check=ident)

log(f"[Step 16] 探索性中介 {Z_STRESS} → {M_COL} → 就业选择（自举 {N_REP} 次）…")
t16_rows = []
for col, cn, en, kind in OUTCOMES:
    r = boot_mediation(d, col, M_COL, Z_STRESS, COVS16)
    r.update(dict(结局=cn, 英文=en))
    t16_rows.append(r)
    log(f"  {cn}: indirect={r['indirect']:+.4f} [{r['ind_lo']:+.4f},{r['ind_hi']:+.4f}] "
        f"prop={r['prop']:.3f} 恒等式残差={r['ident_check']:.2e}")
t16 = pd.DataFrame(t16_rows)
t16["ind_p_BH"] = bh_adjust(t16["ind_p"].values)
_mx = float(t16["ident_check"].max())
log(f"  线性恒等式 c = c′ + a·b 的最大残差 = {_mx:.2e}"
    f"{'  ✅' if _mx < 1e-8 else '  ⚠'}")

# ---- 与 statsmodels.Mediation 的口径对照（证明自实现没写错）----
_chk_col, _chk_cn = "Q10_1", "国企 / 事业单位"
_sub = d[[_chk_col, M_COL, Z_STRESS] + COVS16].dropna().reset_index(drop=True)
_f = f"{_chk_col} ~ {M_COL} + {Z_STRESS} + " + " + ".join(COVS16)
_g = f"{M_COL} ~ {Z_STRESS} + " + " + ".join(COVS16)
_mm = Mediation(sm.OLS.from_formula(_f, _sub), sm.OLS.from_formula(_g, _sub),
                exposure=Z_STRESS, mediator=M_COL).fit(n_rep=N_REP_CHK,
                                                       rng=np.random.default_rng(SEED))
_sm = _mm.summary()          # ⚠ 返回的是**以效应名为索引的 DataFrame**，不是嵌套字典
_my = t16[t16["结局"] == _chk_cn].iloc[0]
CHK = dict(outcome=_chk_cn, mine=float(_my["indirect"]),
           mine_ci=(float(_my["ind_lo"]), float(_my["ind_hi"])),
           statsmodels=float(_sm.loc["ACME (average)", "Estimate"]),
           statsmodels_ci=(float(_sm.loc["ACME (average)", "Lower CI bound"]),
                           float(_sm.loc["ACME (average)", "Upper CI bound"])))
log(f"  对照 statsmodels.Mediation（{_chk_cn}）："
    f"自实现 {CHK['mine']:+.5f} [{CHK['mine_ci'][0]:+.5f},{CHK['mine_ci'][1]:+.5f}] vs "
    f"statsmodels {CHK['statsmodels']:+.5f} "
    f"[{CHK['statsmodels_ci'][0]:+.5f},{CHK['statsmodels_ci'][1]:+.5f}]")

# ======================================================================
# 5. Step 17 · 调节效应（表 17、图 13）
# ======================================================================
MODS = [
    ("年级", [f"grade_{g}" for g in GRADE_ORDER[1:]],
     GRADE_ORDER, ["Y1 (ref.)", "Y2", "Y3", "Y4", "Grad"]),
    ("专业", [f"major_{m}" for m in MAJ_ORDER[1:]],
     MAJ_ORDER, ["Eng (ref.)", "Sci", "Econ", "Hum", "Other"]),
    ("性别", ["female"], ["male", "female"], ["Male (ref.)", "Female"]),
]

def ord_fit(y_codes, X):
    """OrderedModel 拟合（必须传 DataFrame：按 exog_names 取系数）。

    ⚠ 失败时要**把原因打印出来**：bfgs 收敛失败会自动退回 nm，若两者都抛异常而这里
      静默吞掉，就会像本脚本初版那样九个结局全部报"拟合失败"却查不出原因
      （当时的真因是顶层没 import OrderedModel，走了 `except Exception: continue`）。
    """
    last = None
    for mth, kw in (("bfgs", {}), ("nm", {"maxiter": 4000})):
        try:
            res = OrderedModel(np.asarray(y_codes), X, distr="logit").fit(
                method=mth, disp=0, **kw)
        except Exception as e:          # noqa: BLE001（ deliberately broad）
            last = e
            continue
        b = np.asarray(res.params)
        if np.all(np.isfinite(b)):
            return res
    if last is not None:
        log(f"    ⚠ OrderedModel 两次都抛异常：{type(last).__name__}: {last}")
    return None

def _codes_for(col, kind, series):
    yv = np.asarray(series, dtype=float)
    if kind == "ord5":
        codes, note, groups = collapse_sparse(yv, min_n=40)
        return codes, ("合并：" + "、".join(note) if note else "5 级直接入模")
    q = pd.qcut(pd.Series(yv), 5, duplicates="drop")
    return pd.Series(q).cat.codes.values.astype(int), "五分位分箱 5 档"

log("[Step 17] 调节效应：压力 × 年级 / 专业 / 性别（分组 LR 检验）…")
mod_rows, slope_rows = [], []
for col, cn, en, kind in OUTCOMES:
    mask = d[col].notna().values
    sd = d[mask]
    codes, note = _codes_for(col, kind, sd[col])
    Xb = design(sd, Z_STRESS)
    rb = ord_fit(codes, Xb)
    if rb is None:
        log(f"  ⚠ {cn} 基线模型拟合失败")
        continue
    for modname, dcols, levels, labels in MODS:
        Xi = Xb.copy()
        for c in dcols:
            Xi[f"zx_{c}"] = (sd[c].values.astype(float) if c not in Xb.columns
                             else Xb[c].values.astype(float)) * Xb[Z_STRESS].values
        ri = ord_fit(codes, Xi)
        if ri is None:
            continue
        lr = 2.0 * (float(ri.llf) - float(rb.llf))
        dfree = len(dcols)
        pv = float(stats.chi2.sf(lr, dfree))
        mod_rows.append(dict(结局=cn, 英文=en, 调节变量=modname, LR=lr, df=dfree, p=pv))
        # 简单斜率
        if pv < .05:
            exl = list(ri.model.exog_names)
            iz = exl.index(Z_STRESS)
            V = np.asarray(ri.cov_params()); bb = np.asarray(ri.params)
            oz = float(bb[iz]); vz = V[iz, iz]
            for lev, lab in zip(levels, labels):
                if lev == levels[0]:
                    m_, s_ = oz, np.sqrt(vz)
                else:
                    dd = f"zx_{lev}" if modname == "性别" else \
                        f"zx_{'female' if False else lev}"
                    key = [c for c in Xi.columns if c.endswith(lev) and c.startswith("zx_")]
                    if modname == "性别":
                        key = ["zx_female"]
                    if not key:
                        continue
                    jj = exl.index(key[0])
                    m_ = oz + float(bb[jj])
                    s_ = np.sqrt(max(vz + V[jj, jj] + 2.0 * V[iz, jj], 0))
                slope_rows.append(dict(结局=cn, 英文=en, 调节变量=modname, 水平=lab,
                                       OR=float(np.exp(m_)),
                                       CI_lo=float(np.exp(m_ - Z975 * s_)),
                                       CI_hi=float(np.exp(m_ + Z975 * s_)),
                                       p_interaction=pv))
    log(f"  {cn} 完成")
t17 = pd.DataFrame(mod_rows)
t17s = pd.DataFrame(slope_rows) if slope_rows else pd.DataFrame()
if len(t17):
    t17["p_BH"] = bh_adjust(t17["p"].values)
    _nsig = int((t17["p_BH"] < .05).sum())
    log(f"  27 项交互 BH 后显著 {_nsig} 项")
else:
    t17["p_BH"] = []
    _nsig = 0
    log("  ⚠ 没有任何模型拟合成功，Step 17 无结果 —— 先查 ord_fit 的失败原因")

# ======================================================================
# 6. 图
# ======================================================================
# ---- 图 10：AME 点图 ----
log("[图 10] 意向4类 AME 点图 …")
gp = t15_ame[t15_ame["自变量组"] == "总指数"].copy()
cats = ["直接就业", "深造", "求稳体制", "方向未定"]
gp["ord"] = gp["类别"].map({c: i for i, c in enumerate(cats)})
gp = gp.sort_values("ord")
xx = np.arange(len(gp))
fig, ax = plt.subplots(figsize=(6.85, 3.6))
ax.axhline(0, color="#888780", lw=1, ls="--", zorder=1)
ax.errorbar(xx, gp["AME"], yerr=[gp["AME"] - gp["lo"], gp["hi"] - gp["AME"]],
            fmt="o", ms=7, lw=0, elinewidth=1.6, ecolor="#185FA5",
            mfc="#185FA5", mec="white", mew=1, capsize=4, zorder=3)
for i, (_, r) in enumerate(gp.iterrows()):
    tag = "BH n.s." if r["p"] >= .05 else (
        "BH p<.001" if r["p"] < .001 else f"BH p={r['p']:.3f}")
    ax.text(i, r["hi"] + 0.0016, f"{r['AME']:+.4f}\n{tag}", ha="center",
            va="bottom", fontsize=8, color="#444441")
ax.set_xticks(xx)
ax.set_xticklabels([CATS_EN[c] for c in gp["类别"]], fontsize=9)
_lo0, _hi0 = float(gp["lo"].min()), float(gp["hi"].max())
_rng = _hi0 - _lo0
ax.set_ylim(_lo0 - 0.10 * _rng, _hi0 + 0.34 * _rng)
ax.set_ylabel("Average marginal effect per 1 SD of stressor index", fontsize=9.5)
ax.tick_params(axis="x", labelsize=8.5)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig_RQ3_ame.png"), dpi=DPI, facecolor="white")
plt.close(fig)

# ---- 图 12：探索性中介路径图 + 间接关联 ----
# 版式：上下两栏。初版做成左右两栏，(a) 只有 ~2.9 in 宽，三个节点框必然互相压字、
# 箭头标签盖住框边 —— 路径图要的是横向留白，只能给它整行。
log("[图 12] 探索性中介路径图 + 间接关联 …")
fig, axes = plt.subplots(2, 1, figsize=(6.85, 6.5),
                         gridspec_kw={"height_ratios": [1.0, 1.45]})
ax = axes[0]
_best = t16.sort_values("prop", ascending=False).iloc[0]
_mrow = t16[t16["结局"] == _best["结局"]].iloc[0]
_ny, _bw, _bh = 0.58, 0.20, 0.22
_nodes = {"X": 0.13, "M": 0.50, "Y": 0.87}
_lab = {"X": "Stressor\nindex (X)", "M": "Expectation\nimpact (M)",
        "Y": "Employment\npreference (Y)"}
for k, px in _nodes.items():
    ax.add_patch(mpatches.FancyBboxPatch((px - _bw / 2, _ny - _bh / 2), _bw, _bh,
                                         boxstyle="round,pad=0.012",
                                         fc="#EEF3FA", ec="#185FA5", lw=1.1))
    ax.text(px, _ny, _lab[k], ha="center", va="center", fontsize=8.6, color="#2C2C2A")
# 两条水平路径
ax.annotate("", xy=(_nodes["M"] - _bw / 2, _ny), xytext=(_nodes["X"] + _bw / 2, _ny),
            arrowprops=dict(arrowstyle="-|>", color="#A32D2D", lw=1.2,
                            shrinkA=0, shrinkB=0))
ax.annotate("", xy=(_nodes["Y"] - _bw / 2, _ny), xytext=(_nodes["M"] + _bw / 2, _ny),
            arrowprops=dict(arrowstyle="-|>", color="#A32D2D", lw=1.2,
                            shrinkA=0, shrinkB=0))
ax.text((_nodes["X"] + _nodes["M"]) / 2, _ny + 0.075, f"a = {_mrow['a']:+.3f}",
        ha="center", fontsize=8.6, color="#A32D2D")
ax.text((_nodes["M"] + _nodes["Y"]) / 2, _ny + 0.075, f"b = {_mrow['b']:+.3f}",
        ha="center", fontsize=8.6, color="#A32D2D")
# 直接路径：从 X 底部绕到 Y 底部，不从中间的 M 框穿过
ax.annotate("", xy=(_nodes["Y"], _ny - _bh / 2), xytext=(_nodes["X"], _ny - _bh / 2),
            arrowprops=dict(arrowstyle="-|>", color="#185FA5", lw=1.2,
                            shrinkA=0, shrinkB=0,
                            connectionstyle="arc3,rad=0.42"))
ax.text(0.5, 0.20, f"c' = {_mrow['direct']:+.3f}", ha="center", fontsize=8.6,
        color="#185FA5")
ax.text(0.5, 0.90, f"Indirect a*b = {_mrow['indirect']:+.4f}   "
                   f"({_mrow['prop']*100:.1f}% of total)", ha="center",
        fontsize=9, color="#2C2C2A")
# ⚠ 结局名不要单独放一行：(a) 栏高度只有约 2.2 in，y=0.96 与 y=0.90 两行必然重叠。
#   并入子图标题最稳妥（标题在轴外，不会与轴内元素互撞）。
ax.set_xlim(0, 1); ax.set_ylim(0.15, 1.0); ax.axis("off")
ax.set_title(f"(a)   Outcome: {_mrow['英文']}", fontsize=10, pad=3, loc="left")

ax = axes[1]
gg = t16.sort_values("indirect")
yy = np.arange(len(gg))
ax.axvline(0, color="#888780", lw=1, ls="--", zorder=1)
ax.errorbar(gg["indirect"], yy,
            xerr=[gg["indirect"] - gg["ind_lo"], gg["ind_hi"] - gg["indirect"]],
            fmt="o", ms=6, lw=0, elinewidth=1.5, ecolor="#A32D2D",
            mfc="#A32D2D", mec="white", mew=1, capsize=3.5, zorder=3)
_pad2 = float((gg["ind_hi"] - gg["ind_lo"]).max())
for i, (_, r) in enumerate(gg.iterrows()):
    tag = "BH n.s." if r["ind_p_BH"] >= .05 else f"BH {r['ind_p_BH']:.3f}"
    # ⚠ 标签一律放在 CI 的「远端外侧」：负效应的 CI 右端紧贴 0，若仍按右端加偏移，
    #   标签会横跨 0 参考线（初版就是这样，「Grassroots employment」那行读不清）。
    if r["indirect"] < 0:
        ax.text(r["ind_lo"] - _pad2 * 0.05, i, f"{r['indirect']:+.4f} {tag}  ",
                va="center", ha="right", fontsize=7.8, color="#444441")
    else:
        ax.text(r["ind_hi"] + _pad2 * 0.05, i, f"  {r['indirect']:+.4f} {tag}",
                va="center", ha="left", fontsize=7.8, color="#444441")
ax.set_yticks(yy); ax.set_yticklabels(list(gg["英文"]), fontsize=8.6)
_w = float(max(np.abs(gg["ind_lo"].min()), np.abs(gg["ind_hi"].max())))
ax.set_xlim(-_w * 1.95, _w * 2.35)     # 左侧多留：负效应标签现在放在 CI 左端外面
ax.set_xlabel("Indirect association a*b (bootstrap 95% CI)", fontsize=9)
ax.spines[["top", "right"]].set_visible(False)
ax.set_title("(b)", fontsize=10, pad=3, loc="left")
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig_RQ3_path.png"), dpi=DPI, facecolor="white")
plt.close(fig)

# ---- 图 13：分层 OR ----
log("[图 13] 分层 OR（按最显著的调节变量）…")
FIG13_MOD = None
FIG13_MODE = ""
if len(t17):
    _mdf = t17.groupby("调节变量").agg(
        nsig=("p_BH", lambda v: int((v < .05).sum())),
        minpbh=("p_BH", "min"))
    _pick = _mdf.sort_values(["nsig", "minpbh"], ascending=[False, True]).index[0]
    _lv = MODS[[m[0] for m in MODS].index(_pick)][3]
    _en = {cn: en for _, cn, en, _ in OUTCOMES}
    _sub = t17s[t17s["调节变量"] == _pick].copy() if len(t17s) else pd.DataFrame()

    if len(_sub):
        # 与其他森林图保持一致：结局在 y 轴、OR 在 x 轴（对数）
        _oos = [cn for cn in _en if cn in set(_sub["结局"])]
        _lv_present = [l for l in _lv if l in set(_sub["水平"])]
        _cmap = plt.get_cmap("viridis")
        _cols = [_cmap(t) for t in np.linspace(0.05, 0.85, max(len(_lv_present), 2))]
        yy = np.arange(len(_oos))
        fig, ax = plt.subplots(figsize=(6.85, 5.0))
        # 交替浅灰带：5 个水平 × 5 个结局共 25 条线，没有块分隔时读不出「哪几条属于同一结局」
        for _i in range(len(_oos)):
            if _i % 2 == 0:
                ax.axhspan(_i - 0.5, _i + 0.5, color="#F2F4F7", zorder=0)
        for li, lab in enumerate(_lv_present):
            r_ = _sub[_sub["水平"] == lab].set_index("结局")
            xs, lo, hi = [], [], []
            for cn in _oos:
                if cn in r_.index:
                    xs.append(float(r_.loc[cn, "OR"]))
                    lo.append(float(r_.loc[cn, "CI_lo"]))
                    hi.append(float(r_.loc[cn, "CI_hi"]))
                else:
                    xs.append(np.nan); lo.append(np.nan); hi.append(np.nan)
            xs = np.array(xs); lo = np.array(lo); hi = np.array(hi)
            _off = (li - (len(_lv_present) - 1) / 2) * 0.16
            ax.errorbar(xs, yy + _off, xerr=[xs - lo, hi - xs],
                        fmt="o", ms=5.5, lw=0, elinewidth=1.3, color=_cols[li],
                        mfc=_cols[li], mec="white", mew=0.6, capsize=2.5, zorder=3)
        ax.axvline(1.0, color="#888780", lw=1, ls="--", zorder=1)
        ax.set_yticks(yy)
        ax.set_yticklabels([_en[cn] for cn in _oos], fontsize=8.3)
        ax.set_xscale("log")
        _allv = np.concatenate([_sub["OR"].values, _sub["CI_lo"].values,
                                _sub["CI_hi"].values])
        _allv = _allv[np.isfinite(_allv) & (_allv > 0)]
        _xlo, _xhi = float(_allv.min()) * 0.90, float(_allv.max()) * 1.10
        ax.set_xlim(_xlo, _xhi)
        # ⚠ 对数轴不要让 matplotlib 自己写刻度：它会输出 mathtext 的
        #   「3 × 10^{-1}」形式，与其他图的显式刻度不一致（本项目已踩过一次）。
        _tk = [t for t in (0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0)
               if _xlo <= t <= _xhi]
        if _tk:
            ax.set_xticks(_tk)
            ax.set_xticklabels([f"{t:g}" for t in _tk])
            ax.minorticks_off()
        ax.set_xlabel("Stratified OR per 1 SD of stressor index (95% CI, log scale)",
                      fontsize=9)
        ax.tick_params(axis="x", labelsize=8.5)
        # ⚠ 图例标题必须英文：图内字体是 Arial，中文会渲染成缺字形方框
        _mod_en = {"年级": "Grade", "专业": "Major", "性别": "Sex"}.get(_pick, _pick)
        # 图例移到轴下方横排：放 lower right 会压住最低那一块（薄层 Y4 的 CI 能伸到 2.1）
        ax.legend([mpatches.Patch(color=_cols[i]) for i in range(len(_lv_present))],
                  _lv_present, fontsize=8, loc="upper center",
                  bbox_to_anchor=(0.5, -0.155), ncol=len(_lv_present),
                  frameon=False, title=_mod_en, title_fontsize=8,
                  columnspacing=1.2, handlelength=1.4, handletextpad=0.4,
                  borderaxespad=0)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        fig.savefig(os.path.join(FIG, "fig_RQ3_moderation.png"), dpi=DPI,
                    facecolor="white")
        plt.close(fig)
        FIG13_MOD = _pick
        FIG13_MODE = "分层 OR"
    else:
        # 没有交互达到 p<.05：不能出空的森林图，改为如实展示「三块交互都不显著」
        fig, ax = plt.subplots(figsize=(6.85, 3.6))
        _g = t17.copy()
        _g["neglog10p"] = -np.log10(np.clip(_g["p_BH"].values, 1e-12, 1))
        _mm = [m[0] for m in MODS]
        for mi, mname in enumerate(_mm):
            _r = _g[_g["调节变量"] == mname]
            _j = np.array([mi] * len(_r)) + (np.arange(len(_r)) - len(_r) / 2) * 0.12
            ax.scatter(_j, _r["neglog10p"], s=34, alpha=.85, zorder=3)
        ax.axhline(-np.log10(.05), color="#A32D2D", lw=1.2, ls="--", zorder=2)
        ax.text(2.42, -np.log10(.05) + 0.05, "BH p = .05", fontsize=8,
                color="#A32D2D", ha="right")
        ax.set_xticks(range(len(_mm)))
        ax.set_xticklabels([{"年级": "Grade", "专业": "Major", "性别": "Sex"}.get(m, m)
                            for m in _mm], fontsize=9.5)
        ax.set_ylabel("-log10 (BH-adjusted p) of block LR test", fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
        ax.text(0.5, 0.05, "No stress-by-moderator interaction reached significance",
                transform=ax.transAxes, ha="center", fontsize=8.5, color="#5F5E5A")
        fig.tight_layout()
        fig.savefig(os.path.join(FIG, "fig_RQ3_moderation.png"), dpi=DPI,
                    facecolor="white")
        plt.close(fig)
        FIG13_MOD = _pick
        FIG13_MODE = "无显著交互（如实展示）"
else:
    FIG13_MOD = None

# ======================================================================
# 7. Excel
# ======================================================================
log("写出 Excel …")
sheets = {
    "表14_意向RRR": t15,
    "表14_意向AME": t15_ame.drop(columns=["AME_解析"]),
    "表14_维度差异": t15_diff,
    "表14_IIA敏感性": t15_sens,
    "表16_中介": t16.drop(columns=["ident_check"]),
    "表17_交互LR": t17,
}
if len(t17s):
    sheets["表17_简单斜率"] = t17s.drop(columns=["是否BH显著"], errors="ignore")
with pd.ExcelWriter(XLSX, engine="openpyxl") as w:
    for nm, tb in sheets.items():
        tb.to_excel(w, sheet_name=nm, index=False)
log(f"  ✅ {XLSX}（{len(sheets)} sheet）")

# ======================================================================
# 8. 报告
# ======================================================================
log("写出报告 …")

for nm, rc in refcheck:
    pass

# ---- Step 15 ----
for grp in ["总指数", "双因子"]:
    for c in (["z_压力指数"] if grp == "总指数" else [ZF1, ZF2]):
        gi = t15[(t15["自变量组"] == grp) & (t15["自变量"] == c)]
        for _, r in gi.iterrows():
            pass
if len(t15_diff):
    for _, r in t15_diff.iterrows():
        pass
if len(t15_ame):
    for _, r in t15_ame.iterrows():
        if r["自变量组"] != "总指数":
            continue
for _, r in t15_sens.iterrows():
    pass
_sens_bad = t15_sens[t15_sens["判定"].str.startswith("⚠")]

# ---- Step 16 ----
for _, r in t16.iterrows():
    pass
_ns16 = int((t16["ind_p_BH"] < .05).sum())
# ⚠ 负中介占比 = **不一致中介 / 压制**：间接与直接效应符号相反，此时
#   indirect/total 失去比例含义（可能 >1、<0），**不可当作「解释了多少」来读**。
_neg = t16[t16["prop"] < 0]
if len(_neg):
    pass

# ---- Step 17 ----
if len(t17):
    for _, r in t17.iterrows():
        pass
if len(t17s):
    for _, r in t17s.iterrows():
        pass
    # ---- 跨结局的一致性：每个结局内部把各水平的 OR 排序，看最值是否落在同一水平 ----
    _rk = []
    for (cn, mn), gg in t17s.groupby(["结局", "调节变量"]):
        gg = gg.copy()
        gg["rk_max"] = gg["OR"].rank(ascending=False) == 1
        gg["rk_min"] = gg["OR"].rank(ascending=True) == 1
        _rk.append(gg)
    _rk = pd.concat(_rk)
    _concl = (_rk.groupby(["调节变量", "水平"])
                .agg(涉及结局数=("结局", "nunique"),
                     该层最强次数=("rk_max", "sum"),
                     该层最弱次数=("rk_min", "sum"))
                .reset_index())
    _concl = _concl[_concl["涉及结局数"] >= 3]
    if len(_concl):
        for _, r in _concl.iterrows():
            pass
        _mode = _concl.sort_values("该层最弱次数", ascending=False).iloc[0]
        if int(_mode["该层最弱次数"]) >= 3:
            pass
        _mode_s = _concl.sort_values("该层最强次数", ascending=False).iloc[0]
        if int(_mode_s["该层最强次数"]) >= 3:
            pass
        _thin = {"Y4": 45, "Other": 84, "Hum": 77}
        _hit = [k for k in _thin if k in set(_concl["水平"])]
        if _hit:
            pass
else:
    pass

# ---- 话术 ----

# ---- 遗留 ----

log("批 4b 完成。")
