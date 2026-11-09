# -*- coding: utf-8 -*-
r"""
Step 7–11 · 研究问题二：年级梯度（grade gradient）
======================================================================
输入：论文二数据集_1026.xlsx / 论文二数据集_1579.xlsx（Step 1 产出）
输出：论文二_输出/Step7_11_年级梯度结果.xlsx
      figs2/fig_RQ2_trend.png        图 5  六结局五档趋势（2×3）
      figs2/fig_RQ2_intention.png    图 6  意向4类堆叠比例条
      figs2/fig_RQ2_forest.png       图 7  OR 森林图（1026 vs 1579）
      figs2/fig_RQ2_shape.png        图 8  正交多项式形状诊断（线性 vs 二次）

⚠ 措辞红线：全文写 grade gradient / grade differences；
   禁止 "随年级升高而变化" "发展趋势" "developmental"。

自实现的统计量：
  · Jonckheere–Terpstra 趋势检验（scipy 1.18 无此函数）
      —— 统计量用列联表计数，p 值用 10000 次多项抽样置换
  · Brant 平行线检验（statsmodels 无此检验）
      —— delta 法联合 Wald，总量 + 逐变量
  · Kendall's tau-b 的 bootstrap 95% CI
  · 加权正交多项式对比码（Gram–Schmidt）
"""
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
from scipy.linalg import block_diag
from scipy.optimize import minimize
from scipy.special import expit
from statsmodels.tools.numdiff import approx_hess
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

XLSX = os.path.join(OUT, "Step7_11_年级梯度结果.xlsx")

plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["pdf.fonttype"] = 42
DPI = 600
SEED = 42
Z975 = 1.959963984540054      # 正态 97.5% 分位，手算 Wald CI 用
N_PERM = 10000          # JT 置换次数（卡点 4）
N_BOOT_TAU = 2000       # tau-b bootstrap
N_BOOT_AME = 400        # MNLogit AME bootstrap

GRADE_ORDER = ["y1", "y2", "y3", "y4", "grad"]
GRADE3_ORDER = ["y1", "y2", "y3", "grad_group"]
GRADE_CN = {"y1": "大一", "y2": "大二", "y3": "大三", "y4": "大四", "grad": "研究生"}
GRADE_EN = {"y1": "Year 1", "y2": "Year 2", "y3": "Year 3", "y4": "Year 4", "grad": "Postgrad"}
GRADE_SHORT = {"y1": "Y1", "y2": "Y2", "y3": "Y3", "y4": "Y4", "grad": "PG"}
GRADE3_EN = {"y1": "Year 1", "y2": "Year 2", "y3": "Year 3", "grad_group": "Graduating"}

# 六个结局：(列, 中文名, 英文短名, 类型)
OUTCOMES = [
    ("Q4_压力源指数",            "压力源指数",     "Stressor index",    "cont"),
    ("Q5_身心健康",              "身心健康影响",   "Health impact",     "ord5"),
    ("Q10_体制内偏好",           "体制内偏好",     "Public-sector pref.", "ord5"),
    ("Q10_灵活新就业偏好",       "灵活新业态偏好", "Flexible-work pref.", "cont"),
    ("Q11_指数A_剔除不确定",     "期望受影响指数", "Expectation impact", "cont"),
    ("意向4类",                  "职业意向",       "Career intention",  "nominal"),
]
NUM_OUT = [o for o in OUTCOMES if o[3] != "nominal"]   # 五个可排序结局
INTENT_REF = "直接就业"

def log(msg=""):
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)

# ======================================================================
# 1. 统计工具（自实现部分）
# ======================================================================
def bh_adjust(p):
    """Benjamini–Hochberg 校正，返回与 p 同形的 q 值。"""
    p = np.asarray(p, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    q = ranked * n / (np.arange(n) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0, 1)
    out = np.empty(n)
    out[order] = q
    return out

def cramers_v(x, y):
    """Cramér's V + Bergsma 偏差校正版 + χ²。"""
    tab = pd.crosstab(x, y).values.astype(float)
    n = tab.sum()
    chi2 = stats.chi2_contingency(tab, correction=False)[0]
    phi2 = chi2 / n
    r, k = tab.shape
    phi2c = max(0.0, phi2 - (k - 1) * (r - 1) / (n - 1))
    rc = r - (r - 1) ** 2 / (n - 1)
    kc = k - (k - 1) ** 2 / (n - 1)
    denom = min(kc - 1, rc - 1)
    v = np.sqrt(phi2 / min(k - 1, r - 1))
    vc = np.sqrt(phi2c / denom) if denom > 0 else np.nan
    dof = (r - 1) * (k - 1)
    return dict(chi2=chi2, dof=dof, p=stats.chi2.sf(chi2, dof), V=v, V_corr=vc, n=int(n))

def _jt_from_table(cnt):
    """由 水平×组 计数列联表算 JT 统计量。
    M[a,b] = 1 if a<b, 0.5 if a==b, 0 otherwise。"""
    V, G = cnt.shape
    M = np.triu(np.ones((V, V)), 1) + 0.5 * np.eye(V)
    total = 0.0
    for i in range(G - 1):
        for j in range(i + 1, G):
            total += cnt[:, i] @ M @ cnt[:, j]
    return total

def jt_test(values, groups, n_perm=N_PERM, seed=SEED):
    """Jonckheere–Terpstra 有序备择趋势检验。
    H1: 组均值按 groups 给定顺序单调上升。
    统计量仅用于两侧/单侧方向判断，p 值由多项抽样置换获得（卡点 4）。
    """
    vals = np.asarray(values)
    grp = pd.Categorical(groups, categories=list(dict.fromkeys(groups)), ordered=True)
    ok = ~pd.isna(vals)
    vals, gcode = vals[ok], grp.codes[ok]
    gcode = np.asarray(gcode)
    okcnt = gcode >= 0
    vals, gcode = vals[okcnt], gcode[okcnt]

    levels, inv = np.unique(vals, return_inverse=True)
    G = len(np.unique(gcode))
    cnt = np.zeros((len(levels), G))
    np.add.at(cnt, (inv, gcode), 1.0)
    ns = cnt.sum(axis=0)
    jt_obs = _jt_from_table(cnt)

    rng = np.random.default_rng(seed)
    rN = cnt.sum(axis=1) / cnt.sum()
    jt_perm = np.empty(n_perm)
    for b in range(n_perm):
        sim = np.column_stack([rng.multinomial(int(n), rN) for n in ns])
        jt_perm[b] = _jt_from_table(sim)

    # 单侧：观测值 ≥ 置换分布（升序趋势）
    p_up = float((np.sum(jt_perm >= jt_obs) + 1) / (n_perm + 1))
    p_two = float((np.sum(np.abs(jt_perm - jt_perm.mean())
                          >= abs(jt_obs - jt_perm.mean())) + 1) / (n_perm + 1))

    # 标准化 z
    z = (jt_obs - jt_perm.mean()) / jt_perm.std(ddof=1)
    return dict(JT=float(jt_obs), E_JT=float(jt_perm.mean()),
                SD_JT=float(jt_perm.std(ddof=1)), z=float(z),
                p_one=p_up, p_two=p_two, n=int(len(vals)))

def kendall_tau_ci(x, y, n_boot=N_BOOT_TAU, seed=SEED):
    """Kendall's tau-b + bootstrap 百分位 CI。"""
    m = np.asarray(~pd.isna(x)) & np.asarray(~pd.isna(y))
    x = np.asarray(x)[m]
    y = np.asarray(y)[m]
    tau, p = stats.kendalltau(x, y, variant="b")
    rng = np.random.default_rng(seed)
    n = len(x)
    bs = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        try:
            bs[b] = stats.kendalltau(x[idx], y[idx], variant="b")[0]
        except Exception:
            bs[b] = np.nan
    bs = bs[~np.isnan(bs)]
    return dict(tau=float(tau), p=float(p),
                lo=float(np.percentile(bs, 2.5)), hi=float(np.percentile(bs, 97.5)))

def ortho_poly_codes(s):
    """加权（组规模）Gram–Schmidt 正交多项式：线性 + 二次对比码。
    返回 DataFrame['L'], DataFrame['Q']，二者均与常数向量正交且彼此正交。"""
    s = np.asarray(s, dtype=float)
    xs = np.unique(s)
    nlev = np.array([(s == v).sum() for v in xs], dtype=float)

    def iprod(a, b):
        return float(np.sum(nlev * a * b))

    one = np.ones_like(xs)
    xmean = iprod(xs, one) / iprod(one, one)
    L = xs - xmean
    q0 = xs ** 2 - iprod(xs ** 2, one) / iprod(one, one)
    Q = q0 - L * (iprod(xs ** 2, L) / iprod(L, L))
    return pd.DataFrame({"L": np.interp(s, xs, L), "Q": np.interp(s, xs, Q)})

def brant_test(y_ord, X, names):
    """Brant 平行线（比例优势）检验 —— statsmodels 未提供，自实现。
    对 J−1 个累积切点各拟合一个**带截距**的二项 logit，用 delta 法联合检验
    各切点的**斜率向量**是否相等（截距允许随意变化，不参与检验）。

    两处极易写错，都修了：
      1. 每个切点模型必须带截距 α_j —— 不带的版本斜率估计本身就是被约束歪的；
      2. J−1 个模型在同一批观测上拟合，估计彼此相关，协方差**不能用 block_diag**
         （那等于假设独立），否则 Wald 偏大、**严格比例优势的数据也会被拒绝**。
         改用 sandwich 式 influence function：
             U_i = H_j^{-1} s_i(y_j) 按观测堆叠，  V = Σ_i U_i U_iᵀ
    """
    y = np.asarray(y_ord)
    levels = np.unique(y)
    J = len(levels)
    if J < 3:
        return None
    Xc = sm.add_constant(np.asarray(X, dtype=float))
    P = Xc.shape[1]
    sl_idx = np.arange(1, P)                      # 跳过截距，只检验斜率
    betas, U_list = [], []
    for j in range(J - 1):
        z = (y > levels[j]).astype(int)
        if z.sum() < 15 or (1 - z).sum() < 15:
            return None
        try:
            res = sm.Logit(z, Xc).fit(disp=0, maxiter=300)
        except Exception:
            return None
        b = np.asarray(res.params)
        if not np.all(np.isfinite(b)):
            return None
        betas.append(b)
        resid = (z - res.predict(Xc))[:, None]
        S = Xc.values * resid if hasattr(Xc, "values") else Xc * resid
        try:
            Hinv = np.linalg.inv(-res.model.hessian(res.params))
        except Exception:
            return None
        U_list.append(S @ Hinv.T)
    b = np.concatenate(betas)
    U = np.column_stack(U_list)
    V = U.T @ U
    k = (J - 1) - 1                               # = J − 2
    nsl = len(sl_idx)
    if k < 1 or nsl == 0:
        return None
    R = np.zeros((k * nsl, (J - 1) * P))
    for i in range(k):                            # β_{i+1} − β_0，逐斜率分量
        for a, si in enumerate(sl_idx):
            row = i * nsl + a
            R[row, si] = -1.0
            R[row, (i + 1) * P + si] = 1.0
    d = R @ b
    M = R @ V @ R.T
    try:
        W = float(d @ np.linalg.pinv(M) @ d)
    except Exception:
        return None
    df = k * nsl
    overall = (W, df, float(stats.chi2.sf(W, df)))
    per = {}
    for vi, nm in enumerate(names):
        rows = [i * nsl + vi for i in range(k)]
        dd, MM = d[rows], M[np.ix_(rows, rows)]
        try:
            w = float(dd @ np.linalg.pinv(MM) @ dd)
        except Exception:
            continue
        per[str(nm)] = (w, k, float(stats.chi2.sf(w, k)))
    return dict(overall=overall, per=per)

def _fmt_vals(vals):
    core = [int(v) for v in vals]
    return "/".join(str(v) for v in core) if len(core) > 1 else str(core[0])

def collapse_sparse(y, min_n=40):
    """有序结局：把样本过少的类别并入相邻类（保序），返回新的类别索引 0..J-1 与映射说明。"""
    y = np.asarray(y)
    vals = np.unique(y)
    groups = [[v] for v in vals]
    counts = np.array([(y == v).sum() for v in vals])
    merged_note = []
    while True:
        small = np.where(counts < min_n)[0]
        if len(small) == 0 or len(counts) <= 2:
            break
        i = int(small[0])
        if i == 0:
            j = 1
        elif i == len(counts) - 1:
            j = i - 1
        else:
            j = i - 1 if counts[i - 1] <= counts[i + 1] else i + 1
        merged_note.append(f"{_fmt_vals(groups[i])}→{_fmt_vals(groups[j])}")
        lo, hi = min(i, j), max(i, j)
        groups[lo:hi + 1] = [sorted(groups[lo] + groups[hi])]
        counts = np.concatenate([counts[:lo],
                                 [np.sum(counts[lo:hi + 1])],
                                 counts[hi + 1:]])
    mapping = {}
    for new_i, g in enumerate(groups):
        for v in g:
            mapping[v] = new_i
    return np.array([mapping[v] for v in y]), merged_note, groups

def mean_ci(v):
    v = np.asarray(v, dtype=float)
    v = v[~np.isnan(v)]
    n = len(v)
    if n < 2:
        return (np.nan, np.nan, np.nan, np.nan)
    m = v.mean()
    sd = v.std(ddof=1)
    se = sd / np.sqrt(n)
    half = stats.t.ppf(0.975, n - 1) * se
    return (m, sd, m - half, m + half)

# ======================================================================
# 2. 数据准备
# ======================================================================
log("读数据 …")
d1026 = pd.read_excel(os.path.join(BASE, "论文二数据集_1026.xlsx"))
d1579 = pd.read_excel(os.path.join(BASE, "论文二数据集_1579.xlsx"))
assert d1026.shape[0] == 1026 and d1579.shape[0] == 1579

for d in (d1026, d1579):
    d["gnum"] = d["grade"].map({g: i for i, g in enumerate(GRADE_ORDER)})
    d["gnum3"] = d["年级_合并毕业"].map({g: i for i, g in enumerate(GRADE3_ORDER)})
    d["male"] = (d["sex"] == "male").astype(int)
    d["net"] = (d["channel"] == "net").astype(int)
if "batch" not in d1026.columns:
    d1026["batch"] = 0

MAJ = sorted(d1026["major"].dropna().unique())
sheets = {}
_chk = []          # 一致性核对

# ======================================================================
# 3. Step 7 · 年级 × 专业混杂
# ======================================================================
log("[Step 7] 年级 × 专业混杂 …")
rows_v = []
for tag, d in [("1026", d1026), ("1579", d1579)]:
    r = cramers_v(d["grade"], d["major"])
    rows_v.append(dict(样本=tag, N=r["n"], χ2=r["chi2"], df=r["dof"], p=r["p"],
                       V=r["V"], V校正=r["V_corr"]))
tV = pd.DataFrame(rows_v)

tab = pd.crosstab(d1026["grade"], d1026["major"]).reindex(GRADE_ORDER)
tab.columns = [c for c in tab.columns]
tab3 = pd.crosstab(d1579["grade"], d1579["major"]).reindex(GRADE_ORDER)
sheets["表8_年级×专业"] = tab.reset_index()
cross_all = pd.concat([tab.add_suffix("_1026"), tab3.add_suffix("_1579")], axis=1)
sheets["表8_交叉表合并"] = cross_all.reset_index()

# 卡点 5：工学内大二人数
eng1026 = d1026[d1026["major"] == "eng"]["grade"].value_counts().reindex(GRADE_ORDER).fillna(0)
eng1579 = d1579[d1579["major"] == "eng"]["grade"].value_counts().reindex(GRADE_ORDER).fillna(0)
eng_tab = pd.DataFrame({"1026": eng1026, "1579": eng1579}).astype(int)
sheets["表8_工学内年级"] = eng_tab.reset_index().rename(columns={"index": "年级"})

V1, V2 = float(tV.loc[tV["样本"] == "1026", "V"].iloc[0]), float(tV.loc[tV["样本"] == "1579", "V"].iloc[0])
VC1, VC2 = float(tV.loc[tV["样本"] == "1026", "V校正"].iloc[0]), float(tV.loc[tV["样本"] == "1579", "V校正"].iloc[0])
_chk.append(f"年级×专业 Cramér's V：1026 = **{V1:.3f}**、1579 = **{V2:.3f}"
            f"（偏差校正 {VC1:.3f} / {VC2:.3f}）→ "
            + ("合并确实稀释了混杂 ✅" if V2 < V1 else "⚠ 合并未降低混杂"))
log(f"  V(1026)={V1:.3f}  V(1579)={V2:.3f}")

# ======================================================================
# 4. Step 8 · 六结局五档描述 + JT 趋势 + BH
# ======================================================================
log("[Step 8] 五档描述 + JT 置换趋势（六结局）…")

desc_rows, jt_rows, trend_curves = [], [], {}
for col, cn, en, typ in NUM_OUT:
    d = d1026[["gnum", "grade", col]].dropna()
    for g in GRADE_ORDER:
        v = d.loc[d["grade"] == g, col].values
        m, sd, lo, hi = mean_ci(v)
        desc_rows.append(dict(结局=cn, 结局英文=en, 类型=typ, 年级=g,
                              年级英文=GRADE_EN[g], N=len(v), 均值=m, SD=sd,
                              CI下限=lo, CI上限=hi,
                              中位数=float(np.median(v)) if len(v) else np.nan))
    trend_curves[cn] = pd.DataFrame(desc_rows[-len(GRADE_ORDER):])

# 意向4类的五档构成
intent_tab = pd.crosstab(d1026["grade"], d1026["意向4类"], normalize="index")
intent_tab = intent_tab.reindex(GRADE_ORDER)
intent_cnt = pd.crosstab(d1026["grade"], d1026["意向4类"]).reindex(GRADE_ORDER)
for col in ["深造", "直接就业", "求稳体制", "方向未定"]:
    if col not in intent_tab.columns:
        intent_tab[col] = np.nan
        intent_cnt[col] = 0
intent_tab = intent_tab[["深造", "直接就业", "求稳体制", "方向未定"]]
intent_cnt = intent_cnt[["深造", "直接就业", "求稳体制", "方向未定"]]
sheets["表9_意向4类构成"] = pd.concat(
    [intent_cnt.add_suffix("_n"), intent_tab.add_suffix("_pct")], axis=1).reset_index()

desc_t = pd.DataFrame(desc_rows)
sheets["表9_五档描述"] = desc_t

# JT 趋势（五个可排序结局）
_raw_jt = []
for col, cn, en, typ in NUM_OUT:
    d = d1026[["grade", "gnum", col]].dropna()
    jr = jt_test(d[col].values, d["grade"].values)
    kt = kendall_tau_ci(d[col].values, d["gnum"].values)
    _raw_jt.append(dict(结局=cn, 结局英文=en, N=jr["n"], JT=jr["JT"],
                        E_JT=jr["E_JT"], SD_JT=jr["SD_JT"], z=jr["z"],
                        p单侧=jr["p_one"], p双侧=jr["p_two"],
                        tau_b=kt["tau"], tau_lo=kt["lo"], tau_hi=kt["hi"], tau_p=kt["p"]))
jt_t = pd.DataFrame(_raw_jt)
jt_t["p_BH"] = bh_adjust(jt_t["p双侧"].values)
jt_t["p单侧_BH"] = bh_adjust(jt_t["p单侧"].values)
sheets["表9_JT趋势"] = jt_t

# 意向4类内部：各类别占比随年级的 JT
intent_jt = []
intent_codes = {"深造": 0, "直接就业": 1, "求稳体制": 2, "方向未定": 3}
for cat in ["深造", "直接就业", "求稳体制", "方向未定"]:
    y = (d1026["意向4类"] == cat).astype(int).values
    jr = jt_test(y, d1026["grade"].values)
    intent_jt.append(dict(类别=cat, JT=jr["JT"], z=jr["z"],
                          p单侧=jr["p_one"], p双侧=jr["p_two"]))
intent_jt = pd.DataFrame(intent_jt)
intent_jt["p_BH"] = bh_adjust(intent_jt["p双侧"].values)
sheets["表9_意向JT"] = intent_jt
log(f"  JT 全部完成；BH 后显著 {int((jt_t['p_BH'] < .05).sum())}/{len(jt_t)}")

# ======================================================================
# 5. Step 9 · 正交多项式形状诊断
# ======================================================================
log("[Step 9] OLS 正交多项式对比（线性 vs 二次）…")

def design(d, gcol="gnum", covs=True, extra=None):
    """构造设计矩阵：L/Q 对比码 + 协变量。"""
    codes = ortho_poly_codes(d[gcol].values)
    X = codes.copy()
    if covs:
        for m in MAJ[1:]:
            X[f"major_{m}"] = (d["major"].values == m).astype(float)
        X["male"] = d["male"].values.astype(float)
        X["net"] = d["net"].values.astype(float)
    if extra:
        for k, v in extra.items():
            X[k] = v
    return X.astype(float)

poly_rows = []
for col, cn, en, typ in NUM_OUT:
    d = d1026[["gnum", "major", "male", "net", col]].dropna().copy()
    X = design(d)
    # 模型 A：L 与 Q 同时入模 —— 这是"形状诊断"，L 检验的是**扣除曲率后的**线性成分
    res = sm.OLS(d[col].values, sm.add_constant(X)).fit()
    # 模型 B：只放 L —— 与 Step 10 的主效应口径一致，供两节对照
    res_lin = sm.OLS(d[col].values, sm.add_constant(X.drop(columns=["Q"]))).fit()
    poly_rows.append(dict(
        结局=cn, 结局英文=en, N=int(res.nobs),
        b线性=res.params["L"], se线性=res.bse["L"], p线性=res.pvalues["L"],
        b二次=res.params["Q"], se二次=res.bse["Q"], p二次=res.pvalues["Q"],
        b_仅线性=res_lin.params["L"], p_仅线性=res_lin.pvalues["L"],
        R2=res.rsquared, R2_adj=res.rsquared))
poly_t = pd.DataFrame(poly_rows)
poly_t["p线性_BH"] = bh_adjust(poly_t["p线性"].values)
poly_t["p二次_BH"] = bh_adjust(poly_t["p二次"].values)
poly_t["p_仅线性_BH"] = bh_adjust(poly_t["p_仅线性"].values)

def _shape(b_l, p_l_bh, p_q_bh):
    """形状判定必须以 **BH 校正后** 的 p 为准，否则会和报告末尾的结论自相矛盾。"""
    if p_q_bh < .05:
        return "存在拐点（二次显著）"
    if p_l_bh < .05:
        return "单调递减" if b_l < 0 else "单调递增"
    return "无显著趋势（BH 后）"

poly_t["形状判定"] = [_shape(r["b线性"], r["p线性_BH"], r["p二次_BH"])
                  for _, r in poly_t.iterrows()]
sheets["表_正交多项式"] = poly_t

log(f"  形状判定完成：{poly_t['形状判定'].tolist()}")

# ======================================================================
# 6. Step 10 · 主模型
# ======================================================================
log("[Step 10] 主模型：有序 logistic / OLS / 多项 logistic …")

def cov_matrix(d, use_batch=None):
    """协变量矩阵。**自动丢弃零方差列** —— 子集分析（如仅工学）时专业哑变量
    会变成常数（major_eng 恒 1、其余恒 0），OrderedModel 会直接报
    "There should not be a constant in the model"。

    ⚠ 采集来源是三类（纸质一批 / 网络 / 纸质二批），等价于 mode + batch 两个
    哑变量，正文声明**两者贯穿始终**。因此 mode(net) 恒放；batch 在样本内
    有方差时（合并样本 1579）自动附加。旧版把两者做成二选一，导致 1579 模型
    只控制了 mode，与正文不符。
    """
    X = pd.DataFrame(index=d.index)
    for m in MAJ[1:]:
        v = (d["major"].values == m).astype(float)
        if v.std() > 0:
            X[f"major_{m}"] = v
    X["male"] = d["male"].values.astype(float)
    X["net"] = d["net"].values.astype(float)
    if use_batch is None:
        use_batch = ("batch" in d.columns and d["batch"].nunique() > 1)
    if use_batch:
        X["batch2"] = (d["batch"].values == 1).astype(float)
    keep = [c for c in X.columns if X[c].std() > 0]
    return X[keep]

def _fmt_merge(note):
    """把 collapse_sparse 的 [np.int64(5)] → [np.int64(4)] 整理成人读的 5→4。"""
    if not note:
        return ""
    out = []
    for part in note:
        part = part.replace("[np.int64(", "").replace(")]", "").replace("[", "").replace("]", "")
        part = part.replace(" ", "")
        out.append(part)
    return "; ".join(out)

def fit_main(d, col, kind, gcol="gnum", adjust=True, ):
    """返回 dict(效应, lo, hi, p, N, 备注)。kind ∈ {cont, ord5}。"""
    need = [gcol, "major", "male", "net", "batch", col]
    dd = d[need].dropna().copy()
    X = cov_matrix(dd) if adjust else pd.DataFrame(index=dd.index)
    X.insert(0, "grade", dd[gcol].values.astype(float))
    if kind == "cont":
        res = sm.OLS(dd[col].values, sm.add_constant(X)).fit(cov_type="HC3")
        b = res.params["grade"]
        ci = res.conf_int().loc["grade"].values
        return dict(model="OLS (HC3)", beta=float(b), OR=np.nan,
                    lo=float(ci[0]), hi=float(ci[1]), p=float(res.pvalues["grade"]),
                    N=int(res.nobs), extra="")
    dd = dd.copy()
    y_raw = dd[col].values
    y_codes, note, groups = collapse_sparse(y_raw, min_n=40)
    dd["_y"] = y_codes
    try:
        res = OrderedModel(dd["_y"].values, X, distr="logit").fit(method="bfgs", disp=0)
    except Exception:
        try:
            res = OrderedModel(dd["_y"].values, X, distr="logit").fit(method="nm", disp=0, maxiter=2000)
        except Exception:
            return dict(model="OrderedModel", beta=np.nan, OR=np.nan, lo=np.nan, hi=np.nan,
                        p=np.nan, N=int(len(dd)), extra="拟合失败")
    b = float(res.params["grade"])
    lo, hi = res.conf_int().loc["grade"].values
    # 方向实证校验：OR>1 是否真的对应"向高等级移动"
    Xp = X.copy()
    Xp["grade"] = float(X["grade"].max())
    p_hi = np.asarray(res.predict(exog=Xp)).mean(axis=0)
    Xp["grade"] = float(X["grade"].min())
    p_lo = np.asarray(res.predict(exog=Xp)).mean(axis=0)
    top = int(np.argmax(p_hi - p_lo) == (p_hi.shape[0] - 1))
    br = brant_test(y_codes, X.values, list(X.columns))
    return dict(model="OrderedModel (logit)", beta=b, OR=float(np.exp(b)),
                lo=float(np.exp(lo)), hi=float(np.exp(hi)),
                p=float(res.pvalues["grade"]), N=int(res.nobs),
                extra=_fmt_merge(note),
                nclass=int(len(np.unique(y_codes))), brant=br,
                dir_ok=int((p_hi[-1] > p_lo[-1])), bse=float(res.bse["grade"]),
                topclass_note=f"{p_lo[-1]:.3f}→{p_hi[-1]:.3f}")

main_rows, brant_rows = [], []
for col, cn, en, typ in NUM_OUT:
    r = fit_main(d1026, col, typ)
    main_rows.append(dict(结局=cn, 结局英文=en, 类型=typ, 模型=r["model"],
                          β=r["beta"], OR=r["OR"], OR下限=r["lo"], OR上限=r["hi"],
                          p=r["p"], N=r["N"], 类别合并=r["extra"],
                          方向校验=r.get("topclass_note", "")))
    if r.get("brant"):
        br = r["brant"]
        W, df, pv = br["overall"]
        brant_rows.append(dict(结局=cn, 模型="主模型（调整后）", W=W, df=df, p=pv,
                               判定="通过（不拒绝平行性）" if pv > .05 else "⚠ 违反，需部分比例优势或降为二分"))
        for nm, (w, dfi, pv_i) in br["per"].items():
            brant_rows.append(dict(结局=cn, 模型=f"  逐变量：{nm}", W=w, df=dfi, p=pv_i,
                                   判定="通过" if pv_i > .05 else "⚠ 违反"))
main_t = pd.DataFrame(main_rows)
brant_t = pd.DataFrame(brant_rows) if brant_rows else pd.DataFrame()
sheets["表10_主模型"] = main_t
if len(brant_t):
    sheets["表10_Brant"] = brant_t

# ---- Brant 违反的处置 ----
# statsmodels 没有部分比例优势模型（R 的 ordinal::clm nominal=  / VGAM 的 parallel=FALSE），
# 自己写。核心：P(Y<=j) = logistic(a_j - Xpar·β - Xfree·γ_j)
#   β 对所有切点共用 → grade 走这里，仍然只出一个 OR，森林图刻度不破
#   γ_j 逐切点变化   → 只放给 Brant 判定违反的协变量（本例是 male，不是 grade）
# 这样既回应了审稿人，又不牺牲主效应的可读性。
def fit_ppo(y, Xpar, Xfree, maxiter=4000):
    """部分比例优势有序 logit。Xpar 平行（共享 β），Xfree 非平行（逐切点 γ_j）。"""
    y = np.asarray(y, dtype=int)
    J = int(np.unique(y).size)
    K = J - 1                      # 累积切点数
    n1, n2 = Xpar.shape[1], Xfree.shape[1]
    ar = np.arange(len(y))

    def split(p):
        return p[:K], p[K:K + n1], p[K + n1:].reshape(K, n2)

    def nll(p):
        a, b, g = split(p)
        eta = Xpar @ b
        F = np.empty((len(y), K))
        for j in range(K):
            F[:, j] = expit(a[j] - eta - Xfree @ g[j])
        hi = np.where(y >= K, 1.0, F[ar, np.minimum(y, K - 1)])
        lo = np.where(y <= 0, 0.0, F[ar, np.maximum(y - 1, 0)])
        pr = np.clip(hi - lo, 1e-12, 1.0)
        if not np.all(np.isfinite(pr)) or np.any(pr <= 0):
            return 1e10
        return float(-np.log(pr).sum())

    # 初值：截距取经验累积 logit，β 与 γ 置 0（不依赖 OrderedModel 的参数排列顺序）
    cum = np.array([(y <= j).mean() for j in range(K)])
    a0 = np.log(np.clip(cum, 1e-3, 1 - 1e-3) / (1 - np.clip(cum, 1e-3, 1 - 1e-3)))
    p0 = np.concatenate([a0, np.zeros(n1), np.zeros(K * n2)])

    best = None
    for meth in ("bfgs", "nm", "powell", "bfgs"):
        try:
            r = minimize(nll, p0 if best is None else best.x, method=meth,
                         options={"maxiter": maxiter})
            if np.isfinite(r.fun) and (best is None or r.fun < best.fun - 1e-8):
                best = r
        except Exception:
            continue
    H = approx_hess(best.x, nll)
    V = np.linalg.pinv(H)
    se = np.sqrt(np.maximum(np.diag(V), 0.0))
    a, b, g = split(best.x)
    return dict(a=a, b=b, g=g, se_b=se[K:K + n1], nll=float(best.fun),
                nclass=J, se_all=se)

log("[Step 10] Brant 违反的处置：部分比例优势 + 二分降级 …")
remedy_rows = []
for col, cn, en, typ in NUM_OUT:
    if typ != "ord5":
        continue
    dd = d1026[[col, "gnum", "major", "male", "net", "batch"]].dropna().copy()
    Xr = cov_matrix(dd)
    Xr.insert(0, "grade", dd["gnum"].values.astype(float))
    y_raw = dd[col].values
    y_codes, note, _ = collapse_sparse(y_raw, min_n=40)
    br = brant_test(y_codes, Xr.values, list(Xr.columns))
    pv = br["overall"][2]
    viol = [nm for nm in br["per"] if br["per"][nm][2] <= 0.05]
    if pv > 0.05:
        remedy_rows.append(dict(结局=cn, 处置="Brant 通过，不需处置", 模型="—", OR=np.nan,
                                OR下限=np.nan, OR上限=np.nan, p=pv, N=len(dd),
                                说明=f"总量 p={pv:.4f}"))
        continue
    free = [c for c in viol if c in Xr.columns and c != "grade"]
    if "grade" in viol:
        remedy_rows.append(dict(结局=cn, 处置="⚠ grade 自身违反平行线", 模型="—", OR=np.nan,
                                OR下限=np.nan, OR上限=np.nan, p=np.nan, N=len(dd),
                                说明="单一 OR 不成立，必须逐切点报 OR，不能进图 7 森林图"))
    if free:
        par = [c for c in Xr.columns if c not in free]
        rp = fit_ppo(y_codes, Xr[par].values.astype(float), Xr[free].values.astype(float))
        # 完全平行模型的对数似然，用于 LR 检验"放开是否值得"
        r_po = OrderedModel(y_codes, Xr.values.astype(float), distr="logit").fit(method="bfgs", disp=0)
        # ⚠ 符号：r_po.llf 是**对数似然（负）**，rp["nll"] 是**负对数似然（正）**，
        #   直接相减会把 LR 算成负的四位数。统一到对数似然尺度再比。
        llf_po, llf_ppo = float(r_po.llf), -rp["nll"]
        lr = 2 * (llf_ppo - llf_po)
        lr_df = (rp["nclass"] - 2) * len(free)
        gmax = float(np.max(np.abs(rp["g"])))
        gi = par.index("grade")
        b, sb = float(rp["b"][gi]), float(rp["se_b"][gi])
        remedy_rows.append(dict(
            结局=cn, 处置="① 部分比例优势（grade 仍平行）", 模型="PPO logit",
            OR=float(np.exp(b)), OR下限=float(np.exp(b - Z975 * sb)),
            OR上限=float(np.exp(b + Z975 * sb)),
            p=float(2 * stats.norm.sf(abs(b / sb))), N=len(dd),
            说明=f"放开非平行：{'、'.join(free)}（max|γ|={gmax:.3f}）；"
                 f"LL: PO={llf_po:.2f} → PPO={llf_ppo:.2f}；"
                 f"LR={lr:.2f}, df={lr_df}, p={float(stats.chi2.sf(lr, lr_df)):.4f}"))
    yb = (y_raw >= 4).astype(int)
    if 0 < yb.mean() < 1:
        rl = sm.Logit(yb, sm.add_constant(Xr)).fit(disp=0)
        bb, sb2 = float(rl.params["grade"]), float(rl.bse["grade"])
        remedy_rows.append(dict(
            结局=cn, 处置=f"② 降为二分（{col}≥4 vs ≤3）", 模型="二项 logit",
            OR=float(np.exp(bb)), OR下限=float(np.exp(bb - Z975 * sb2)),
            OR上限=float(np.exp(bb + Z975 * sb2)), p=float(rl.pvalues["grade"]),
            N=int(rl.nobs),
            说明=f"阳性 {yb.mean() * 100:.1f}%（n={int(yb.sum())}）；二分下比例优势自动成立"))
remedy_t = pd.DataFrame(remedy_rows)
if len(remedy_t):
    sheets["表10_Brant处置"] = remedy_t

# ---- 统一 OR：三个准连续结局做五分位箱，切点固定在 1026 ----
log("[Step 10] 统一 OR：准连续结局五分位分箱（切点固定自 1026）…")
QTILES = [0.2, 0.4, 0.6, 0.8]
bin_cut = {}
for col, cn, en, typ in NUM_OUT:
    if typ == "cont":
        bin_cut[col] = list(np.quantile(d1026[col].dropna().values, QTILES))
    elif typ == "ord5":
        yc, note, _ = collapse_sparse(d1026[col].values, min_n=40)
        # 记录 1026 上的合并方案，并在另一份数据上重跑同一规则（用 min_n 阈值，非切点）
        bin_cut[col] = None
for d in (d1026, d1579):
    for col, cn, en, typ in NUM_OUT:
        if typ == "nominal":
            continue
        if typ == "cont":
            cuts = bin_cut[col]
            d[f"q_{col}"] = np.searchsorted(cuts, d[col].values, side="right")
        else:
            yc, note, _ = collapse_sparse(d[col].fillna(d[col].median()).values, min_n=40)
            d[f"q_{col}"] = yc
unified_rows = []
for col, cn, en, typ in NUM_OUT:
    for tag, d in [("1026", d1026), ("1579", d1579)]:
        r = fit_main(d, f"q_{col}", "ord5", adjust=True)
        unified_rows.append(dict(结局=cn, 结局英文=en, 样本=tag,
                                 OR=r["OR"], lo=r["lo"], hi=r["hi"], p=r["p"], N=r["N"]))
uni_t = pd.DataFrame(unified_rows)
sheets["表_统一OR五分位"] = uni_t

# ---- 意向4类：多项 logistic + AME ----
log("[Step 10] 意向4类：多项 logistic + AME …")

def mnlogit_coef(res, var_pos, k):
    """MNLogit 的 params/bse/pvalues 都是 (变量 × 类别)，不是 (类别 × 变量)。
    且 conf_int() 返回 MultiIndex 展平的 ((K-1)*P, 2)，既容易看错又在不同
    statsmodels 版本间不一致。这里直接用 bse 算 Wald CI，绕开结构歧义。"""
    b = float(res.params.values[var_pos, k])
    se = float(res.bse.values[var_pos, k])
    return b, b - Z975 * se, b + Z975 * se, float(res.pvalues.values[var_pos, k])

# ⚠ statsmodels 的 MNLogit **没有 base 参数**，参照组固定为编码后 y=0 的那一类
#   （源码里 cov_names 用 ynames[1:]，见 discrete_model.MNLogit.__init__）。
#   因此必须把参照组放在列表第一位，否则整张 RRR 表会相对错误的类别。
CATS_FIT = [INTENT_REF, "深造", "求稳体制", "方向未定"]   # 首位 = 参照组（直接就业）
CATS_DISP = ["深造", INTENT_REF, "求稳体制", "方向未定"]  # 报告展示顺序
CATS = CATS_FIT

def fit_mnlogit(d, exclude_undecided=False, bootstrap=False, seed=SEED):
    dd = d[d["意向4类"].isin(CATS)].copy()
    if exclude_undecided:
        dd = dd[dd["意向4类"] != "方向未定"].copy()
    cats_use = [c for c in CATS if c in dd["意向4类"].unique()]
    mp = {i: c for i, c in enumerate(cats_use)}
    dd["y"] = dd["意向4类"].map({c: i for i, c in enumerate(cats_use)})
    dd = dd.dropna(subset=["y"])
    X = cov_matrix(dd)
    X = pd.concat([pd.Series(dd["gnum"].values.astype(float), index=dd.index, name="grade"), X], axis=1)
    Xc = sm.add_constant(X)
    try:
        res = sm.MNLogit(dd["y"].values.astype(int), Xc).fit(disp=0, maxiter=400)
    except Exception:
        res = sm.MNLogit(dd["y"].values.astype(int), Xc).fit(disp=0, maxiter=800, method="bfgs")
    return res, dd, Xc, mp

res_mn, dd_mn, Xc_mn, mp_mn = fit_mnlogit(d1026)
K = len(mp_mn)
grade_i = list(Xc_mn.columns).index("grade")
p0 = res_mn.params.values              # shape (K-1, p)

ame_rows, rrr_rows = [], []
Xbase = Xc_mn.copy()
Xhi = Xbase.copy()
Xhi["grade"] = Xbase["grade"] + 1
pr0 = res_mn.predict(exog=Xbase).values
pr1 = res_mn.predict(exog=Xhi).values
for k in range(K):
    cat = mp_mn[k]
    ame = float(np.mean(pr1[:, k] - pr0[:, k]))
    if k == 0:                                  # 参照组
        rrr_rows.append(dict(类别=cat, RRR=1.0, lo=np.nan, hi=np.nan,
                             p=np.nan, 参照=True))
    else:                                       # column k-1 才是第 k 类
        b, blo, bhi, pv_i = mnlogit_coef(res_mn, grade_i, k - 1)
        rrr_rows.append(dict(类别=cat, RRR=float(np.exp(b)),
                             lo=float(np.exp(blo)), hi=float(np.exp(bhi)),
                             p=pv_i, 参照=False))
    ame_rows.append(dict(类别=cat, AME=ame))

# AME bootstrap CI
rng = np.random.default_rng(SEED)
boot = {mp_mn[k]: [] for k in range(K)}
y_all = dd_mn["y"].values.astype(int)
n_mn = len(dd_mn)
for b in range(N_BOOT_AME):
    idx = rng.integers(0, n_mn, n_mn)
    yb = y_all[idx]
    Xb = Xc_mn.iloc[idx].reset_index(drop=True)
    if len(np.unique(yb)) < K:
        continue
    try:
        rb = sm.MNLogit(yb, Xb).fit(disp=0, maxiter=300)
    except Exception:
        continue
    pa = rb.predict(exog=Xb).values
    Xb2 = Xb.copy()
    Xb2["grade"] = Xb["grade"] + 1
    pb = rb.predict(exog=Xb2).values
    for k in range(K):
        boot[mp_mn[k]].append(float(np.mean(pb[:, k] - pa[:, k])))
for r in ame_rows:
    v = np.array(boot[r["类别"]])
    if len(v) > 30:
        r["lo"] = float(np.percentile(v, 2.5))
        r["hi"] = float(np.percentile(v, 97.5))
    else:
        r["lo"] = r["hi"] = np.nan
ame_t = pd.DataFrame(ame_rows)
rrr_t = pd.DataFrame(rrr_rows)
rrr_t["参照组"] = INTENT_REF
sheets["表11_RRR"] = rrr_t
sheets["表11_AME"] = ame_t

# ======================================================================
# 7. Step 11 · 五套稳健性
# ======================================================================
log("[Step 11] 五套稳健性 …")
rob_rows = []

VERSIONS = [
    ("① 粗估计（不调整协变量）", "crude", "gnum", False, d1026),
    ("② 主模型（调整专业/性别/来源）", "main", "gnum", True, d1026),
    ("③ 合并毕业年级（四档）", "grad3", "gnum3", True, d1026),
    ("④ 工学子集（调整后）", "eng", "gnum", True, d1026[d1026["major"] == "eng"]),
    ("⑤ 1579 合并样本（含 batch）", "merged", "gnum", True, d1579),
    ("⑥ 工学内 · 1579（调整后）", "eng1579", "gnum", True, d1579[d1579["major"] == "eng"]),
]

for label, key, gcol, adj, dsub in VERSIONS:
    for col, cn, en, typ in NUM_OUT:
        if typ == "nominal":
            continue
        r = fit_main(dsub, col, typ, gcol=gcol, adjust=adj)
        eff = r["OR"] if not np.isnan(r["OR"]) else r["beta"]
        lo = r["lo"] if not np.isnan(r["lo"]) else np.nan
        hi = r["hi"] if not np.isnan(r["hi"]) else np.nan
        rob_rows.append(dict(版本=label, 版本key=key, 结局=cn, 结局英文=en, 类型=typ,
                             效应量="OR" if not np.isnan(r["OR"]) else "β",
                             估计=eff, 下限=lo, 上限=hi, p=r["p"], N=r["N"]))
rob_t = pd.DataFrame(rob_rows)
sheets["表_稳健性_连续结局"] = rob_t

# 意向4类的稳健性
rob_intent = []
for label, key, gcol, adj, dsub in VERSIONS:
    dd = dsub[dsub["意向4类"].isin(CATS)].copy()
    cats_use = [c for c in CATS if c in dd["意向4类"].unique()]
    if len(cats_use) < 3:
        continue
    dd["y"] = dd["意向4类"].map({c: i for i, c in enumerate(cats_use)})
    dd = dd.dropna(subset=["y"])
    X = cov_matrix(dd) if adj else pd.DataFrame(index=dd.index)
    X.columns = [c for c in X.columns]
    X.insert(0, "grade", dd[gcol].values.astype(float))
    Xc = sm.add_constant(X)
    try:
        rr = sm.MNLogit(dd["y"].values.astype(int), Xc).fit(disp=0, maxiter=400)
    except Exception:
        continue
    gi = list(Xc.columns).index("grade")
    for k in range(1, len(cats_use)):
        b, blo, bhi, pv_i = mnlogit_coef(rr, gi, k - 1)
        rob_intent.append(dict(版本=label, 类别=cats_use[k], RRR=float(np.exp(b)),
                               lo=float(np.exp(blo)), hi=float(np.exp(bhi)),
                               p=pv_i, N=int(rr.nobs)))
# ⑦ 卡点 6：剔除方向未定
_d41 = d1026[d1026["意向4类"] != "方向未定"].copy()
_d41 = _d41[_d41["意向4类"].isin(["深造", "直接就业", "求稳体制"])]
# ⚠ MNLogit 的参照组固定是编码后 y=0 的那一类，所以必须把"直接就业"放在 0
# （初版写成 深造=0，导致 ⑦ 这一版的参照组与 ② 主模型不一致，RRR 不可比）
_d41["y"] = _d41["意向4类"].map({"直接就业": 0, "深造": 1, "求稳体制": 2})
X = cov_matrix(_d41)
X.insert(0, "grade", _d41["gnum"].values.astype(float))
Xc = sm.add_constant(X)
r41 = sm.MNLogit(_d41["y"].values.astype(int), Xc).fit(disp=0, maxiter=400)
gi = list(Xc.columns).index("grade")
for k in range(2):
    b, blo, bhi, pv_i = mnlogit_coef(r41, gi, k)
    rob_intent.append(dict(版本="⑦ 剔除方向未定（卡点 6 敏感性）", 类别=["深造", "求稳体制"][k],
                           RRR=float(np.exp(b)), lo=float(np.exp(blo)), hi=float(np.exp(bhi)),
                           p=pv_i, N=int(r41.nobs)))
rob_intent_t = pd.DataFrame(rob_intent)
sheets["表_稳健性_意向"] = rob_intent_t
log("  稳健性完成")

# ======================================================================
# 8. 图
# ======================================================================
log("[图 5] 六结局五档趋势 …")
fig, axes = plt.subplots(2, 3, figsize=(6.85, 4.5), facecolor="white")
panel_char = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)"]
zinfo = {}
for i, (col, cn, en, typ) in enumerate(NUM_OUT):
    ax = axes.flat[i]
    t = trend_curves[cn]
    x = np.arange(len(t))
    ax.errorbar(x, t["均值"], yerr=[t["均值"] - t["CI下限"], t["CI上限"] - t["均值"]],
                fmt="o-", color="#c00000", ecolor="#c00000", elinewidth=0.9,
                capsize=3, markersize=4, linewidth=1.1)
    z = (t["均值"] - t["均值"].mean()) / t["均值"].std(ddof=1) if t["均值"].std(ddof=1) > 0 else t["均值"] * 0
    zinfo[cn] = (x, z.values, en)
    ax.set_xticks(x)
    ax.set_xticklabels([GRADE_SHORT[g] for g in t["年级"]], fontsize=8.5)
    ax.tick_params(axis="y", labelsize=8)
    _rng = float(t["CI上限"].max() - t["CI下限"].min())
    ax.set_ylim(float(t["CI下限"].min()) - _rng * 0.25,
                float(t["CI上限"].max()) + _rng * 0.35)
    ax.grid(axis="y", ls=":", lw=0.6, color="#bfbfbf")
    ax.set_axisbelow(True)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.text(0.02, 1.02, f"{panel_char[i]} {en}", transform=ax.transAxes,
            fontsize=8.5, fontweight="bold", va="bottom")
    ax.set_ylabel("Mean (95% CI)" if i % 3 == 0 else "", fontsize=8)
# (f) 标准化叠加
ax = axes.flat[5]
cmap = plt.cm.tab10
for j, (cn, (x, zv, en)) in enumerate(zinfo.items()):
    ax.plot(x, zv, "-o", markersize=3.5, linewidth=1.2,
            color=cmap(j / max(1, len(zinfo) - 1)), label=en)
ax.axhline(0, color="#bfbfbf", lw=0.8)
ax.set_xticks(x)
ax.set_xticklabels([GRADE_SHORT[g] for g in GRADE_ORDER], fontsize=8.5)
ax.tick_params(axis="y", labelsize=8)
ax.grid(axis="y", ls=":", lw=0.6, color="#bfbfbf")
ax.set_axisbelow(True)
for s in ["top", "right"]:
    ax.spines[s].set_visible(False)
ax.text(0.02, 1.02, f"(f) Standardised composites", transform=ax.transAxes,
        fontsize=8.5, fontweight="bold", va="bottom")
_h, _l = axes.flat[5].get_legend_handles_labels()
axes.flat[5].legend_.remove() if axes.flat[5].legend_ else None
fig.legend(_h, _l, fontsize=7.5, frameon=False, ncol=5,
           loc="lower center", bbox_to_anchor=(0.5, 0.005))
fig.tight_layout(h_pad=2.0, w_pad=1.6, rect=[0, 0.075, 1, 1])
fig.savefig(os.path.join(FIG, "fig_RQ2_trend.png"), dpi=DPI, facecolor="white")
plt.close(fig)

log("[图 6] 意向4类堆叠比例条 …")
fig, ax = plt.subplots(figsize=(6.85, 2.9), facecolor="white")
COLORS4 = {"深造": "#1f4e79", "直接就业": "#c00000", "求稳体制": "#7f7f7f", "方向未定": "#bfbfbf"}
EN4 = {"深造": "Further study", "直接就业": "Direct employment",
       "求稳体制": "Public sector", "方向未定": "Undecided"}
bottom = np.zeros(len(GRADE_ORDER))
for cat in ["深造", "直接就业", "求稳体制", "方向未定"]:
    vals = intent_tab[cat].values
    ax.bar(GRADE_ORDER, vals, bottom=bottom, color=COLORS4[cat],
           label=EN4[cat], width=0.62, edgecolor="white", linewidth=0.6)
    for xi, (v, b_) in enumerate(zip(vals, bottom)):
        if v > 0.045:
            ax.text(xi, b_ + v / 2, f"{100*v:.1f}%", ha="center", va="center",
                    fontsize=8, color="white")
    bottom = bottom + vals
for xi, g in enumerate(GRADE_ORDER):
    ax.text(xi, 1.01, f"n={int(intent_cnt.loc[g].sum())}", ha="center",
            fontsize=8, va="bottom")
ax.set_xticks(np.arange(len(GRADE_ORDER)))
ax.set_xticklabels([GRADE_EN[g] for g in GRADE_ORDER], fontsize=9)
ax.set_ylim(0, 1.12)
ax.set_yticks(np.arange(0, 1.01, 0.25))
ax.set_yticklabels([f"{int(v*100)}%" for v in np.arange(0, 1.01, 0.25)], fontsize=9)
ax.set_ylabel("Share of respondents", fontsize=9)
for s in ["top", "right"]:
    ax.spines[s].set_visible(False)
ax.legend(fontsize=8, frameon=False, ncol=4, loc="upper center",
          bbox_to_anchor=(0.5, -0.14))
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig_RQ2_intention.png"), dpi=DPI, facecolor="white")
plt.close(fig)

log("[图 7] OR 森林图（1026 vs 1579，统一五分位 OR）…")
fig, ax = plt.subplots(figsize=(6.85, 3.4), facecolor="white")
uni_p = uni_t.copy()
order_names = [cn for _, cn, _, _ in NUM_OUT]
yy = np.arange(len(order_names))
off = {"1026": -0.18, "1579": 0.18}
COL = {"1026": "#c00000", "1579": "#1f4e79"}
for tag in ["1026", "1579"]:
    xs, los, his, ys = [], [], [], []
    for i, cn in enumerate(order_names):
        rows = uni_p[(uni_p["结局"] == cn) & (uni_p["样本"] == tag)]
        if len(rows) == 0:
            continue
        r = rows.iloc[0]
        xs.append(r["OR"]); los.append(r["lo"]); his.append(r["hi"])
        ys.append(i + off[tag])
    ax.errorbar(xs, ys, xerr=[np.array(xs) - np.array(los), np.array(his) - np.array(xs)],
                fmt="o", color=COL[tag], ecolor=COL[tag], elinewidth=1.0,
                capsize=3, markersize=5, linestyle="none",
                label=f"{'Main sample (N=1,026)' if tag=='1026' else 'Pooled (N=1,579)'}")
ax.axvline(1, color="#404040", ls="--", lw=0.9)
ax.set_yticks(yy)
ax.set_yticklabels([uni_p[uni_p["结局"] == cn]["结局英文"].iloc[0] for cn in order_names],
                   fontsize=9)
ax.set_xscale("log")
_ticks = [0.6, 0.8, 1.0, 1.25, 1.6, 2.0]
ax.set_xticks(_ticks)
ax.set_xticklabels([str(t) for t in _ticks], fontsize=9)
ax.set_xlabel("Adjusted OR per one grade step (ordinal logit on quintiles)", fontsize=9)
ax.legend(fontsize=8.5, frameon=False, loc="lower left",
          bbox_to_anchor=(0.0, -0.02))
ax.grid(axis="x", ls=":", lw=0.6, color="#bfbfbf")
ax.set_axisbelow(True)
for s in ["top", "right", "left"]:
    ax.spines[s].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig_RQ2_forest.png"), dpi=DPI, facecolor="white")
plt.close(fig)

log("[图 8] 正交多项式形状诊断：线性 vs 二次对比系数 …")
fig, axes = plt.subplots(1, 2, figsize=(6.85, 2.8), facecolor="white")
_p = poly_t.copy()
_p["pL_BH_plot"] = bh_adjust(_p["p线性"].values)
_p["pQ_BH_plot"] = bh_adjust(_p["p二次"].values)
_p = _p.sort_values("b线性")
_pen = [_o[2] for _o in NUM_OUT]
for _ax, _bcol, _pcol, _pc, _lab in [
        (axes[0], "b线性", "pL_BH_plot", "(a)", "Linear contrast (b_L)"),
        (axes[1], "b二次", "pQ_BH_plot", "(b)", "Quadratic contrast (b_Q)")]:
    y = np.arange(len(_p))
    b = _p[_bcol].values
    se = _p["se线性" if _bcol == "b线性" else "se二次"].values
    err = 1.959964 * se
    pv = _p[_pcol].values
    cols = ["#c00000" if (p < .05 and v > 0) else "#1f4e79" if (p < .05 and v < 0)
            else "#bfbfbf" for p, v in zip(pv, b)]
    _ax.barh(y, b, color=cols, alpha=0.85, height=0.6)
    _ax.errorbar(b, y, xerr=err, fmt="none", ecolor="#404040",
                 elinewidth=0.8, capsize=2)
    _ax.axvline(0, color="#404040", lw=0.8)
    _ax.set_yticks(y)
    _ax.set_yticklabels([_p.iloc[i]["结局英文"] for i in range(len(_p))], fontsize=8)
    _ax.tick_params(axis="x", labelsize=8)
    _ax.set_xlabel(_lab, fontsize=8.5)
    _ax.grid(axis="x", ls=":", lw=0.6, color="#bfbfbf")
    _ax.set_axisbelow(True)
    for s in ["top", "right"]:
        _ax.spines[s].set_visible(False)
    _ax.text(0.02, 1.02, _pc, transform=_ax.transAxes,
             fontsize=9, fontweight="bold", va="bottom")
fig.tight_layout(w_pad=2.4)
fig.savefig(os.path.join(FIG, "fig_RQ2_shape.png"), dpi=DPI, facecolor="white")
plt.close(fig)

# ======================================================================
# 9. Excel + 报告
# ======================================================================
log("[输出] 写 Excel …")
with pd.ExcelWriter(XLSX, engine="openpyxl") as w:
    if sheets:
        for k, v in sheets.items():
            v.to_excel(w, sheet_name=k[:31], index=False)

def f2(x, d=3):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"{x:.{d}f}"

def stars(p):
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return "—"
    return "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else ""

log("[输出] 写报告 …")

for _, r in tV.iterrows():
    pass
for g in GRADE_ORDER:
    v = tab.loc[g].values
for g in GRADE_ORDER:
    n1, n2 = int(eng_tab.loc[g, "1026"]), int(eng_tab.loc[g, "1579"])
    j1 = f"仅定性（n={n1}）" if n1 < 40 else "可推断"
    j2 = f"仅定性（n={n2}）" if n2 < 40 else "可推断"

for cn in [o[1] for o in NUM_OUT]:
    t = trend_curves[cn]
    en = [o[2] for o in NUM_OUT if o[1] == cn][0]
    for _, r in t.iterrows():
        pass
for _, r in jt_t.iterrows():
    direction = ("↑ 递增" if r["tau_b"] > 0 else "↓ 递减") if r["p_BH"] < .05 else "—"
for _, r in intent_jt.iterrows():
    pass
for g in GRADE_ORDER:
    v = intent_tab.loc[g].values

for _, r in poly_t.iterrows():
    pass
n_curve = int((poly_t["p二次_BH"] < .05).sum())
if n_curve > 0:
    _cc = poly_t.loc[poly_t["p二次_BH"] < .05, "结局"].tolist()
else:
    pass

for _, r in main_t.iterrows():
    est = f"OR {f2(r['OR'])}" if not pd.isna(r["OR"]) else f"β {f2(r['β'],4)}"
    ci = f"{f2(r['OR下限'],3)}–{f2(r['OR上限'],3)}"
if len(brant_t):
    for _, r in brant_t.iterrows():
        pass
    _viol = brant_t[(brant_t["模型"] == "主模型（调整后）") & (brant_t["p"] < .05)]

if len(remedy_t):
    for _, r in remedy_t.iterrows():
        _or = "—" if not np.isfinite(r["OR"]) else f"{r['OR']:.3f}"
        _ci = "—" if not np.isfinite(r["OR下限"]) else f"{r['OR下限']:.3f}–{r['OR上限']:.3f}"
        _p = "—" if not np.isfinite(r["p"]) else f"{r['p']:.4f}"

for cn in order_names:
    r1 = uni_t[(uni_t["结局"] == cn) & (uni_t["样本"] == "1026")]
    r2 = uni_t[(uni_t["结局"] == cn) & (uni_t["样本"] == "1579")]
    def _cell(r):
        if len(r) == 0:
            return "—"
        r = r.iloc[0]
        return f"{f2(r['OR'],3)} ({f2(r['lo'],3)}–{f2(r['hi'],3)}){stars(r['p'])}"

for _cat in CATS_DISP:                       # 按展示顺序，参照组也列出来
    _rr = rrr_t[rrr_t["类别"] == _cat]
    _am = ame_t[ame_t["类别"] == _cat]
    if len(_rr) == 0:
        continue
    _rr, _am = _rr.iloc[0], (_am.iloc[0] if len(_am) else None)
    ame_s = f2(_am["AME"], 4) if _am is not None else "—"
    ame_ci = (f"{f2(_am['lo'],4)}–{f2(_am['hi'],4)}"
              if _am is not None and not pd.isna(_am.get("lo", np.nan)) else "—")

for cn in [o[1] for o in NUM_OUT]:
    cells = []
    for label, key, *_ in VERSIONS:
        r = rob_t[(rob_t["结局"] == cn) & (rob_t["版本key"] == key)]
        if len(r) == 0:
            cells.append("—")
            continue
        r = r.iloc[0]
        cells.append(f"{f2(r['估计'],3)}{stars(r['p'])}<br>({f2(r['下限'],3)}, {f2(r['上限'],3)})")
for _, r in rob_intent_t.iterrows():
    pass
_n_und = int((d1026["意向4类"] == "方向未定").sum())
_r_main = rob_intent_t[rob_intent_t["版本"] == "② 主模型（调整专业/性别/来源）"]
_r_exc = rob_intent_t[rob_intent_t["版本"].str.startswith("⑦")]
if len(_r_main) and len(_r_exc):
    _sens = []
    for cat in ["深造", "求稳体制"]:
        a = _r_main[_r_main["类别"] == cat]
        b = _r_exc[_r_exc["类别"] == cat]
        if len(a) and len(b):
            la, ha, ra = np.log(a.iloc[0]["lo"]), np.log(a.iloc[0]["hi"]), np.log(a.iloc[0]["RRR"])
            rb = np.log(b.iloc[0]["RRR"])
            se = (ha - la) / (2 * Z975)
            dl = abs(rb - ra)
            flip = (ra > 0) != (rb > 0)
            ok = (dl <= 0.5 * se) and not flip
            _sens.append((cat, ok, flip))
    if _sens and not all(s[1] for s in _sens):
        pass
    elif _sens:
        pass

for c in _chk:
    pass

os.makedirs(OUT, exist_ok=True)

log("Done.")
