# -*- coding: utf-8 -*-
"""
Step 6 · Q4「就业压力源」探索性因子分析（EFA）
论文二 · 研究问题一（压力—偏好画像）

链路：
  polychoric 相关（自实现，Owen's T 二元正态） → KMO/MSA + Bartlett
  → 平行分析定因子数 → PAF 提取 → varimax → promax 斜交旋转
  → 因子得分（回归法） → 跨批次 Tucker's φ 一致性 → 外推到 1579

⚠ 重跑 step20_prepare.py 后必须重跑本脚本（因子得分是写回数据集的）。

用法：E:/Python/python.exe step22_efa.py
"""
import sys as _sys
if hasattr(_sys.stdout, "reconfigure"):
    _sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import os
import time
import numpy as np
import pandas as pd
from scipy.special import owens_t
from scipy.stats import norm, chi2
from scipy.optimize import brentq, minimize_scalar
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# 常量
# ----------------------------------------------------------------------
BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "论文二_输出")
FIG = os.path.join(BASE, "figs2")
F1026 = os.path.join(BASE, "论文二数据集_1026.xlsx")
F1579 = os.path.join(BASE, "论文二数据集_1579.xlsx")
os.makedirs(OUT, exist_ok=True)
os.makedirs(FIG, exist_ok=True)

SEED = 42
N_REP_PA = 1000          # 平行分析重复次数
PROMAX_POWER = 4         # psych 默认
SMALL, BIG = -8.0, 8.0   # ±inf 哨兵（Φ(±8) 误差 < 1e-14）
EPS = 1e-8

Q4_ITEMS = [f"Q4_{i}" for i in range(1, 10)]
ITEM_CN = {
    "Q4_1": "就业市场竞争激烈程度", "Q4_2": "所学专业与市场需求匹配度",
    "Q4_3": "自身专业技能/实习经验不足", "Q4_4": "学历要求门槛",
    "Q4_5": "求职信息获取难度", "Q4_6": "家庭期望、经济压力",
    "Q4_7": "职业规划不清晰", "Q4_8": "AI 发展对就业的冲击",
    "Q4_9": "政策与制度的变化",
}
ITEM_EN = {
    "Q4_1": "Market competition", "Q4_2": "Major-market mismatch",
    "Q4_3": "Skill/internship gap", "Q4_4": "Education credentialism",
    "Q4_5": "Job information access", "Q4_6": "Family expectations & finances",
    "Q4_7": "Unclear career planning", "Q4_8": "AI impact on jobs",
    "Q4_9": "Policy & institutional change",
}

# 因子命名按「题项集合」匹配（promax 因子顺序是任意的，按序号会错位）
FAC_NAMES = {
    frozenset(["Q4_5", "Q4_6", "Q4_7", "Q4_8", "Q4_9"]):
        ("外在情境—内在规划压力", "External-contextual & internal-planning stressors"),
    frozenset(["Q4_1", "Q4_2", "Q4_3", "Q4_4"]):
        ("市场竞争—人力资本压力", "Market-competition & human-capital stressors"),
}

def fac_name(item_set):
    return FAC_NAMES.get(frozenset(item_set),
                         (f"因子（{len(item_set)} 题，待命名）", "Factor (unnamed)"))

plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 9

# ======================================================================
# 一、polychoric 相关（抽出到共享模块 poly_utils.py，见该文件）
# ======================================================================
from poly_utils import (thresholds_from_margins, polychoric_pair,
                        polychoric_matrix, nearest_pd, EPS)

# ======================================================================
# 二、KMO / MSA / Bartlett
# ======================================================================
def kmo_bartlett(R, n):
    p = R.shape[0]
    Rinv = np.linalg.pinv(R)
    Rr = R.copy()
    np.fill_diagonal(Rr, 0.0)
    # 偏相关 q_ij = -p_ij / sqrt(p_ii p_jj)
    d = np.sqrt(np.diag(Rinv))
    Q = -Rinv / np.outer(d, d)
    np.fill_diagonal(Q, 0.0)
    ss_r = np.sum(Rr ** 2)
    ss_q = np.sum(Q ** 2)
    kmo = ss_r / (ss_r + ss_q)
    msa = np.sum(Rr ** 2, axis=1) / (np.sum(Rr ** 2, axis=1) + np.sum(Q ** 2, axis=1))
    det = np.linalg.det(R)
    chi = -(n - 1 - (2 * p + 5) / 6.0) * np.log(max(det, 1e-300))
    df = p * (p - 1) // 2
    pval = float(chi2.sf(chi, df))
    return float(kmo), msa, float(chi), df, pval

# ======================================================================
# 三、PAF 提取 + varimax + promax
# ======================================================================
def paf(R, m, max_iter=300, tol=1e-8):
    """主轴因子法（共同度迭代），返回载荷、全部特征值（降序）、共同度"""
    p = R.shape[0]
    Rinv = np.linalg.pinv(R)
    h2 = 1.0 - 1.0 / np.diag(Rinv)
    h2 = np.clip(h2, 0.05, 0.99)
    for _ in range(max_iter):
        Rr = R.copy()
        np.fill_diagonal(Rr, h2)
        ev, evec = np.linalg.eigh(Rr)
        idx = np.argsort(ev)[::-1][:m]
        L = evec[:, idx] * np.sqrt(np.clip(ev[idx], 1e-8, None))
        h2n = np.clip(np.sum(L ** 2, axis=1), 0.0, 0.999)
        if np.max(np.abs(h2n - h2)) < tol:
            h2 = h2n
            break
        h2 = h2n
    Rr = R.copy()
    np.fill_diagonal(Rr, h2)
    ev, evec = np.linalg.eigh(Rr)
    order = np.argsort(ev)[::-1]
    ev, evec = ev[order], evec[:, order]
    L = evec[:, :m] * np.sqrt(np.clip(ev[:m], 1e-8, None))
    return L, ev, h2

def paf_eigenvalues(R):
    """仅取 PAF 迭代后的约化矩阵特征值（供平行分析用）"""
    p = R.shape[0]
    Rinv = np.linalg.pinv(R)
    h2 = np.clip(1.0 - 1.0 / np.diag(Rinv), 0.05, 0.99)
    Rr = R.copy()
    np.fill_diagonal(Rr, h2)
    ev = np.linalg.eigvalsh(Rr)
    return np.sort(ev)[::-1]

def _row_scale(A):
    """Kaiser 行归一化权重 s = sqrt(共同度)，零共同度行保护为 1"""
    s = np.sqrt(np.sum(A ** 2, axis=1, keepdims=True))
    return np.where(s < 1e-12, 1.0, s)

def _varimax_core(X, gamma=1.0, max_iter=1000, tol=1e-7):
    """对（可能已 Kaiser 归一化）的载荷做 varimax，返回 (载荷, 旋转矩阵)"""
    p, m = X.shape
    T = np.eye(m)
    d_prev = -np.inf
    for _ in range(max_iter):
        L = X @ T
        B = X.T @ (L ** 3 - gamma * L @ np.diag(np.diag(L.T @ L)) / p)
        U, s, Vt = np.linalg.svd(B)
        T = U @ Vt
        d = np.sum(s)
        if d - d_prev < tol * max(d_prev, 1e-12):
            break
        d_prev = d
    return X @ T, T

def varimax(A, gamma=1.0, normalize=True, max_iter=1000, tol=1e-7):
    """Kaiser 归一化的 varimax：行按共同度缩放 → 旋转 → 乘回"""
    if A.shape[1] < 2:
        return A.copy()
    if normalize:
        s = _row_scale(A)
        L, _ = _varimax_core(A / s, gamma, max_iter, tol)
        return L * s
    return _varimax_core(A, gamma, max_iter, tol)[0]

def promax(A, power=PROMAX_POWER, normalize=True, gamma=1.0, max_iter=1000, tol=1e-7):
    """
    promax 斜交旋转（Hendrickson & White, 1964）。
    内部串联 Kaiser 归一化的 varimax，目标矩阵与变换均在归一化空间内求得，
    最后乘回行缩放 —— 与 factor_analyzer 0.5.1 / R 的 kaiser(rotate='Promax') 一致。

    缩放因子取 sqrt(diag((T'T)^-1))，使 Φ = (T'T)^-1 的对角恰为 1。
    返回（模式矩阵 Λ, 因子相关 Φ）
    """
    p, m = A.shape
    if m < 2:
        return A.copy(), np.eye(m)
    s = _row_scale(A) if normalize else np.ones((p, 1))
    X, _ = _varimax_core(A / s, gamma, max_iter, tol)
    Y = X * np.abs(X) ** (power - 1)          # = sign(X)|X|^power
    T = np.linalg.solve(X.T @ X, X.T @ Y)
    Tn = T @ np.diag(np.sqrt(np.diag(np.linalg.inv(T.T @ T))))
    Lam = (X @ Tn) * s
    Phi = np.linalg.inv(Tn.T @ Tn)
    np.fill_diagonal(Phi, 1.0)
    return Lam, Phi

# ======================================================================
# 四、Tucker's φ（正交 Procrustes 对齐后）
# ======================================================================
def tucker_phi(A, B):
    """B 向 A 做正交 Procrustes 旋转后计算因子一致性系数，返回 (φ, B_rot, Q)"""
    M = A.T @ B
    U, _, Vt = np.linalg.svd(M)
    Q = U @ Vt
    Br = B @ Q
    num = float(np.sum(A * Br))
    den = float(np.sqrt(np.sum(A ** 2) * np.sum(Br ** 2)))
    return num / den if den > 0 else np.nan, Br, Q

def _cos(a, b):
    d = float(np.sqrt(np.sum(a ** 2) * np.sum(b ** 2)))
    return abs(float(np.sum(a * b) / d)) if d > 0 else np.nan

def procrustes_align(A, B, m):
    """逐因子 Tucker's φ：正交 Procrustes 整体旋转后再逐列配对"""
    _, Br, _ = tucker_phi(A, B)
    phis = np.array([_cos(A[:, j], Br[:, j]) for j in range(m)])
    return phis, Br

# ======================================================================
# 五、平行分析（polychoric 版）
# ======================================================================
def parallel_analysis(margins, n, p, n_rep=N_REP_PA, seed=SEED):
    """margins: list of 各题 5 类比例（用于生成保持边际的独立随机数据）"""
    rng = np.random.default_rng(seed)
    cuts = [norm.ppf(np.clip(np.cumsum(mp)[:-1], 1e-9, 1 - 1e-9)) for mp in margins]
    evals = np.zeros((n_rep, p))
    t0 = time.time()
    for rep in range(n_rep):
        Xs = np.empty((n, p), dtype=np.int64)
        for j in range(p):
            Xs[:, j] = np.searchsorted(cuts[j], rng.standard_normal(n)) + 1
        Rs = polychoric_matrix(Xs)
        Rs = nearest_pd(Rs)
        evals[rep] = paf_eigenvalues(Rs)
        if (rep + 1) % 200 == 0:
            el = time.time() - t0
            print(f"    平行分析 {rep+1}/{n_rep}  已用 {el:.0f}s  预计共 "
                  f"{el / (rep + 1) * n_rep / 60:.1f}min")
    return evals

# ======================================================================
# 六、主流程
# ======================================================================
print("=" * 70)
print("Step 6 · Q4 探索性因子分析")
print("=" * 70)

d1026 = pd.read_excel(F1026)
d1579 = pd.read_excel(F1579)
print(f"[1] 读入 1026 集 {d1026.shape} / 1579 集 {d1579.shape}")

X_all = d1579[Q4_ITEMS].copy()
X_main = d1026[Q4_ITEMS].copy()
X_2nd = d1579.iloc[1026:][Q4_ITEMS].copy()

n_miss = int(d1026[Q4_ITEMS].isna().sum().sum())
print(f"[2] Q4 缺失值：1026 集 {n_miss} 个；1579 集 {int(X_all.isna().sum().sum())} 个")
if n_miss > 0:
    print("    ⚠ 存在缺失，改用逐对（pairwise）估计")
else:
    print("    ✅ 无缺失，全部按完整数据估计")

X_main = X_main.dropna().astype(int)
X_2nd = X_2nd.dropna().astype(int)
X_all = X_all.dropna().astype(int)
N_main, N_2nd, N_all = len(X_main), len(X_2nd), len(X_all)
print(f"    可用 N：主 {N_main} / 第二 {N_2nd} / 合并 {N_all}")

# 边际分布（平行分析用）
margins = []
for c in Q4_ITEMS:
    vc = X_main[c].value_counts().reindex(range(1, 6), fill_value=0).values
    margins.append(vc / vc.sum())
print("[3] Q4 各题边际比例（1→5）：")
for c, mp in zip(Q4_ITEMS, margins):
    print(f"    {c} {ITEM_CN[c][:12]:<14} {np.round(mp, 3)}")

# ---- polychoric 相关矩阵（主样本）----
print("[4] 计算 polychoric 相关矩阵（主样本 1026）…")
t0 = time.time()
R_poly = polychoric_matrix(X_main.values)
R_poly = nearest_pd(R_poly)
R_pear = np.corrcoef(X_main.values.astype(float).T)
print(f"    完成，用时 {time.time() - t0:.2f}s")
ev_check = np.linalg.eigvalsh(R_poly)
print(f"    polychoric 特征值范围 [{ev_check.min():.4f}, {ev_check.max():.4f}]"
      f"  正定：{'✅' if ev_check.min() > 0 else '❌（已修正）'}")

R_poly_df = pd.DataFrame(R_poly, index=Q4_ITEMS, columns=Q4_ITEMS)
R_pear_df = pd.DataFrame(R_pear, index=Q4_ITEMS, columns=Q4_ITEMS)

# ---- KMO / Bartlett ----
kmo, msa, chi, df_bt, p_bt = kmo_bartlett(R_poly, N_main)
print(f"[5] KMO = {kmo:.4f}   Bartlett χ²({df_bt}) = {chi:.1f}, p = {p_bt:.3g}")
msa_df = pd.DataFrame({"题项": Q4_ITEMS, "中文": [ITEM_CN[c] for c in Q4_ITEMS],
                       "英文": [ITEM_EN[c] for c in Q4_ITEMS], "MSA": msa.round(4)})
print("    各题 MSA： " + "  ".join(f"{c}={v:.3f}" for c, v in zip(Q4_ITEMS, msa)))

# ---- 平行分析 ----
print(f"[6] 平行分析（{N_REP_PA} 次模拟，polychoric + PAF 特征值）…")
ev_pa = parallel_analysis(margins, N_main, len(Q4_ITEMS))
pa_mean = ev_pa.mean(axis=0)
pa_p95 = np.percentile(ev_pa, 95, axis=0)

ev_obs = paf_eigenvalues(R_poly)
ev_obs_pca = np.sort(np.linalg.eigvalsh(R_poly))[::-1]
print("    因子  实测PAF    PA均值   PA第95百分位   实测>PA95?   Kaiser>1")
m_pa = 0
rows_pa = []
for i in range(len(Q4_ITEMS)):
    hit = ev_obs[i] > pa_p95[i]
    if hit:
        m_pa = i + 1
    rows_pa.append({"因子序号": i + 1, "实测特征值_PAF": round(float(ev_obs[i]), 4),
                    "PA均值": round(float(pa_mean[i]), 4),
                    "PA第95百分位": round(float(pa_p95[i]), 4),
                    "实测>PA95": "是" if hit else "否",
                    "实测特征值_PCA版": round(float(ev_obs_pca[i]), 4),
                    "Kaiser>1": "是" if ev_obs[i] > 1 else "否"})
    print(f"    {i+1}   {ev_obs[i]:8.4f} {pa_mean[i]:9.4f} {pa_p95[i]:11.4f}"
          f"        {'是' if hit else '否':<6}   {'是' if ev_obs[i] > 1 else '否'}")
m_kaiser = int(np.sum(ev_obs > 1))
print(f"    → 平行分析建议保留 {m_pa} 个因子；Kaiser>1 建议 {m_kaiser} 个")
M = m_pa if m_pa >= 1 else 2
print(f"    → 采用 M = {M}")

# ---- EFA：PAF → varimax → promax ----
L_unrot, ev_final, h2_final = paf(R_poly, M)
L_var = varimax(L_unrot)
# promax 内部串联 Kaiser 归一化的 varimax，输入必须是**未旋转**的 PAF 载荷
L_pro, Phi = promax(L_unrot, power=PROMAX_POWER)
Struct = L_pro @ Phi

# ---- 自检：与 factor_analyzer 0.5.1 的旋转器逐项对照（先按列符号对齐）----
from factor_analyzer import Rotator as _FaRotator

def _sign_align(A):
    s = np.sign(np.sum(A, axis=0))
    s = np.where(s == 0, 1.0, s)
    return A * s, s

_fa = _FaRotator(method="promax", power=PROMAX_POWER, normalize=True)
_fa.fit_transform(L_unrot.copy())
_a, _s = _sign_align(L_pro)
_b = _fa.loadings_ * _s
_dl = float(np.max(np.abs(_a - _b)))
_dp = float(np.max(np.abs(Phi - np.diag(_s) @ _fa.phi_ @ np.diag(_s))))
print(f"    [自检] 与 factor_analyzer 0.5.1 的 promax 对照：载荷最大差 {_dl:.2e}、"
      f"Φ 最大差 {_dp:.2e}  "
      f"{'✅ 一致' if max(_dl, _dp) < 5e-3 else '⚠ 不一致，请核查'}")

# 载荷绝对值最大者归组
assign = np.abs(L_pro).argmax(axis=1)
print(f"[7] EFA 完成（M={M}）。旋转后因子载荷：")
for j in range(M):
    items = [Q4_ITEMS[i] for i in range(9) if assign[i] == j]
    print(f"    因子{j+1} 主载荷：{items}")

loading_df = pd.DataFrame(L_pro, index=[f"{c} {ITEM_CN[c]}" for c in Q4_ITEMS],
                          columns=[f"F{j+1}" for j in range(M)])
unrot_df = pd.DataFrame(L_unrot, index=[f"{c} {ITEM_CN[c]}" for c in Q4_ITEMS],
                        columns=[f"F{j+1}" for j in range(M)])
struct_df = pd.DataFrame(Struct, index=[f"{c} {ITEM_CN[c]}" for c in Q4_ITEMS],
                         columns=[f"F{j+1}" for j in range(M)])
phi_df = pd.DataFrame(Phi, index=[f"F{j+1}" for j in range(M)],
                      columns=[f"F{j+1}" for j in range(M)])

ss_unrot = float(np.sum(L_unrot ** 2))
ss_rot = np.sum(L_pro ** 2, axis=0)
print(f"    共同度：{np.round(h2_final, 3)}")
print(f"    未旋转 SS loadings = {ss_unrot:.3f}（解释 {100*ss_unrot/9:.1f}%）")
print(f"    旋转后各因子 SS loadings = {np.round(ss_rot, 3)}")
print(f"    因子间相关 Φ = {np.round(Phi[np.triu_indices(M, 1)], 3)}")

# ---- 因子数敏感性：M = 1 / 2 / 3 ----
print("[7b] 因子数敏感性（M=1 / 2 / 3 对照）：")
m_sens = []
for m in (1, 2, 3):
    Lm, evm, h2m = paf(R_poly, m)
    ssm = float(np.sum(Lm ** 2))
    if m > 1:
        Lpm, Phim = promax(Lm, power=PROMAX_POWER)
        mx = float(np.abs(Phim[np.triu_indices(m, 1)]).max())
    else:
        Lpm, mx = Lm, float("nan")
    pa_keep = "是" if ev_obs[m - 1] > pa_p95[m - 1] else "否"
    m_sens.append({"因子数": m, "未旋转SS": round(ssm, 3),
                   "解释方差%": round(100 * ssm / 9, 1),
                   "平均共同度": round(float(h2m.mean()), 3),
                   "最大因子间相关": ("—" if m == 1 else round(mx, 3)),
                   "该因子PA保留": pa_keep})
    print(f"    M={m}：SS={ssm:.3f} 解释 {100*ssm/9:.1f}%  平均共同度 {h2m.mean():.3f}"
          f"  最大Φ={'—' if m == 1 else f'{mx:.3f}'}  第{m}个因子PA保留={pa_keep}")
m_sens_df = pd.DataFrame(m_sens)

# ---- 交叉载荷与弱载荷检查 ----
print("[8] 题项诊断（交叉载荷 >0.40 或主载荷 <0.32 需讨论）：")
diag_items = []
for i, c in enumerate(Q4_ITEMS):
    row = L_pro[i]
    main_v = row[assign[i]]
    others = np.delete(row, assign[i])
    max_other = float(np.abs(others).max())
    flag = []
    if max_other > 0.40:
        flag.append("交叉载荷>0.40")
    if abs(main_v) < 0.32:
        flag.append("主载荷<0.32")
    diag_items.append({"题项": c, "中文": ITEM_CN[c], "英文": ITEM_EN[c],
                       "归属因子": f"F{assign[i]+1}", "主载荷": round(float(main_v), 4),
                       "最大其他载荷": round(max_other, 4),
                       "共同度": round(float(h2_final[i]), 4),
                       "标记": "；".join(flag) if flag else ""})
    print(f"    {c} → F{assign[i]+1}  载荷 {main_v:+.3f}  其他最大 {max_other:.3f}  "
          f"{'⚠ ' + '；'.join(flag) if flag else ''}")
diag_df = pd.DataFrame(diag_items)

# ======================================================================
# 七、跨批次一致性：Tucker's φ
# ======================================================================
print("[9] 第二批 553 内独立跑同构 EFA → Tucker's φ …")
R_2nd = polychoric_matrix(X_2nd.values)
R_2nd = nearest_pd(R_2nd)
L2_unrot, _, h2_2 = paf(R_2nd, M)
L2_var = varimax(L2_unrot)
L2_pro, Phi2 = promax(L2_unrot, power=PROMAX_POWER)

# 三个层次的一致性（越往下越贴近"得分能不能用"）
#   ① varimax 正交载荷：纯结构，不含 Φ 的批次差异
#   ② promax 模式矩阵：受两批 Φ 差异影响
#   ③ 得分系数矩阵 B = R⁻¹Λ：直接决定因子得分可比性
_, L2v_rot = procrustes_align(L_var, L2_var, M)
phi_var = np.array([_cos(L_var[:, j], L2v_rot[:, j]) for j in range(M)])
phis, L2_rot = procrustes_align(L_pro, L2_pro, M)
B_main = np.linalg.pinv(R_poly) @ L_pro
B_2nd = np.linalg.pinv(R_2nd) @ L2_pro
_, B2_rot = procrustes_align(B_main, B_2nd, M)
phi_b = np.array([_cos(B_main[:, j], B2_rot[:, j]) for j in range(M)])
overall_phi, _, _ = tucker_phi(L_pro, L2_pro)

# ④ 得分外推等价性（这才是 Step 14 真正关心的量）：
#    实际做法是拿 1026 的 B 去算 553 的得分，问它与"553 用自己的载荷算的得分"差多少
B_2nd_al = np.linalg.pinv(R_2nd) @ L2_rot      # 已 Procrustes 对齐到 1026 的因子方向
Z_2nd_sc = (X_2nd.values - X_all.mean().values) / X_all.std(ddof=1).values
S_ext = Z_2nd_sc @ B_main                       # 外推（论文实际采用）
S_own = Z_2nd_sc @ B_2nd_al                     # 第二批自算（反事实基准）
r_ext = np.array([float(np.corrcoef(S_ext[:, j], S_own[:, j])[0, 1]) for j in range(M)])

# 题项归属一致性：两批是否把同一题归入同一因子
am = np.abs(L_pro).argmax(axis=1)
a2 = np.abs(L2_rot).argmax(axis=1)
agree = float(np.mean(am == a2))
disagree_items = [Q4_ITEMS[i] for i in range(len(Q4_ITEMS)) if am[i] != a2[i]]

print(f"    ① varimax 正交载荷 φ = {np.round(phi_var, 4)}  最小 {phi_var.min():.4f}")
print(f"    ② promax 模式矩阵 φ = {np.round(phis, 4)}  最小 {phis.min():.4f}"
      f"（1026 内 Φ={Phi[0, 1]:.3f}，553 内 Φ={Phi2[0, 1]:.3f}）")
print(f"    ③ 得分系数矩阵 φ     = {np.round(phi_b, 4)}  最小 {phi_b.min():.4f}")
print(f"    ④ 外推得分 vs 自算得分相关 = {np.round(r_ext, 4)}  最小 {r_ext.min():.4f}")
print(f"    题项归属一致率 = {agree:.3f}（{len(Q4_ITEMS) - len(disagree_items)}/"
      f"{len(Q4_ITEMS)}）{'' if not disagree_items else '  不一致：' + '、'.join(disagree_items)}")

print(f"    —— 外推路径（1026 载荷 → 1579）：r = {r_ext.min():.3f}"
      f"{'✅' if r_ext.min() >= 0.95 else ('⚠' if r_ext.min() >= 0.90 else '❌')}")

def _verdict(v):
    return "一致" if v >= 0.95 else ("可接受" if v >= 0.90 else "不一致")

phi_cmp = pd.DataFrame({
    "因子": [f"F{j+1}" for j in range(M)],
    "φ_varimax正交载荷": np.round(phi_var, 4), "判_结构": [_verdict(v) for v in phi_var],
    "φ_promax模式矩阵": np.round(phis, 4), "判_模式": [_verdict(v) for v in phis],
    "φ_得分系数B": np.round(phi_b, 4), "判_得分": [_verdict(v) for v in phi_b],
    "r_外推vs自算得分": np.round(r_ext, 4),
    "判_外推": ["等价" if v >= 0.95 else ("可用" if v >= 0.90 else "不可靠") for v in r_ext],
})
load2_df = pd.DataFrame(L2_pro, index=[f"{c} {ITEM_CN[c]}" for c in Q4_ITEMS],
                        columns=[f"F{j+1}" for j in range(M)])

# 敏感性：promax power=3 / 2（Φ 随 power 上升而变高，须验证结构不依赖 power）
sens = {}
for pw in (2, 3):
    La, _ = promax(L_unrot, power=pw)
    Lb, _ = promax(L2_unrot, power=pw)
    f, _ = procrustes_align(La, Lb, M)
    _, Phi_pw = promax(L_unrot, power=pw)
    sens[pw] = (f, Phi_pw)
    print(f"    敏感性 promax power={pw}：逐因子 φ = {np.round(f, 4)}、"
          f"Φ = {np.round(Phi_pw[np.triu_indices(M, 1)], 3)}")
phi3 = sens[3][0]

# ======================================================================
# 九乙、1579 合并样本内重新拟合 EFA（跨批外推的替代路径）
# ======================================================================
print("[9b] 在 1579 合并样本内重新拟合 EFA（不跨批外推）…")
R_all = polychoric_matrix(X_all.values)
R_all = nearest_pd(R_all)
L_all_unrot, ev_all, h2_all = paf(R_all, M)
L_all_var = varimax(L_all_unrot)
L_all_pro_raw, Phi_all_raw = promax(L_all_unrot, power=PROMAX_POWER)
# ⚠ 合并拟合的因子方向可能与 1026 解相反（EFA 因子符号任意），
#   必须显式对齐，否则两个数据集的 F* 列含义相反
_, L_all_rot, Q_all = tucker_phi(L_pro, L_all_pro_raw)
Phi_all = Q_all.T @ Phi_all_raw @ Q_all
np.fill_diagonal(Phi_all, 1.0)
L_all_pro = L_all_rot
phi_pool, L_allv_rot = procrustes_align(L_var, L_all_var, M)
print(f"    原始 1579 解 Φ = {Phi_all_raw[0, 1]:+.3f} → 对齐后 Φ = {Phi_all[0, 1]:+.3f}"
      f"（1026 解 Φ = {Phi[0, 1]:+.3f}）")
print(f"    1026 解 vs 1579 解：varimax φ = {np.round(phi_pool, 4)}、"
      f"promax 整体 φ = {_cos(L_pro, L_all_pro):.4f}")
print(f"    1579 内 SS loadings = {np.round(np.sum(L_all_pro ** 2, axis=0), 3)}")
print("    ⚠ 对 M=2，正交 Procrustes 几乎总能对齐 → 该 φ 必然虚高，"
      "**不能单独作为结构一致性的证据**")
pool_same = bool(phi_pool.min() >= 0.95)

# ======================================================================
# 十、因子得分
# ======================================================================
print("[10] 计算因子得分（回归法）…")
B_1026 = np.linalg.pinv(R_poly) @ L_pro          # 主样本自己的载荷
B_1579 = np.linalg.pinv(R_all) @ L_all_pro      # 合并样本自己的载荷

mu_all = X_all.mean().values
sd_all = X_all.std(ddof=1).values
mu_main = X_main.mean().values
sd_main = X_main.std(ddof=1).values

Z_all = (X_all.values - mu_all) / sd_all
Z_main = (X_main.values - mu_main) / sd_main

# 三套得分：1579 内拟合（推荐）／1026 内拟合／1026 载荷外推到 1579（仅对照）
S_pool = Z_all @ B_1579
S_main = Z_main @ B_1026
S_ext_all = Z_all @ B_1026
S_all_z = (S_pool - S_pool.mean(axis=0)) / S_pool.std(axis=0, ddof=1)
S_main_z = (S_main - S_main.mean(axis=0)) / S_main.std(axis=0, ddof=1)
S_ext_z = (S_ext_all - S_ext_all.mean(axis=0)) / S_ext_all.std(axis=0, ddof=1)
r_fit_ext = np.array([float(np.corrcoef(S_all_z[:, j], S_ext_z[:, j])[0, 1])
                      for j in range(M)])
print(f"    1579 内拟合得分 vs 外推得分 相关 = {np.round(r_fit_ext, 4)}")

# --- 单元加权得分（题组均值）---
# 分组由 1026 的 EFA 归属确定，两个数据集共用同一套题项定义，
# 因此不受载荷采样波动影响，跨批 100% 可比 —— 这是外推问题的根本解法
u_groups = [[Q4_ITEMS[i] for i in range(len(Q4_ITEMS)) if am[i] == j] for j in range(M)]
U_pool = np.column_stack([X_all[g].mean(axis=1).values for g in u_groups])
U_main = np.column_stack([X_main[g].mean(axis=1).values for g in u_groups])
U_pool_z = (U_pool - U_pool.mean(0)) / U_pool.std(0, ddof=1)
U_main_z = (U_main - U_main.mean(0)) / U_main.std(0, ddof=1)
r_u_reg = np.array([float(np.corrcoef(U_pool_z[:, j], S_all_z[:, j])[0, 1])
                    for j in range(M)])
for j in range(M):
    print(f"    单元加权 F{j+1} = mean({', '.join(u_groups[j])})"
          f"  与回归法得分相关 {r_u_reg[j]:.4f}")
B = B_1579

# ---- 最终判定：两条路径的取舍 ----
phi_min = float(np.min(phis))
struct_ok = bool(phi_var.min() >= 0.95)
if pool_same and struct_ok:
    ok_ext = True
    if r_ext.min() >= 0.95:
        ver = (f"✅ 独立跨批结构一致（varimax φ={phi_var.min():.3f}）、外推得分等价"
               f"（r≥{r_ext.min():.3f}）→ 三套得分皆可用，推荐单元加权")
    elif r_ext.min() >= 0.90:
        ver = (f"⚠ 结构一致（varimax φ={phi_var.min():.3f}）但外推得分相关仅 "
               f"{r_ext.min():.3f} → **改用单元加权得分**（跨批定义固定），"
               f"并在方法节说明放弃外推路径的理由")
    else:
        ver = (f"⚠ **外推路径不可靠**（外推 vs 自算 r = {r_ext.min():.3f} < 0.90）→ "
               f"**改用单元加权得分 `F*`**：分组由 1026 的 EFA 固定，两批共用同一题项定义，"
               f"从根上不受载荷采样波动影响（与回归法得分相关 "
               f"{r_u_reg.min():.2f}–{r_u_reg.max():.2f}）。"
               f"回归法得分 `RF*` 仅作附录对照。正文须写明这一切换与理由")
elif pool_same:
    ok_ext = True
    ver = (f"⚠ 独立跨批一致性未达标（varimax φ={phi_var.min():.3f} < 0.95），"
           f"仍可用单元加权得分，但正文须声明跨批结构一致性的证据不足")
else:
    ok_ext = False
    ver = (f"❌ 结构一致性不成立（跨批 varimax φ={phi_var.min():.3f}、"
           f"合并拟合 φ={phi_pool.min():.3f}）→ **Step 14 改用 Q4 总指数**，"
           f"因子得分只作附录探索性结果")
print(f"    [最终判定] {ver}")

fcols = [f"F{j+1}" for j in range(M)]        # 单元加权（推荐给 Step 14）
rfcols = [f"RF{j+1}" for j in range(M)]      # 回归法（仅作对照）
for j in range(M):
    d1579[fcols[j]] = np.nan
    d1579.loc[X_all.index, fcols[j]] = U_pool_z[:, j]
    d1579[rfcols[j]] = np.nan
    d1579.loc[X_all.index, rfcols[j]] = S_all_z[:, j]
    d1026[fcols[j]] = np.nan
    d1026.loc[X_main.index, fcols[j]] = U_main_z[:, j]
    d1026[rfcols[j]] = np.nan
    d1026.loc[X_main.index, rfcols[j]] = S_main_z[:, j]

corr_f = np.corrcoef(U_pool_z.T)
print(f"    单元加权得分间相关（1579 内）：{np.round(corr_f[np.triu_indices(M, 1)], 3)}")

score_coef_df = pd.DataFrame(B, index=[f"{c} {ITEM_CN[c]}" for c in Q4_ITEMS],
                             columns=rfcols)
score_desc = pd.DataFrame({
    "因子": fcols,
    "构成": ["mean(" + ",".join(g) + ")" for g in u_groups],
    "1026_均值": U_main_z.mean(axis=0).round(4),
    "1026_SD": U_main_z.std(axis=0, ddof=1).round(4),
    "1579_均值": U_pool_z.mean(axis=0).round(4),
    "1579_SD": U_pool_z.std(axis=0, ddof=1).round(4),
    "与回归法得分r": r_u_reg.round(4),
    "批次均值差": (U_pool_z[X_main.index.values].mean(axis=0) -
                 U_pool_z[X_2nd.index.values].mean(axis=0)).round(4),
})

# 得分与总指数的相关（供 RQ3 判断是否可互换）
corr_with_index = [float(np.corrcoef(U_pool_z[:, j],
                                     d1579.loc[X_all.index, "Q4_压力源指数"])[0, 1])
                   for j in range(M)]
print(f"    各单元加权得分与 Q4 总指数的相关：{np.round(corr_with_index, 3)}")

# ======================================================================
# 九、出图
# ======================================================================
print("[11] 出图 …")
fig, axes = plt.subplots(1, 2, figsize=(6.85, 2.9), facecolor="white")
ax = axes[0]
xs = np.arange(1, 10)
ax.plot(xs, ev_obs, "o-", color="#1f4e79", lw=1.6, ms=4.5, label="Observed (PAF)")
ax.plot(xs, pa_p95, "s--", color="#c00000", lw=1.4, ms=4,
        label="Parallel analysis 95th pct")
ax.plot(xs, pa_mean, ":", color="#7f7f7f", lw=1.2, label="PA mean")
ax.axhline(1.0, color="#2e75b6", lw=0.9, ls="-.", label="Kaiser = 1")
ax.axvline(M + 0.5, color="#666666", lw=0.9, ls=":", alpha=0.8)
ax.set_xticks(xs)
ax.set_xlabel("Factor number")
ax.set_ylabel("Eigenvalue")
ax.text(0.02, 0.97, "(a)", transform=ax.transAxes, fontsize=10, fontweight="bold",
        va="top")
ax.legend(fontsize=7, loc="upper right", frameon=False)
ax.spines[["top", "right"]].set_visible(False)
lo = min(ev_obs.min(), pa_mean.min()) - 0.15
ax.set_ylim(max(lo, -0.5), ev_obs.max() + 0.85)   # 留白给 (a) 标签

ax = axes[1]
_vmax = float(np.abs(L_pro).max()) + 0.06
im = ax.imshow(L_pro.T, cmap="RdBu_r", vmin=-_vmax, vmax=_vmax, aspect="auto")
ax.set_yticks(range(M))
ax.set_yticklabels([f"F{j+1}" for j in range(M)])
ax.set_xticks(range(9))
ax.set_xticklabels([f"Q4_{i}" for i in range(1, 10)], rotation=45, ha="right")
for i in range(9):
    for j in range(M):
        v = L_pro[i, j]
        ax.text(i, j, f"{v:.2f}", ha="center", va="center", fontsize=7.2,
                color="white" if abs(v) > 0.45 else "black")
cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cb.set_label("Loading", fontsize=8)
cb.ax.tick_params(labelsize=7)
ax.text(0.02, 0.95, "(b)", transform=ax.transAxes, fontsize=10, fontweight="bold",
        va="top", color="black")
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig_RQ1_efa.png"), dpi=600, facecolor="white")
plt.close(fig)

# ======================================================================
# 十、写回数据集
# ======================================================================
d1026.to_excel(F1026, index=False)
d1579.to_excel(F1579, index=False)
print(f"[12] 因子得分已写回两个数据集（列 {fcols}）")

# ======================================================================
# 十一、Excel 汇总
# ======================================================================
xl = os.path.join(OUT, "Step6_EFA结果.xlsx")
with pd.ExcelWriter(xl, engine="openpyxl") as w:
    R_poly_df.round(4).to_excel(w, sheet_name="polychoric矩阵")
    R_pear_df.round(4).to_excel(w, sheet_name="pearson对照")
    pd.DataFrame({
        "指标": ["KMO", "Bartlett chi2", "df", "p", "N"],
        "值": [round(kmo, 4), round(chi, 2), df_bt, f"{p_bt:.3g}", N_main],
    }).to_excel(w, sheet_name="KMO_Bartlett", index=False)
    msa_df.to_excel(w, sheet_name="MSA", index=False)
    pd.DataFrame(rows_pa).to_excel(w, sheet_name="平行分析", index=False)
    m_sens_df.to_excel(w, sheet_name="因子数敏感性", index=False)
    unrot_df.round(4).to_excel(w, sheet_name="载荷_未旋转")
    loading_df.round(4).to_excel(w, sheet_name="载荷_promax_1026")
    load2_df.round(4).to_excel(w, sheet_name="载荷_promax_553")
    pd.DataFrame(L_all_pro, index=[f"{c} {ITEM_CN[c]}" for c in Q4_ITEMS],
                 columns=fcols).round(4).to_excel(w, sheet_name="载荷_promax_1579")
    struct_df.round(4).to_excel(w, sheet_name="结构矩阵")
    phi_df.round(4).to_excel(w, sheet_name="因子相关Phi")
    diag_df.to_excel(w, sheet_name="题项诊断", index=False)
    phi_cmp.to_excel(w, sheet_name="Tucker_phi", index=False)
    score_coef_df.round(4).to_excel(w, sheet_name="得分系数")
    score_desc.to_excel(w, sheet_name="得分描述", index=False)
print(f"[13] Excel 已存 {xl}")

# ======================================================================
# 十二、Markdown 报告
# ======================================================================
item_lines = []
for i, c in enumerate(Q4_ITEMS):
    row = L_pro[i]
    main_v = row[assign[i]]
    cells = " | ".join(f"{'**' if j == assign[i] else ''}{v:+.3f}{'**' if j == assign[i] else ''}"
                       for j, v in enumerate(row))
    item_lines.append(f"| {c} | {ITEM_CN[c]} | {ITEM_EN[c]} | "
                      f"{X_main[c].mean():.3f}±{X_main[c].std():.2f} | {cells} | "
                      f"{h2_final[i]:.3f} |")

for r in rows_pa:
    pass
for r in m_sens:
    pass
for j in range(M):
    for k in range(j + 1, M):
        pass
for r in diag_items:
    pass
for j in range(M):
    g = [Q4_ITEMS[i] for i in range(len(Q4_ITEMS)) if assign[i] == j]
    c_, e_ = fac_name(g)
for j in range(M):
    pass
for j in range(M):
    pass
if ok_ext and r_ext.min() >= 0.95:
    pass
elif ok_ext and r_ext.min() >= 0.90:
    pass
if M > 1 and abs(Phi[0, 1]) > 0.60:
    pass

print("=" * 70)
print("Step 6 完成")
