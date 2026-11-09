# -*- coding: utf-8 -*-
"""
Step 3–5 · 测量质量 + 量表描述 + 多重响应（论文二 · RQ1 前半段）
论文二 · 研究问题一：压力—偏好画像

覆盖手册三个 Step：
  Step 3  测量质量：Cronbach's α / KR-20 / McDonald's ω / Shapiro–Wilk
  Step 4  Q4、Q10 描述 + Kendall's W（并列秩校正）
  Step 5  Q6/Q8/Q9 多重响应（Wilson 95% CI）+ Q11 + Q7 双口径 + 性别/专业分层（BH）

产出：
  表 1 样本结构 ｜ 表 2 Q4 描述+α ｜ 表 3 Q10 描述+W ｜ 表 4 多重响应
  表 5 Q11 ｜ 表 6 Q7 双口径
  图 1 两榜 ｜ 图 2 多重响应棒棒糖 ｜ 图 3 Q7 双口径 + Q11

⚠ 依赖 poly_utils.polychoric_matrix（ω 用 polychoric 相关算）。
⚠ 本脚本只读 1026 主样本（RQ1 口径不合并）；不写回数据集。

用法：E:/Python/python.exe step21_descriptives.py
"""
import sys as _sys
if hasattr(_sys.stdout, "reconfigure"):
    _sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import os
import time
import numpy as np
import pandas as pd
from scipy.stats import (norm, chi2, shapiro, skew, kurtosis, mannwhitneyu,
                         kruskal, rankdata, chi2_contingency)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from poly_utils import polychoric_matrix, nearest_pd

T0 = time.time()

# ----------------------------------------------------------------------
# 常量
# ----------------------------------------------------------------------
BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "论文二_输出")
FIG = os.path.join(BASE, "figs2")
F1026 = os.path.join(BASE, "论文二数据集_1026.xlsx")
os.makedirs(OUT, exist_ok=True)
os.makedirs(FIG, exist_ok=True)

XL = os.path.join(OUT, "Step3_5_描述与测量质量.xlsx")

Q4_ITEMS = [f"Q4_{i}" for i in range(1, 10)]
Q6_ITEMS = [f"Q6_{i}" for i in range(1, 10)]
Q8_ITEMS = [f"Q8_{i}" for i in range(1, 10)]
Q9_ITEMS = [f"Q9_{i}" for i in range(1, 8)]
Q10_ITEMS = [f"Q10_{i}" for i in range(1, 9)]
Q11_ITEMS = [f"Q11_{i}" for i in range(1, 8)]

F1_ITEMS = ["Q4_5", "Q4_6", "Q4_7", "Q4_8", "Q4_9"]   # 由 Step 6 固定
F2_ITEMS = ["Q4_1", "Q4_2", "Q4_3", "Q4_4"]

Q4_CN = {
    "Q4_1": "就业市场竞争激烈程度", "Q4_2": "所学专业与市场需求匹配度",
    "Q4_3": "自身专业技能/实习经验不足", "Q4_4": "学历要求门槛",
    "Q4_5": "求职信息获取难度", "Q4_6": "家庭期望、经济压力",
    "Q4_7": "职业规划不清晰", "Q4_8": "AI 发展对就业的冲击",
    "Q4_9": "政策与制度的变化",
}
Q4_EN = {
    "Q4_1": "Market competition", "Q4_2": "Major-market mismatch",
    "Q4_3": "Skill/internship gap", "Q4_4": "Education credentialism",
    "Q4_5": "Job information access", "Q4_6": "Family expectations & finances",
    "Q4_7": "Unclear career planning", "Q4_8": "AI impact on jobs",
    "Q4_9": "Policy & institutional change",
}
Q6_CN = {"Q6_1": "专业能力优势", "Q6_2": "综合能力优势", "Q6_3": "价值观正确",
         "Q6_4": "实践经历优势", "Q6_5": "资源人脉优势", "Q6_6": "家庭支持",
         "Q6_7": "技能特长优势", "Q6_8": "职业目标清晰/自我定位明确", "Q6_9": "其他"}
Q6_EN = {"Q6_1": "Professional competence", "Q6_2": "Comprehensive competence",
         "Q6_3": "Sound values", "Q6_4": "Practical experience",
         "Q6_5": "Resources & networks", "Q6_6": "Family support",
         "Q6_7": "Technical skills", "Q6_8": "Clear career goals",
         "Q6_9": "Other"}
Q8_CN = {"Q8_1": "焦虑失眠、情绪低落", "Q8_2": "自我怀疑、否定自身能力",
         "Q8_3": "学习/求职动力下降", "Q8_4": "回避社交或就业相关话题",
         "Q8_5": "盲目跟风投递简历", "Q8_6": "无明显影响",
         "Q8_7": "降低对第一份工作的期望", "Q8_8": "考虑和专业不相关的工作",
         "Q8_9": "其他"}
Q8_EN = {"Q8_1": "Anxiety / low mood", "Q8_2": "Self-doubt",
         "Q8_3": "Reduced motivation", "Q8_4": "Avoiding job-related talk",
         "Q8_5": "Indiscriminate applying", "Q8_6": "No noticeable impact",
         "Q8_7": "Lowered expectations", "Q8_8": "Considering off-major jobs",
         "Q8_9": "Other"}
Q9_CN = {"Q9_1": "主动提升技能、积极寻求机会", "Q9_2": "降低期望，先就业再择业",
         "Q9_3": "转向考研/考公等延迟就业路径", "Q9_4": "寻求家人、朋友或专业指导",
         "Q9_5": "暂时回避，让自己缓一缓", "Q9_6": "跟随身边大多数人的选择",
         "Q9_7": "其他"}
Q9_EN = {"Q9_1": "Upskilling & active search", "Q9_2": "Lower expectations, job first",
         "Q9_3": "Deferral (postgrad / civil service)", "Q9_4": "Seeking guidance",
         "Q9_5": "Temporary avoidance", "Q9_6": "Following the majority",
         "Q9_7": "Other"}
Q10_CN = {"Q10_1": "国企/事业单位", "Q10_2": "外资企业", "Q10_3": "民营企业（大型）",
          "Q10_4": "民营企业（中小型）", "Q10_5": "自主创业", "Q10_6": "自由职业",
          "Q10_7": "新业态从业", "Q10_8": "基层就业"}
Q10_EN = {"Q10_1": "State-owned / public institution", "Q10_2": "Foreign-invested enterprise",
          "Q10_3": "Large private enterprise", "Q10_4": "Small & medium private enterprise",
          "Q10_5": "Entrepreneurship", "Q10_6": "Freelancing",
          "Q10_7": "New forms of employment", "Q10_8": "Grassroots employment"}
Q11_CN = {"Q11_1": "薪资待遇期望", "Q11_2": "工作地点时间期望", "Q11_3": "工作量预期",
          "Q11_4": "工作内容兴趣度", "Q11_5": "公司规模知名度预期",
          "Q11_6": "未来发展前景", "Q11_7": "工作氛围预期"}
Q11_EN = {"Q11_1": "Salary expectations", "Q11_2": "Location & working hours",
          "Q11_3": "Workload", "Q11_4": "Job-content interest",
          "Q11_5": "Employer size & reputation", "Q11_6": "Career prospects",
          "Q11_7": "Work atmosphere"}

SEX_CN = {"male": "男", "female": "女"}
MAJOR_CN = {"eng": "工学", "sci": "理学", "econ": "经管", "hum": "文史哲", "oth": "其他"}
INTENT_ORDER = ["深造", "直接就业", "求稳体制", "方向未定"]
INTENT_EN = {"深造": "Further study", "直接就业": "Direct employment",
             "求稳体制": "Public-sector stability", "方向未定": "Undecided"}

# ⚠ 基线按「题项名」对应，不能按降序位次对——论文一原文给的是降序排列，
#   而后四项的实际题序是 Q10_6 > Q10_7 > Q10_8 > Q10_5（按均值）
BASELINE_Q10 = {"Q10_1": 3.811, "Q10_2": 3.802, "Q10_3": 3.584, "Q10_4": 3.232,
                "Q10_5": 2.884, "Q10_6": 3.221, "Q10_7": 3.178, "Q10_8": 2.973}
BASELINE_Q4_INDEX = 3.661
BASELINE_Q5_HEAVY = 0.172

plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 9

Z95 = float(norm.ppf(0.975))


# ======================================================================
# 通用函数
# ======================================================================
def log(msg):
    print(f"[{time.time()-T0:6.1f}s] {msg}", flush=True)


def wilson_ci(k, n, z=Z95):
    """Wilson 得分区间（Agresti & Coull 1998）——小比例/大比例都稳"""
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, c - h), min(1.0, c + h))


def cronbach_alpha(X):
    """α（二分题时与 KR-20 代数等价）"""
    X = np.asarray(X, float)
    k = X.shape[1]
    if k < 2:
        return np.nan
    tot = X.sum(axis=1)
    ss_item = X.var(axis=0, ddof=1).sum()
    ss_tot = tot.var(ddof=1)
    return float(k / (k - 1) * (1 - ss_item / ss_tot))


def kr20(X):
    """Kuder–Richardson 20（0/1 计分专用），这里显式写成 p·q 形式以示区分"""
    X = np.asarray(X, float)
    k = X.shape[1]
    p = X.mean(axis=0)
    tot = X.sum(axis=1)
    ss_tot = tot.var(ddof=1)
    return float(k / (k - 1) * (1 - (p * (1 - p)).sum() / ss_tot))


def alpha_if_deleted(X):
    X = np.asarray(X, float)
    return np.array([cronbach_alpha(np.delete(X, j, axis=1))
                     for j in range(X.shape[1])])


def omega_total(R):
    """单因子模型的 McDonald's ω_total：ω = (Σλ)² / [(Σλ)² + Σ(1−λ²)]"""
    R = (R + R.T) / 2
    ev, evec = np.linalg.eigh(R)
    j = int(np.argmax(ev))
    lam = np.sqrt(max(ev[j], 0.0)) * evec[:, j]
    if lam.sum() < 0:
        lam = -lam
    uniq = np.clip(1 - lam ** 2, 0, None)
    num = lam.sum() ** 2
    return float(num / (num + uniq.sum())), lam


def mean_interitem_r(R):
    p = R.shape[0]
    iu = np.triu_indices(p, 1)
    return float(np.mean(R[iu]))


def kendall_w(X):
    """Kendall 和谐系数 W（并列秩校正版，Grothe & Schmid 2011）
    X: (m 个评判者) × (k 个被评对象)
    """
    X = np.asarray(X, float)
    m, k = X.shape
    R = np.apply_along_axis(rankdata, 1, X)      # 行内平均秩（处理并列）
    Rs = R.sum(axis=0)                            # 每个对象的秩和
    S = np.sum((Rs - m * (k + 1) / 2) ** 2)
    T = 0.0
    for row in R:
        _, cnt = np.unique(row, return_counts=True)
        T += np.sum(cnt ** 3 - cnt)
    denom = m * m * (k ** 3 - k) - m * T
    w = 12 * S / denom
    x2 = m * (k - 1) * w
    p = float(chi2.sf(x2, k - 1))
    return float(w), float(x2), p, float(T)


def bh_adjust(p):
    """Benjamini–Hochberg"""
    p = np.asarray(p, float)
    n = len(p)
    o = np.argsort(p)
    adj = p[o] * n / np.arange(1, n + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    out = np.empty(n)
    out[o] = np.clip(adj, 0, 1)
    return out


def epsilon_sq(h, k, n):
    """Kruskal–Wallis 的 ε²： (H − k + 1)/(n − k)"""
    return float((h - k + 1) / (n - k))


def fmt_p(p):
    if p < 1e-4:
        return "<0.0001"
    return f"{p:.4f}"


def pct(x):
    return f"{100*x:.1f}%"


# ======================================================================
# 读入
# ======================================================================
log("读入主样本 1026 …")
df = pd.read_excel(F1026)
N = len(df)
log(f"N = {N}，列数 {df.shape[1]}")

sheets = {}
notes = []

# 口径自检：确认两个偏好复合列的定义（论文一踩过 Q10_2 的坑）
_chk_inside = float(np.max(np.abs(df["Q10_体制内偏好"] - df["Q10_1"])))
_chk_flex = float(np.max(np.abs(df["Q10_灵活新就业偏好"]
                                - df[["Q10_5", "Q10_6", "Q10_7"]].mean(axis=1))))
log(f"口径自检 |Q10_体制内偏好 − Q10_1|max = {_chk_inside:.2e}；"
    f"|Q10_灵活偏好 − mean(Q10_5,6,7)|max = {_chk_flex:.2e}")
notes.append(
    f"口径自检：`Q10_体制内偏好` 与 `Q10_1` 逐行最大差 {_chk_inside:.1e}（→ 该列 = "
    f"**Q10_1 单题**，不是 mean(Q10_1,Q10_2)）；`Q10_灵活新就业偏好` 与 "
    f"mean(Q10_5,Q10_6,Q10_7) 最大差 {_chk_flex:.1e}（→ 三题均值）。")


# ======================================================================
# 表 1 · 样本结构
# ======================================================================
log("[表 1] 样本结构 …")
def freq_block(col, order=None, cn=None, en=None):
    vc = df[col].value_counts()
    if order:
        vc = vc.reindex(order)
    return pd.DataFrame({
        "类别": vc.index,
        "English": [en.get(v, "") for v in vc.index] if en else ["" for _ in vc.index],
        "n": vc.values,
        "%": np.round(vc.values / vc.sum() * 100, 1),
    })


t1_rows = []
for label, col, order, cndict, endict in [
    ("性别", "sex", ["male", "female"], SEX_CN,
     {"male": "Male", "female": "Female"}),
    ("年级", "grade", ["y1", "y2", "y3", "y4", "grad"],
     {"y1": "大一", "y2": "大二", "y3": "大三", "y4": "大四", "grad": "研究生"},
     {"y1": "Year 1", "y2": "Year 2", "y3": "Year 3", "y4": "Year 4",
      "grad": "Postgraduate"}),
    ("专业", "major", ["eng", "sci", "econ", "hum", "oth"], MAJOR_CN,
     {"eng": "Engineering", "sci": "Science", "econ": "Economics & management",
      "hum": "Humanities", "oth": "Other"}),
    ("渠道", "channel", None, {"paper": "纸质", "net": "网络"},
     {"paper": "Paper", "net": "Online"}),
    ("优先意向（单选口径）", "意向4类", INTENT_ORDER, {},
     {k: v for k, v in INTENT_EN.items()}),
]:
    b = freq_block(col, order, cndict, endict)
    b.insert(0, "变量", label)
    if not cndict:
        b.insert(2, "中文", [INTENT_EN and x for x in b["类别"]])
        b["中文"] = b["类别"].map(lambda x: x)
    else:
        b.insert(2, "中文", [cndict[v] for v in b["类别"]])
    t1_rows.append(b)
t1 = pd.concat(t1_rows, ignore_index=True)
sheets["表1_样本结构"] = t1


# ======================================================================
# Step 3 · 测量质量
# ======================================================================
log("[Step 3] 测量质量：α / KR-20 / ω / Shapiro–Wilk …")

# --- 3.1 α 与 ω（李克特量表题）---
rel_cols, rel_rows = [], []
def add_reliability(name, items, X, kind_desc):
    """X 已按完整个案处理"""
    a = cronbach_alpha(X)
    aid = alpha_if_deleted(X)
    R = nearest_pd(polychoric_matrix(X), tol=1e-9)
    om, lam = omega_total(R)
    mir = mean_interitem_r(R)
    rel_rows.append({
        "量表": name, "题数": len(items), "N": X.shape[0], "类型": kind_desc,
        "α": round(a, 4), "KR-20": round(kr20(X), 4) if np.unique(X).size <= 2 else np.nan,
        "ω_total": round(om, 4), "平均题间相关（polychoric）": round(mir, 4),
        "α 的 95% CI 下限": np.nan,
    })
    for j, it in enumerate(items):
        rel_cols.append({"量表": name, "题项": it, "α_if_deleted": round(aid[j], 4),
                         "载荷 λ（单因子）": round(float(lam[j]), 4)})
    return a, om

Q4_X = df[Q4_ITEMS].dropna().values
Q10_X = df[Q10_ITEMS].dropna().values
Q11_X = df[Q11_ITEMS].dropna().values          # 列表删除 → 七题全答者
Q11_N = Q11_X.shape[0]
F1_X = df[F1_ITEMS].dropna().values
F2_X = df[F2_ITEMS].dropna().values

a_q4, om_q4 = add_reliability("Q4 就业压力源", Q4_ITEMS, Q4_X, "5 点李克特（反映型）")
a_q10, om_q10 = add_reliability("Q10 工作类型偏好", Q10_ITEMS, Q10_X, "5 点李克特（反映型）")
a_q11, om_q11 = add_reliability("Q11 期望受影响", Q11_ITEMS, Q11_X, "3 点有序（合成型，见注）")
a_f1, om_f1 = add_reliability("F1 外在情境—内在规划（Step 6）", F1_ITEMS, F1_X, "5 点李克特")
a_f2, om_f2 = add_reliability("F2 市场竞争—人力资本（Step 6）", F2_ITEMS, F2_X, "5 点李克特")

# α 的 bootstrap CI（1000 次，行重抽）——回应"点估计不够"
rng = np.random.default_rng(42)
def alpha_ci(X, B=1000):
    n = X.shape[0]
    out = []
    for _ in range(B):
        idx = rng.integers(0, n, n)
        try:
            out.append(cronbach_alpha(X[idx]))
        except Exception:
            pass
    return np.nanpercentile(out, [2.5, 97.5])

for name, XX in [("Q4 就业压力源", Q4_X), ("Q10 工作类型偏好", Q10_X),
                 ("Q11 期望受影响", Q11_X)]:
    lo, hi = alpha_ci(XX)
    for r in rel_rows:
        if r["量表"] == name:
            r["α 的 95% CI 下限"] = round(lo, 4)
            r["α 的 95% CI 上限"] = round(hi, 4)
for r in rel_rows:
    r.setdefault("α 的 95% CI 上限", np.nan)
rel = pd.DataFrame(rel_rows)
sheets["信度_α_ω"] = rel
sheets["α_if_deleted"] = pd.DataFrame(rel_cols)

# --- 3.2 KR-20（二分多选项）---
kr_rows = []
for name, items, cnd, end in [("Q6 未来发展优势", Q6_ITEMS, Q6_CN, Q6_EN),
                              ("Q8 心理/行为表现", Q8_ITEMS, Q8_CN, Q8_EN),
                              ("Q9 应对方式", Q9_ITEMS, Q9_CN, Q9_EN)]:
    Xb = df[items].dropna().values.astype(float)
    a = cronbach_alpha(Xb)
    k20 = kr20(Xb)
    Rp = nearest_pd(polychoric_matrix(Xb), tol=1e-9)
    om, _ = omega_total(Rp)
    kr_rows.append({
        "题组": name, "选项数": len(items),
        "平均选中数": round(float(df[f"{items[0][:2]}_选中数"].mean()), 3),
        "KR-20": round(k20, 4), "α（对照）": round(a, 4),
        "ω_total（polychoric，仅供对照）": round(om, 4),
        "平均四分相关 φ": round(mean_interitem_r(Rp), 4),
    })
kr20_df = pd.DataFrame(kr_rows)
sheets["KR20_多选题组"] = kr20_df

# --- 3.3 正态性 ---
log("[Step 3] Shapiro–Wilk 正态性 …")
CONT = {
    "Q4_压力源指数": ("Q4 压力源指数", "Stressor index"),
    "Q5_身心健康": ("Q5 身心健康影响", "Health impact"),
    "Q10_体制内偏好": ("Q10 体制内偏好", "Public-sector preference"),
    "Q10_灵活新就业偏好": ("Q10 灵活/新就业偏好", "Flexible/new-form preference"),
    "Q11_指数A_剔除不确定": ("Q11 期望受影响指数", "Expectation-impact index"),
    "F1": ("F1 外在情境—内在规划（Step 6）", "F1 External-contextual"),
    "F2": ("F2 市场竞争—人力资本（Step 6）", "F2 Market-competition"),
}
norm_rows = []
for col, (cn, en) in CONT.items():
    x = df[col].dropna().values
    w, p = shapiro(x)
    norm_rows.append({
        "变量": col, "中文": cn, "English": en, "N": len(x),
        "均值": round(float(x.mean()), 4), "SD": round(float(x.std(ddof=1)), 4),
        "偏度": round(float(skew(x)), 3), "峰度": round(float(kurtosis(x)), 3),
        "Shapiro–Wilk W": round(float(w), 4), "p": float(f"{p:.3e}"),
        "拒绝正态（p<0.05）": "是" if p < 0.05 else "否",
    })
norm_df = pd.DataFrame(norm_rows)
sheets["正态性检验"] = norm_df
_n_reject = int((norm_df["p"] < 0.05).sum())
log(f"    正态性：{_n_reject}/{len(norm_df)} 个变量拒绝正态")


# ======================================================================
# Step 4 · 描述 + Kendall's W
# ======================================================================
log("[Step 4] Q4 / Q10 描述 + Kendall's W …")
def likert_table(items, cnd, end, top2_label="选 4–5 占比"):
    rows = []
    X = df[items]
    for it in items:
        x = X[it].dropna()
        rows.append({
            "题项": it, "中文": cnd[it], "English": end[it],
            "均值": round(float(x.mean()), 3), "SD": round(float(x.std(ddof=1)), 3),
            "中位数": float(x.median()),
            "选1–2占比": round(float((x <= 2).mean()), 3),
            top2_label: round(float((x >= 4).mean()), 3),
            "95%CI下限": round(float(x.mean() - Z95 * x.std(ddof=1) / np.sqrt(len(x))), 3),
            "95%CI上限": round(float(x.mean() + Z95 * x.std(ddof=1) / np.sqrt(len(x))), 3),
        })
    t = pd.DataFrame(rows).sort_values("均值", ascending=False).reset_index(drop=True)
    t.insert(0, "排名", np.arange(1, len(t) + 1))
    return t

t_q4 = likert_table(Q4_ITEMS, Q4_CN, Q4_EN)
t_q10 = likert_table(Q10_ITEMS, Q10_CN, Q10_EN)
sheets["表2_Q4描述"] = t_q4
sheets["表3_Q10描述"] = t_q10

# 基线核对
_idx = float(df["Q4_压力源指数"].mean())
_chk_desc = []
_chk_desc.append(f"Q4 压力源指数均值 = **{_idx:.4f}**（论文一基线 {BASELINE_Q4_INDEX}，"
                 f"差 {abs(_idx-BASELINE_Q4_INDEX):.4f}）")
_base = pd.DataFrame({
    "题项": [f"Q10_{i}" for i in range(1, 9)],
    "题项中文": [Q10_CN[f"Q10_{i}"] for i in range(1, 9)],
    "基线_论文一": [BASELINE_Q10[f"Q10_{i}"] for i in range(1, 9)],
})
_base["本次"] = _base["题项"].map(t_q10.set_index("题项")["均值"]).astype(float)
_base["差"] = (_base["本次"] - _base["基线_论文一"]).round(3)
_base["排序一致"] = (_base["本次"].rank(ascending=False).astype(int)
                 == _base["基线_论文一"].rank(ascending=False).astype(int))
sheets["Q10基线核对"] = _base

_q5_heavy = float((df["Q5_身心健康"] >= 4).mean())
_chk_desc.append(f"Q5 重度（4–5）占比 = **{pct(_q5_heavy)}**"
                 f"（论文一基线 {pct(BASELINE_Q5_HEAVY)}）")

# Kendall's W
X10 = df[Q10_ITEMS].values
w10, x210, p10, T10 = kendall_w(X10)
X4 = df[Q4_ITEMS].values
w4, x24, p4, T4 = kendall_w(X4)
# 未校正版作对照（看并列校正的影响有多大）
def kendall_w_nocorr(X):
    X = np.asarray(X, float)
    m, k = X.shape
    R = np.apply_along_axis(rankdata, 1, X)
    Rs = R.sum(axis=0)
    S = np.sum((Rs - m * (k + 1) / 2) ** 2)
    return float(12 * S / (m * m * (k ** 3 - k)))

w_rows = [{
    "题组": "Q10 工作类型偏好", "被评对象数 k": 8, "评判者数 m": N,
    "并列校正项 ΣT": round(T10, 1),
    "W（并列校正）": round(w10, 4), "χ²": round(x210, 2), "df": 7,
    "p": float(f"{p10:.3e}"),
    "W（不校正，对照）": round(kendall_w_nocorr(X10), 4),
    "平均序位一致性解读": ("存在共识" if (p10 < 0.05 and w10 >= 0.30)
                    else ("统计显著但量级偏弱" if p10 < 0.05 else "无共识")),
}, {
    "题组": "Q4 就业压力源", "被评对象数 k": 9, "评判者数 m": N,
    "并列校正项 ΣT": round(T4, 1),
    "W（并列校正）": round(w4, 4), "χ²": round(x24, 2), "df": 8,
    "p": float(f"{p4:.3e}"),
    "W（不校正，对照）": round(kendall_w_nocorr(X4), 4),
    "平均序位一致性解读": ("存在共识" if (p4 < 0.05 and w4 >= 0.30)
                    else ("统计显著但量级偏弱" if p4 < 0.05 else "无共识")),
}]
w_df = pd.DataFrame(w_rows)
sheets["Kendall_W"] = w_df


# ======================================================================
# Step 5 · 多重响应 / Q11 / Q7 / 人口学
# ======================================================================
log("[Step 5] 多重响应 + Q11 + Q7 + 人口学 …")
def multi_response(items, cnd, end, subset_mask=None, label=""):
    sub = df if subset_mask is None else df[subset_mask]
    n = len(sub)
    rows = []
    for it in items:
        k = int(sub[it].sum())
        lo, hi = wilson_ci(k, n)
        rows.append({
            "题组": label, "题项": it, "中文": cnd[it], "English": end[it],
            "选中人数": k, "N": n, "选中率": round(k / n, 4),
            "Wilson 95% CI 下限": round(lo, 4), "Wilson 95% CI 上限": round(hi, 4),
        })
    return pd.DataFrame(rows).sort_values("选中率", ascending=False).reset_index(drop=True)

mr = pd.concat([
    multi_response(Q6_ITEMS, Q6_CN, Q6_EN, label="Q6 未来发展优势"),
    multi_response(Q8_ITEMS, Q8_CN, Q8_EN, label="Q8 心理/行为表现"),
    multi_response(Q9_ITEMS, Q9_CN, Q9_EN, label="Q9 应对方式"),
], ignore_index=True)
sheets["表4_多重响应"] = mr

# Q9 卡点 2：超限 98 人的敏感性（主分析 1026，附录 928）
_mask_ok = df["Q9_超限"] == 0
mr_q9_alt = multi_response(Q9_ITEMS, Q9_CN, Q9_EN, subset_mask=_mask_ok,
                           label="Q9（剔除超限 98 人后，N=928）")
mr_q9_main = multi_response(Q9_ITEMS, Q9_CN, Q9_EN, label="Q9（主分析 N=1026）")
cmp9 = mr_q9_main[["题项", "中文", "选中率"]].merge(
    mr_q9_alt[["题项", "选中率"]], on="题项", suffixes=("_1026", "_928"))
cmp9["差(pp)"] = ((cmp9["选中率_1026"] - cmp9["选中率_928"]) * 100).round(2)
sheets["Q9敏感性_剔除超限"] = cmp9.sort_values("差(pp)", ascending=False)
_max9 = float(cmp9["差(pp)"].abs().max())
_mean9 = float(cmp9["差(pp)"].abs().mean())

# Q9 勾选数分布
ch9 = pd.DataFrame({
    "勾选数": df["Q9_选中数"].value_counts().sort_index().index,
    "人数": df["Q9_选中数"].value_counts().sort_index().values,
})
ch9["占比"] = (ch9["人数"] / N).round(4)
ch9["是否超限"] = ch9["勾选数"].apply(lambda v: "—" if v <= 2 else "⚠ 超出题设限选 2 项")
sheets["Q9勾选数分布"] = ch9

# --- Q11 ---
q11_rows = []
for it in Q11_ITEMS:
    s = df[it]
    n_unsure = int(s.isna().sum())
    v = s.dropna()
    n0, n1, n2 = int((v == 0).sum()), int((v == 1).sum()), int((v == 2).sum())
    lo, hi = wilson_ci(n_unsure, N)
    q11_rows.append({
        "题项": it, "中文": Q11_CN[it], "English": Q11_EN[it],
        "没有影响 n": n0, "影响较小 n": n1, "影响较大 n": n2,
        "没有影响 %": round(n0 / len(v), 4), "影响较小 %": round(n1 / len(v), 4),
        "影响较大 %": round(n2 / len(v), 4),
        "不确定人数": n_unsure, "不确定率": round(n_unsure / N, 4),
        "不确定率 CI 下限": round(lo, 4), "不确定率 CI 上限": round(hi, 4),
        "均值（剔除不确定）": round(float(v.mean()), 3),
        "SD": round(float(v.std(ddof=1)), 3), "有效n": len(v),
    })
q11 = pd.DataFrame(q11_rows)
sheets["表5_Q11"] = q11
_unsure_rng = (q11["不确定率"].min(), q11["不确定率"].max())
_idxA_n = int(df["Q11_指数A_剔除不确定"].notna().sum())
_idxA_mean = float(df["Q11_指数A_剔除不确定"].mean())
_cell_missing = int(df[Q11_ITEMS].isna().sum().sum())

# --- Q7 双口径 ---
single = df["意向4类"].value_counts().reindex(INTENT_ORDER)
multi = {lab: int(df[f"Y多选_{lab}"].sum()) for lab in INTENT_ORDER}
t7 = pd.DataFrame({
    "类别": INTENT_ORDER,
    "English": [INTENT_EN[c] for c in INTENT_ORDER],
    "单选口径 n": single.values,
    "单选口径 %": np.round(single.values / N * 100, 1),
    "多选口径 n": [multi[c] for c in INTENT_ORDER],
    "多选口径 %": np.round([multi[c] / N * 100 for c in INTENT_ORDER], 1),
    "倍差(多选/单选)": np.round([multi[c] / single[c] for c in INTENT_ORDER], 2),
})
sheets["表6_Q7双口径"] = t7
_n_multi = int((df["Q7_勾选数"] > 1).sum())

# 多选口径基线核对（手册：深造 55.9 / 直接就业 40.5 / 求稳体制 12.2 / 方向未定 4.5）
_chk_desc.append(
    "Q7 多选口径 " + "、".join(
        f"{c} {100*multi[c]/N:.1f}%" for c in INTENT_ORDER)
    + "（手册基线 55.9 / 40.5 / 12.2 / 4.5）")

# --- 人口学分层（性别 / 专业；年级留给 RQ2）---
test_rows = []
for col, (cn, en) in CONT.items():
    v = df[col].dropna()
    m = df.loc[v.index]
    # 性别：Mann–Whitney
    g1 = v[m["sex"] == "male"].values
    g2 = v[m["sex"] == "female"].values
    u, pu = mannwhitneyu(g1, g2, alternative="two-sided")
    rb = 2 * u / (len(g1) * len(g2)) - 1
    test_rows.append({"结局": cn, "English": en, "变量": "性别（男 vs 女）",
                      "方法": "Mann–Whitney U", "统计量": round(float(u), 1),
                      "效应量": round(float(rb), 4), "效应量类型": "rank-biserial r",
                      "n1": len(g1), "n2": len(g2), "p": float(pu)})
    # 专业：Kruskal–Wallis
    groups = [v[m["major"] == g].values for g in MAJOR_CN.keys()]
    h, ph = kruskal(*groups)
    e2 = epsilon_sq(h, len(groups), len(v))
    test_rows.append({"结局": cn, "English": en, "变量": "专业（5 类）",
                      "方法": "Kruskal–Wallis H", "统计量": round(float(h), 3),
                      "效应量": round(e2, 4), "效应量类型": "ε²",
                      "n1": len(v), "n2": np.nan, "p": float(ph)})

for vname, levels in [("性别（4 类交叉）", "sex"), ("专业（4 类交叉）", "major")]:
    ct = pd.crosstab(df["意向4类"], df[levels]).reindex(INTENT_ORDER)
    c2, pc, dof, _exp = chi2_contingency(ct.values)
    v_cramer = float(np.sqrt(c2 / (ct.values.sum() * min(ct.shape[0] - 1,
                                                         ct.shape[1] - 1))))
    test_rows.append({"结局": "意向4类", "English": "Career-intention categories",
                      "变量": vname, "方法": "χ² 独立性检验",
                      "统计量": round(float(c2), 3), "效应量": round(v_cramer, 4),
                      "效应量类型": "Cramér's V", "n1": int(ct.values.sum()),
                      "n2": np.nan, "p": float(pc)})
    min_exp = float(np.min(_exp))
    if min_exp < 5:
        notes.append(f"⚠ {vname} 的 χ² 期望频数最小为 {min_exp:.1f}（<5），"
                     f"须在正文标注；建议补充 Fisher 精确检验或合并稀疏类别。")

tests = pd.DataFrame(test_rows)
tests["p_BH"] = bh_adjust(tests["p"].values)
tests["显著(BH<0.05)"] = np.where(tests["p_BH"] < 0.05, "是", "否")
tests = tests.sort_values("p_BH").reset_index(drop=True)
sheets["人口学检验_BH"] = tests
_n_sig = int((tests["p_BH"] < 0.05).sum())


# ======================================================================
# 图 1 · 两榜
# ======================================================================
log("[图 1] 压力源强度榜 + 就业偏好榜 …")
fig, axes = plt.subplots(1, 2, figsize=(6.85, 3.6), facecolor="white")
for ax, t, title_char, col, lab in [
        (axes[0], t_q4, "(a)", "#c00000", "Stressors"),
        (axes[1], t_q10, "(b)", "#1f4e79", "Job-type preferences")]:
    t = t.sort_values("均值")           # 横向条形：从小到大画 → 图内自上而下降序
    y = np.arange(len(t))
    err = np.column_stack([t["均值"] - t["95%CI下限"], t["95%CI上限"] - t["均值"]]).T
    ax.barh(y, t["均值"], color=col, alpha=0.85, height=0.62)
    ax.errorbar(t["均值"], y, xerr=err, fmt="none", ecolor="#404040",
                elinewidth=0.8, capsize=2)
    # 第二信息层：选 4–5 占比，用橙色与灰色误差线区分，并留出端点间距
    for yy, (m, sh, hi) in enumerate(zip(t["均值"], t["选 4–5 占比"],
                                        t["95%CI上限"])):
        ax.text(max(m + 0.12, hi + 0.07), yy, f"{100*sh:.0f}%", fontsize=7,
                color="#c55a11", va="center")
    ax.set_yticks(y)
    ax.set_yticklabels(t["English"], fontsize=8)
    ax.set_xlim(2.2, 4.75)
    ax.set_xticks([2.5, 3.0, 3.5, 4.0, 4.5])
    ax.text(0.0, 1.015, title_char, transform=ax.transAxes,
            fontsize=9, fontweight="bold", va="bottom")
    ax.grid(axis="x", ls=":", lw=0.6, color="#bfbfbf")
    ax.set_axisbelow(True)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
fig.supxlabel("Mean rating (1-5)     |     orange figures = share of respondents "
              "choosing 4-5", fontsize=8)
fig.tight_layout(rect=(0, 0.04, 1, 1))
fig.savefig(os.path.join(FIG, "fig_RQ1_rankings.png"), dpi=600, facecolor="white")
plt.close(fig)

# ======================================================================
# 图 2 · 多重响应棒棒糖
# ======================================================================
log("[图 2] 多重响应棒棒糖图（Q6 / Q8 / Q9）…")
fig, axes = plt.subplots(3, 1, figsize=(6.85, 6.6), facecolor="white")
for ax, (name, items, cnd, end) in zip(
        axes,
        [("Q6 Perceived advantages", Q6_ITEMS, Q6_CN, Q6_EN),
         ("Q8 Psychological / behavioural responses", Q8_ITEMS, Q8_CN, Q8_EN),
         ("Q9 Coping strategies (max 2)", Q9_ITEMS, Q9_CN, Q9_EN)]):
    sub = mr[mr["题组"].str.startswith(name.split()[0])].sort_values("选中率")
    y = np.arange(len(sub))
    ax.hlines(y, 0, sub["选中率"], color="#8c8c8c", lw=1.1, zorder=2)
    ax.scatter(sub["选中率"], y, s=26, color="#1f4e79", zorder=3)
    ax.errorbar(sub["选中率"], y,
                xerr=[sub["选中率"] - sub["Wilson 95% CI 下限"],
                      sub["Wilson 95% CI 上限"] - sub["选中率"]],
                fmt="none", ecolor="#c00000", elinewidth=0.9, capsize=2.4, zorder=2)
    ax.set_yticks(y)
    ax.set_yticklabels(sub["English"], fontsize=8)
    _xmax = max(0.72, float(sub["Wilson 95% CI 上限"].max()) * 1.06)
    ax.set_xlim(0, _xmax)
    _tk = np.arange(0, _xmax * 0.98, 0.1)
    ax.set_xticks(_tk)
    ax.set_xticklabels([f"{int(v*100)}%" for v in _tk])
    ax.set_xlabel("Selection rate (Wilson 95% CI)")
    ax.grid(axis="x", ls=":", lw=0.6, color="#bfbfbf")
    ax.set_axisbelow(True)
    for s in ["top", "right", "left"]:
        ax.spines[s].set_visible(False)
    ax.text(0.995, 1.015, name.split()[0], transform=ax.transAxes, ha="right",
            fontsize=9, fontweight="bold", va="bottom")
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig_RQ1_multiresponse.png"), dpi=600, facecolor="white")
plt.close(fig)

# ======================================================================
# 图 3 · Q7 双口径 + Q11
# ======================================================================
log("[图 3] Q7 双口径 + Q11 受影响结构 …")
fig, axes = plt.subplots(2, 1, figsize=(6.85, 5.9), facecolor="white")

# (a) Q7 单选 vs 多选
ax = axes[0]
y = np.arange(len(INTENT_ORDER))
h = 0.36
ax.barh(y + h / 2, t7["单选口径 %"] / 100, height=h, color="#1f4e79",
        label="Single-choice coding")
ax.barh(y - h / 2, t7["多选口径 %"] / 100, height=h, color="#e8a33d",
        label="Multiple-response coding")
for i, (a1, a2) in enumerate(zip(t7["单选口径 %"], t7["多选口径 %"])):
    ax.text(a1 / 100 + 0.008, i + h / 2, f"{a1:.1f}%", va="center", fontsize=7.5)
    ax.text(a2 / 100 + 0.008, i - h / 2, f"{a2:.1f}%", va="center", fontsize=7.5)
ax.set_yticks(y)
ax.set_yticklabels([INTENT_EN[c] for c in INTENT_ORDER], fontsize=8.5)
ax.set_xlim(0, max(t7["多选口径 %"].max(), t7["单选口径 %"].max()) / 100 * 1.22)
ax.set_xticks(np.arange(0, 0.7, 0.1))
ax.set_xticklabels([f"{int(v*100)}%" for v in np.arange(0, 0.7, 0.1)])
ax.set_xlabel("Share of respondents")
ax.legend(ncol=2, loc="lower left", bbox_to_anchor=(0.0, 1.01),
          fontsize=7.5, frameon=False, borderaxespad=0.2)
ax.grid(axis="x", ls=":", lw=0.6, color="#bfbfbf")
ax.set_axisbelow(True)
for s in ["top", "right", "left"]:
    ax.spines[s].set_visible(False)
ax.text(0.995, 1.015, "(a)", transform=ax.transAxes, ha="right",
        fontsize=9, fontweight="bold", va="bottom")

# (b) Q11 堆叠结构
ax = axes[1]
q = q11.iloc[::-1]
yy = np.arange(len(q))
left = np.zeros(len(q))
for col, colname, c in [("没有影响 %", "No impact", "#a6cee3"),
                        ("影响较小 %", "Minor impact", "#fdbf6f"),
                        ("影响较大 %", "Major impact", "#b2182b")]:
    vals = q[col].values
    ax.barh(yy, vals, left=left, color=c, height=0.62, label=colname)
    left += vals
for i, rrow in q.reset_index(drop=True).iterrows():
    ax.text(1.005, i, f"{rrow['均值（剔除不确定）']:.2f}", va="center",
            fontsize=7.5, color="#404040")
ax.set_yticks(yy)
ax.set_yticklabels(q["English"], fontsize=8)
ax.set_xlim(0, 1.06)
ax.set_xticks(np.arange(0, 1.01, 0.2))
ax.set_xticklabels([f"{int(v*100)}%" for v in np.arange(0, 1.01, 0.2)])
ax.set_xlabel("Distribution of responses (excluding 'uncertain'); "
              "right-hand numbers = mean on 0-2 scale")
ax.legend(ncol=3, loc="lower left", bbox_to_anchor=(0.0, 1.01),
          fontsize=7.5, frameon=False, borderaxespad=0.2)
ax.grid(axis="x", ls=":", lw=0.6, color="#bfbfbf")
ax.set_axisbelow(True)
for s in ["top", "right", "left"]:
    ax.spines[s].set_visible(False)
ax.text(0.995, 1.015, "(b)", transform=ax.transAxes, ha="right",
        fontsize=9, fontweight="bold", va="bottom")
fig.tight_layout(h_pad=2.2)
fig.savefig(os.path.join(FIG, "fig_RQ1_q7_q11.png"), dpi=600, facecolor="white")
plt.close(fig)


# ======================================================================
# 写报告
# ======================================================================
log("[输出] 写 Excel + Markdown …")
sheets["口径自检与基线核对"] = pd.DataFrame({"核对项": _chk_desc + notes})

with pd.ExcelWriter(XL, engine="openpyxl") as w:
    for k, v in sheets.items():
        v.to_excel(w, sheet_name=k[:31], index=False)

# ---------------- Markdown ----------------
for s in _chk_desc:
    pass

for _, r in rel.iterrows():
    ci = ("—" if np.isnan(r["α 的 95% CI 下限"])
          else f"[{r['α 的 95% CI 下限']:.3f}, {r['α 的 95% CI 上限']:.3f}]")
    kr = "—" if np.isnan(r["KR-20"]) else f"{r['KR-20']:.3f}"

for _, r in kr20_df.iterrows():
    pass

for _, r in norm_df.iterrows():
    pass

for _, r in t_q4.iterrows():
    pass

for _, r in t_q10.iterrows():
    pass
for _, r in _base.iterrows():
    pass

for _, r in w_df.iterrows():
    pass

for name in ["Q6 未来发展优势", "Q8 心理/行为表现", "Q9 应对方式"]:
    sub = mr[mr["题组"] == name]
    for _, r in sub.iterrows():
        pass

for _, r in cmp9.iterrows():
    pass
for _, r in ch9.iterrows():
    pass

for _, r in q11.iterrows():
    pass

for _, r in t7.iterrows():
    pass

for _, r in tests.iterrows():
    pass



log("完成")
print("=" * 70)
print(f"KMO-free summary | α(Q4)={a_q4:.3f} ω={om_q4:.3f} | "
      f"α(Q10)={a_q10:.3f} ω={om_q10:.3f} | α(Q11,N={Q11_N})={a_q11:.3f}")
print(f"W(Q10)={w10:.4f} p={p10:.3e} | W(Q4)={w4:.4f} p={p4:.3e}")
print(f"Q9 敏感性：平均 {_mean9:.2f}pp，最大 {_max9:.2f}pp")
print(f"人口学：{_n_sig}/{len(tests)} 项 BH 显著")
print(f"Excel → {XL}")
