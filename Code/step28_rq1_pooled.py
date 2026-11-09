# -*- coding: utf-8 -*-
"""
RQ1 画像 · 合并样本（N=1579）敏感性分析
=====================================
问题：RQ1 的压力—偏好画像，主口径要不要从 1026 改成 1579？

背景（口径红线，不可改口）：
  - 论文二已定案：RQ1 描述画像用 1026；RQ2 主分析 1026；RQ3 用 1579。
  - 合并的前提（Chow/pooling 检验）只对 **5 个结局的「批次×压力」交互** 做过验证，
    那是 RQ3 的合并依据，**不等于** RQ1 描述性画像也可以直接合并 ——
    描述统计没有"交互项"可检验，合并的代价是**样本构成被 553 那批拉偏**。
  - 553 是工科定向批（工学 94.2%、大二 69.3% + 研究生 29.7%、无大四），
    且采集时点为 9 月（1026 为 5–6 月）。

本脚本做的事：
  1. 在 1026 与 1579 上**用同一套函数**重算 RQ1 全部画像指标（防口径漂移）；
  2. 逐项给出 Δ、Cohen's d、Spearman ρ、Top-3 重合数；
  3. 出 5 张对比图（图内文字英文，中文图题写在报告里）；
  4. 给出"是否改口径"的建议与理由。

产出：
  论文二_输出/RQ1_合并口径结果.xlsx（9 sheet）
  figs2/fig_RQ1_pooled_{composition,rank,indices,multiresp,intent}.png

用法：E:/Python/python.exe step28_rq1_pooled.py
"""
import sys as _sys
if hasattr(_sys.stdout, "reconfigure"):
    _sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import os
import time
import numpy as np
import pandas as pd
from scipy.stats import (norm, chi2, rankdata, spearmanr)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from poly_utils import polychoric_matrix, nearest_pd

T0 = time.time()
BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "论文二_输出")
FIG = os.path.join(BASE, "figs2")
os.makedirs(OUT, exist_ok=True)
os.makedirs(FIG, exist_ok=True)

F1026 = os.path.join(BASE, "论文二数据集_1026.xlsx")
F1579 = os.path.join(BASE, "论文二数据集_1579.xlsx")
XL = os.path.join(OUT, "RQ1_合并口径结果.xlsx")

Z95 = float(norm.ppf(0.975))
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 9

C_A, C_B = "#B4573C", "#185FA5"      # 1026 = 砖红 / 1579 = 蓝


def log(m):
    print(f"[{time.time()-T0:6.1f}s] {m}", flush=True)


# ----------------------------------------------------------------------
# 1. 复用 step21 已校准的统计实现（exec 取用，不复制，防漂移）
# ----------------------------------------------------------------------
_s21 = open(os.path.join(BASE, "step21_descriptives.py"), encoding="utf-8").read()


def _slice(src, a, b):
    i = src.index(a)
    j = src.index(b, i)
    return src[i:j]


_blocks = [
    _slice(_s21, "def wilson_ci(", "def cronbach_alpha("),
    _slice(_s21, "def cronbach_alpha(", "def kr20("),
    _slice(_s21, "def omega_total(", "def kendall_w("),        # 含 mean_interitem_r
    _slice(_s21, "def kendall_w(X):", "def bh_adjust("),
    _slice(_s21, "def kendall_w_nocorr(", "\nw_rows = ["),
]
_ns = {"np": np, "rankdata": rankdata, "chi2": chi2, "Z95": Z95,
       "polychoric_matrix": polychoric_matrix, "nearest_pd": nearest_pd}
exec("\n".join(_blocks), _ns)
wilson_ci = _ns["wilson_ci"]
cronbach_alpha = _ns["cronbach_alpha"]
omega_total = _ns["omega_total"]
kendall_w = _ns["kendall_w"]
kendall_w_nocorr = _ns["kendall_w_nocorr"]
log("已从 step21 复用 5 个已校准函数（wilson_ci / alpha / omega / kendall_w ×2）")

# ----------------------------------------------------------------------
# 2. 题项字典（与 step21 逐字一致）
# ----------------------------------------------------------------------
Q4_ITEMS = [f"Q4_{i}" for i in range(1, 10)]
Q6_ITEMS = [f"Q6_{i}" for i in range(1, 10)]
Q8_ITEMS = [f"Q8_{i}" for i in range(1, 10)]
Q9_ITEMS = [f"Q9_{i}" for i in range(1, 8)]
Q10_ITEMS = [f"Q10_{i}" for i in range(1, 9)]
Q11_ITEMS = [f"Q11_{i}" for i in range(1, 8)]

Q4_CN = {"Q4_1": "就业市场竞争激烈程度", "Q4_2": "所学专业与市场需求匹配度",
         "Q4_3": "自身专业技能/实习经验不足", "Q4_4": "学历要求门槛",
         "Q4_5": "求职信息获取难度", "Q4_6": "家庭期望、经济压力",
         "Q4_7": "职业规划不清晰", "Q4_8": "AI 发展对就业的冲击",
         "Q4_9": "政策与制度的变化"}
Q4_EN = {"Q4_1": "Market competition", "Q4_2": "Major-market mismatch",
         "Q4_3": "Skill/internship gap", "Q4_4": "Education credentialism",
         "Q4_5": "Job information access", "Q4_6": "Family expectations & finances",
         "Q4_7": "Unclear career planning", "Q4_8": "AI impact on jobs",
         "Q4_9": "Policy & institutional change"}
Q6_CN = {"Q6_1": "专业能力优势", "Q6_2": "综合能力优势", "Q6_3": "价值观正确",
         "Q6_4": "实践经历优势", "Q6_5": "资源人脉优势", "Q6_6": "家庭支持",
         "Q6_7": "技能特长优势", "Q6_8": "职业目标清晰/自我定位明确", "Q6_9": "其他"}
Q6_EN = {"Q6_1": "Professional competence", "Q6_2": "Comprehensive competence",
         "Q6_3": "Sound values", "Q6_4": "Practical experience",
         "Q6_5": "Resources & networks", "Q6_6": "Family support",
         "Q6_7": "Technical skills", "Q6_8": "Clear career goals", "Q6_9": "Other"}
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
         "Q9_5": "Temporary avoidance", "Q9_6": "Following the majority", "Q9_7": "Other"}
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

GRADE_CN = {"y1": "大一", "y2": "大二", "y3": "大三", "y4": "大四", "grad": "研究生"}
GRADE_EN = {"y1": "Year 1", "y2": "Year 2", "y3": "Year 3", "y4": "Year 4",
            "grad": "Postgraduate"}
MAJOR_CN = {"eng": "工学", "sci": "理学", "econ": "经管", "hum": "文史哲", "oth": "其他"}
MAJOR_EN = {"eng": "Engineering", "sci": "Sciences", "econ": "Economics & management",
            "hum": "Humanities", "oth": "Other"}
SEX_CN = {"male": "男", "female": "女"}
SEX_EN = {"male": "Male", "female": "Female"}
CHANNEL_CN = {"paper": "纸质", "net": "网络"}
CHANNEL_EN = {"paper": "Paper", "net": "Online"}
INTENT_ORDER = ["深造", "直接就业", "求稳体制", "方向未定"]
INTENT_EN = {"深造": "Further study", "直接就业": "Direct employment",
             "求稳体制": "Public-sector stability", "方向未定": "Undecided"}

CORE = [("Q4_压力源指数", "压力源指数", "Stress-source index"),
        ("Q5_身心健康", "身心健康受影响", "Health & well-being impact"),
        ("Q10_体制内偏好", "体制内偏好", "Public-sector preference"),
        ("Q10_灵活新就业偏好", "灵活/新就业偏好", "Flexible / new-form preference"),
        ("Q11_指数A_剔除不确定", "期望受影响指数", "Expectation-affected index")]

# ----------------------------------------------------------------------
# 3. 读入
# ----------------------------------------------------------------------
log("读入两数据集 …")
d1 = pd.read_excel(F1026)
d2 = pd.read_excel(F1579)
d3 = d2.iloc[1026:].reset_index(drop=True)          # 增量 553
assert len(d1) == 1026 and len(d2) == 1579 and len(d3) == 553
SETS = [("1026", d1), ("1579", d2), ("553", d3)]
log(f"  1026×{d1.shape[1]} | 1579×{d2.shape[1]} | 增量 553×{d3.shape[1]}")

sheets = {}




def md_table(df, cols=None, fmts=None):
    df = df if cols is None else df[cols]
    fmts = fmts or {}
    head = "| " + " | ".join(str(c) for c in df.columns) + " |"
    line = "|" + "|".join(["---"] * len(df.columns)) + "|"
    body = []
    for _, r in df.iterrows():
        cells = []
        for c in df.columns:
            v = r[c]
            if c in fmts:
                cells.append(fmts[c](v))
            elif isinstance(v, float):
                cells.append(f"{v:.3f}")
            else:
                cells.append(str(v))
        body.append("| " + " | ".join(cells) + " |")
    return "\n".join([head, line] + body)


def cohens_d(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    a = a[~np.isnan(a)]; b = b[~np.isnan(b)]
    s = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1))
                / (len(a) + len(b) - 2))
    return float((b.mean() - a.mean()) / s) if s > 0 else np.nan


def d_label(d):
    a = abs(d)
    if a < 0.20:
        return "negligible"
    if a < 0.50:
        return "small"
    if a < 0.80:
        return "medium"
    return "large"


# ----------------------------------------------------------------------
# 4. 样本构成对比
# ----------------------------------------------------------------------
log("[1/6] 样本构成对比 …")
comp_rows = []
_SPECS = [("grade", GRADE_CN, ["y1", "y2", "y3", "y4", "grad"]),
          ("major", MAJOR_CN, ["eng", "sci", "econ", "hum", "oth"]),
          ("sex", SEX_CN, ["male", "female"]),
          ("channel", CHANNEL_CN, ["paper", "net"])]
for col, cn, order in _SPECS:
    for lv in order:
        row = {"维度": col, "取值": lv, "中文": cn.get(lv, lv)}
        for nm, dd in SETS:
            row[f"{nm}_n"] = int((dd[col] == lv).sum())
            row[f"{nm}_%"] = round(float((dd[col] == lv).mean()) * 100, 1)
        row["Δpp"] = round(row["1579_%"] - row["1026_%"], 1)
        comp_rows.append(row)
t_comp = pd.DataFrame(comp_rows)
sheets["构成对比"] = t_comp

# 最大偏移
_max_shift = t_comp.reindex(t_comp["Δpp"].abs().sort_values(ascending=False).index).head(6)

# ----------------------------------------------------------------------
# 5. Q4 / Q10 题项级对比（排序稳定性）
# ----------------------------------------------------------------------
log("[2/6] Q4 / Q10 题项级对比 …")


def item_table(items, cn, en, label):
    rows = []
    for nm, dd in SETS[:2]:
        for it in items:
            x = pd.to_numeric(dd[it], errors="coerce").dropna()
            rows.append({"题组": label, "集": nm, "题项": it,
                         "中文": cn[it], "English": en[it],
                         "均值": float(x.mean()), "SD": float(x.std(ddof=1)),
                         "选4+占比": float((x >= 4).mean()), "n": len(x)})
    t = pd.DataFrame(rows)
    piv = t.pivot_table(index=["题组", "题项", "中文", "English"], columns="集",
                        values=["均值", "SD", "选4+占比"]).reset_index()
    # ⚠ pivot_table 的 MultiIndex columns 会按**字符序重排**（此处 SD < 均值 < 选4+占比），
    #   硬编码重命名顺序会整体错位（初版就是这样把均值列写成了 SD 列，ρ 从 0.983 掉到 0.850）。
    #   一律按 MultiIndex 自身取值动态拼名。
    piv.columns = [f"{a}_{b}" if b else a for a, b in piv.columns]
    piv = piv.rename(columns={"选4+占比_1026": "选4+_1026",
                              "选4+占比_1579": "选4+_1579"})
    piv["Δ均值"] = (piv["均值_1579"] - piv["均值_1026"]).round(4)
    piv["d"] = [cohens_d(d1[it].astype(float), d2[it].astype(float)) for it in piv["题项"]]
    piv["d_量级"] = piv["d"].apply(d_label)
    piv["排名_1026"] = piv["均值_1026"].rank(ascending=False).astype(int)
    piv["排名_1579"] = piv["均值_1579"].rank(ascending=False).astype(int)
    piv["Δ排名"] = piv["排名_1026"] - piv["排名_1579"]
    piv["Δ选4+_pp"] = ((piv["选4+_1579"] - piv["选4+_1026"]) * 100).round(2)
    # Cohen's d 需要按题算：上面列表推导里 it 来自 piv["题项"]，正确
    return piv.sort_values("均值_1579", ascending=False).reset_index(drop=True)


t_q4 = item_table(Q4_ITEMS, Q4_CN, Q4_EN, "Q4 压力源")
t_q10 = item_table(Q10_ITEMS, Q10_CN, Q10_EN, "Q10 偏好")
sheets["Q4题项对比"] = t_q4
sheets["Q10题项对比"] = t_q10

_rho4 = float(spearmanr(t_q4["均值_1026"], t_q4["均值_1579"]).statistic)
_rho10 = float(spearmanr(t_q10["均值_1026"], t_q10["均值_1579"]).statistic)
_top3_4 = len(set(t_q4.nsmallest(3, "排名_1026")["题项"])
              & set(t_q4.nsmallest(3, "排名_1579")["题项"]))
_top3_10 = len(set(t_q10.nsmallest(3, "排名_1026")["题项"])
               & set(t_q10.nsmallest(3, "排名_1579")["题项"]))
_max_d4 = float(t_q4["d"].abs().max())
_max_d10 = float(t_q10["d"].abs().max())
_nflip4 = int((t_q4["Δ排名"] != 0).sum())
_nflip10 = int((t_q10["Δ排名"] != 0).sum())
log(f"  Q4 ρ={_rho4:.4f} Top3={_top3_4}/3 最大|d|={_max_d4:.3f} 换位 {_nflip4}/9")
log(f"  Q10 ρ={_rho10:.4f} Top3={_top3_10}/3 最大|d|={_max_d10:.3f} 换位 {_nflip10}/8")

# ----------------------------------------------------------------------
# 6. 核心连续指标
# ----------------------------------------------------------------------
log("[3/6] 核心连续指标 …")
core_rows = []
for col, cn, en in CORE:
    a = pd.to_numeric(d1[col], errors="coerce")
    b = pd.to_numeric(d2[col], errors="coerce")
    d = cohens_d(a, b)
    core_rows.append({
        "指标": col, "中文": cn, "English": en,
        "n_1026": int(a.notna().sum()), "n_1579": int(b.notna().sum()),
        "均值_1026": round(float(a.mean()), 4), "均值_1579": round(float(b.mean()), 4),
        "SD_1026": round(float(a.std()), 4), "SD_1579": round(float(b.std()), 4),
        "Δ": round(float(b.mean() - a.mean()), 4),
        "CI95_1026": f"[{a.mean()-Z95*a.std()/np.sqrt(a.notna().sum()):.3f}, "
                     f"{a.mean()+Z95*a.std()/np.sqrt(a.notna().sum()):.3f}]",
        "CI95_1579": f"[{b.mean()-Z95*b.std()/np.sqrt(b.notna().sum()):.3f}, "
                     f"{b.mean()+Z95*b.std()/np.sqrt(b.notna().sum()):.3f}]",
        "Cohen_d": round(d, 4), "量级": d_label(d),
    })
t_core = pd.DataFrame(core_rows)
sheets["核心指标对比"] = t_core
_max_core_d = float(t_core["Cohen_d"].abs().max())

# ----------------------------------------------------------------------
# 7. 信度 / 共识度
# ----------------------------------------------------------------------
log("[4/6] 信度与 Kendall's W …")
rel_rows = []
for nm, dd in SETS[:2]:
    for lbl, items in (("Q4 压力源", Q4_ITEMS), ("Q10 偏好", Q10_ITEMS),
                       ("Q11 期望", Q11_ITEMS)):
        X = dd[items].apply(pd.to_numeric, errors="coerce").dropna()
        Xv = X.values
        a_ = cronbach_alpha(Xv)
        try:
            R = nearest_pd(polychoric_matrix(Xv))
            w_, _lam = omega_total(R)
        except Exception:
            w_ = np.nan
        rel_rows.append({"集": nm, "题组": lbl, "指标": "α ／ ω", "n": len(X),
                         "α": round(a_, 4), "ω": round(float(w_), 4)})
    for lbl, items in (("Q4 压力源", Q4_ITEMS), ("Q10 偏好", Q10_ITEMS)):
        Xv = dd[items].apply(pd.to_numeric, errors="coerce").dropna().values
        wv, x2v, pv, Tv = kendall_w(Xv)
        rel_rows.append({"集": nm, "题组": lbl, "指标": "Kendall's W",
                         "n": len(Xv), "α": round(wv, 4),
                         "ω": round(kendall_w_nocorr(Xv), 4)})
t_rel = pd.DataFrame(rel_rows)
t_rel = t_rel.rename(columns={"α": "系数", "ω": "对照值"})
sheets["信度与W对比"] = t_rel

# ----------------------------------------------------------------------
# 8. 多重响应 Q6 / Q8 / Q9
# ----------------------------------------------------------------------
log("[5/6] 多重响应 …")
mr_rows = []
for lbl, items, cn, en in (("Q6 优势", Q6_ITEMS, Q6_CN, Q6_EN),
                           ("Q8 反应", Q8_ITEMS, Q8_CN, Q8_EN),
                           ("Q9 应对", Q9_ITEMS, Q9_CN, Q9_EN)):
    for it in items:
        r = {"题组": lbl, "题项": it, "中文": cn[it], "English": en[it]}
        for nm, dd in SETS[:2]:
            k = int(pd.to_numeric(dd[it], errors="coerce").fillna(0).sum())
            n = len(dd)
            lo, hi = wilson_ci(k, n)
            r[f"率_{nm}"] = round(k / n, 4)
            r[f"CIlo_{nm}"] = round(lo, 4)
            r[f"CIhi_{nm}"] = round(hi, 4)
        r["Δpp"] = round((r["率_1579"] - r["率_1026"]) * 100, 2)
        mr_rows.append(r)
t_mr = pd.DataFrame(mr_rows)
sheets["多重响应对比"] = t_mr
_max_mr_pp = float(t_mr["Δpp"].abs().max())

# ----------------------------------------------------------------------
# 9. Q7 双口径 + Q11 七项
# ----------------------------------------------------------------------
log("[6/6] Q7 双口径 / Q11 七项 …")
q7_rows = []
for nm, dd in SETS:
    n = len(dd)
    single = dd["意向4类"].value_counts()
    for lab in INTENT_ORDER:
        k1 = int(single.get(lab, 0))
        k2 = int(pd.to_numeric(dd[f"Y多选_{lab}"], errors="coerce").fillna(0).sum())
        q7_rows.append({"集": nm, "N": n, "类别": lab, "English": INTENT_EN[lab],
                        "单选_n": k1, "单选_%": round(k1 / n * 100, 2),
                        "多选_n": k2, "多选_%": round(k2 / n * 100, 2),
                        "倍差": round(k2 / k1, 3) if k1 else np.nan})
t_q7 = pd.DataFrame(q7_rows)
sheets["Q7双口径对比"] = t_q7

q11_rows = []
for nm, dd in SETS[:2]:
    for it in Q11_ITEMS:
        x = pd.to_numeric(dd[it], errors="coerce")
        x = x[x.isin([0, 1, 2])]
        q11_rows.append({"集": nm, "题项": it, "中文": Q11_CN[it], "English": Q11_EN[it],
                         "均值0-2": round(float(x.mean()), 4),
                         "选2占比": round(float((x == 2).mean()), 4),
                         "选0占比": round(float((x == 0).mean()), 4),
                         "n": len(x)})
t_q11 = pd.DataFrame(q11_rows)
piv11 = t_q11.pivot_table(index=["题项", "中文", "English"], columns="集",
                          values=["均值0-2", "选2占比"]).reset_index()
piv11.columns = [f"{a}_{b}" if b else a for a, b in piv11.columns]
piv11 = piv11.rename(columns={"均值0-2_1026": "均值_1026", "均值0-2_1579": "均值_1579",
                              "选2占比_1026": "选2%_1026", "选2占比_1579": "选2%_1579"})
piv11["Δ均值"] = (piv11["均值_1579"] - piv11["均值_1026"]).round(4)
piv11["Δ选2_pp"] = ((piv11["选2%_1579"] - piv11["选2%_1026"]) * 100).round(2)
piv11["排名_1026"] = piv11["均值_1026"].rank(ascending=False).astype(int)
piv11["排名_1579"] = piv11["均值_1579"].rank(ascending=False).astype(int)
sheets["Q11七项对比"] = piv11.sort_values("均值_1579", ascending=False)

# ======================================================================
# 图 1 · 样本构成对比
# ======================================================================
log("[图 1] 构成对比 …")
fig, axes = plt.subplots(2, 2, figsize=(6.85, 5.4))
_PANELS = [("grade", [GRADE_EN[l] for l in ["y1", "y2", "y3", "y4", "grad"]],
            ["y1", "y2", "y3", "y4", "grad"], "Grade"),
           ("major", [MAJOR_EN[l] for l in ["eng", "sci", "econ", "hum", "oth"]],
            ["eng", "sci", "econ", "hum", "oth"], "Major"),
           ("sex", ["Male", "Female"], ["male", "female"], "Sex"),
           ("channel", ["Paper", "Online"], ["paper", "net"], "Channel")]
_SETLAB = ["1026", "1579", "553 (added)"]
_COLSET = ["#8C8C89", C_B, C_A]
for ax, (col, ens, lvs, title) in zip(axes.ravel(), _PANELS):
    yy = np.arange(3)
    left = np.zeros(3)
    for k, lv in enumerate(lvs):
        vals = np.array([float((dd[col] == lv).mean()) * 100 for _, dd in SETS])
        ax.barh(yy, vals, left=left, height=0.55, color=f"C{k}",
                edgecolor="white", linewidth=0.6,
                label=ens[k] if col == "grade" else None)
        for i, v in enumerate(vals):
            if v >= 6:
                ax.text(left[i] + v / 2, i, f"{v:.0f}", ha="center", va="center",
                        fontsize=7.4, color="white")
        left += vals
    ax.set_yticks(yy)
    ax.set_yticklabels(_SETLAB, fontsize=8.6)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("%", fontsize=8.4)
    ax.set_title(title, fontsize=9.6, loc="left", pad=3)
    ax.tick_params(labelsize=8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
handles = [mpatches.Patch(color=f"C{k}", label=e) for k, e in enumerate(_PANELS[0][1])]
fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=8, frameon=False,
           bbox_to_anchor=(0.5, -0.045))
fig.suptitle("Sample composition: main sample vs pooled vs added batch",
             fontsize=10.2, y=0.985)
fig.tight_layout(rect=[0, 0.055, 1, 0.965])
fig.savefig(os.path.join(FIG, "fig_RQ1_pooled_composition.png"),
            dpi=600, bbox_inches="tight", facecolor="white")
plt.close(fig)

# ======================================================================
# 图 2 · Q4 / Q10 排序哑铃
# ======================================================================
log("[图 2] 排序哑铃 …")
fig, axes = plt.subplots(1, 2, figsize=(6.85, 4.5))
for ax, (t, title) in zip(axes, [(t_q4, "Q4 stress sources (nine items)"),
                                 (t_q10, "Q10 employer-type preferences (eight items)")]):
    t = t.sort_values("均值_1579", ascending=True).reset_index(drop=True)
    yy = np.arange(len(t))
    ax.plot([t["均值_1026"], t["均值_1579"]],
            [yy, yy], color="#B8B8B4", lw=1.1, zorder=1)
    ax.scatter(t["均值_1026"], yy, s=42, facecolor="white", edgecolor=C_A,
               linewidth=1.4, zorder=2, label="1026")
    ax.scatter(t["均值_1579"], yy, s=42, color=C_B, zorder=3, label="1579")
    ax.set_yticks(yy)
    ax.set_yticklabels(list(t["English"]), fontsize=8.2)
    _xlo = float(min(t["均值_1026"].min(), t["均值_1579"].min()))
    _xhi = float(max(t["均值_1026"].max(), t["均值_1579"].max()))
    ax.set_xlim(_xlo - 0.20, _xhi + 0.30)
    ax.set_xlabel("Mean (1–5)", fontsize=8.4)
    ax.set_title(title, fontsize=9.6, loc="left", pad=3)
    ax.tick_params(labelsize=8)
    ax.grid(axis="x", ls=":", color="#D6D6D2", lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    for i, r in t.iterrows():
        _tag = "" if r["Δ排名"] == 0 else f"  ({r['排名_1026']}→{r['排名_1579']})"
        ax.text(_xhi + 0.30, i, f"{r['Δ均值']:+.2f}{_tag}", va="center",
                ha="right", fontsize=7.2, color="#5F5E5A")
ax2 = axes[0]
ax2.legend(fontsize=8, loc="upper left", frameon=False)
fig.suptitle("Item-level shift after pooling: means, rank changes, and deltas",
             fontsize=10.2, y=0.985)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(os.path.join(FIG, "fig_RQ1_pooled_rank.png"),
            dpi=600, bbox_inches="tight", facecolor="white")
plt.close(fig)

# ======================================================================
# 图 3 · 核心指标（并排 + Cohen's d）
# ======================================================================
log("[图 3] 核心指标 …")
fig, ax = plt.subplots(figsize=(6.85, 3.5))
t = t_core.iloc[::-1].reset_index(drop=True)
yy = np.arange(len(t))
_h = 0.30
for i, r in t.iterrows():
    a = pd.to_numeric(d1[r["指标"]], errors="coerce").dropna()
    b = pd.to_numeric(d2[r["指标"]], errors="coerce").dropna()
    ea = Z95 * a.std() / np.sqrt(len(a))
    eb = Z95 * b.std() / np.sqrt(len(b))
    ax.plot([a.mean() - ea, a.mean() + ea], [i + _h, i + _h], color=C_A, lw=1.4)
    ax.plot([b.mean() - eb, b.mean() + eb], [i - _h, i - _h], color=C_B, lw=1.4)
    ax.scatter([a.mean()], [i + _h], s=40, color=C_A, zorder=3)
    ax.scatter([b.mean()], [i - _h], s=40, color=C_B, zorder=3)
ax.set_yticks(yy)
ax.set_yticklabels([f"{r['English']}  (d={r['Cohen_d']:+.3f})" for _, r in t.iterrows()],
                   fontsize=8.4)
ax.set_xlabel("Mean with 95% CI", fontsize=8.6)
ax.set_title("Five core indices: overlapping CIs after pooling",
             fontsize=10, loc="left", pad=4)
ax.grid(axis="x", ls=":", color="#D6D6D2", lw=0.6)
ax.set_axisbelow(True)
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
ax.legend([mpatches.Patch(color=C_A), mpatches.Patch(color=C_B)],
          ["1026 (main)", "1579 (pooled)"], fontsize=8, loc="lower right", frameon=False)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig_RQ1_pooled_indices.png"),
            dpi=600, bbox_inches="tight", facecolor="white")
plt.close(fig)

# ======================================================================
# 图 4 · 多重响应
# ======================================================================
log("[图 4] 多重响应 …")
fig, axes = plt.subplots(1, 3, figsize=(6.85, 5.2), sharey=False)
for ax, lbl in zip(axes, ["Q6 优势", "Q8 反应", "Q9 应对"]):
    g = t_mr[t_mr["题组"] == lbl].sort_values("率_1579", ascending=True).reset_index(drop=True)
    yy = np.arange(len(g))
    ax.plot([g["率_1026"], g["率_1579"]], [yy, yy], color="#B8B8B4", lw=1.1, zorder=1)
    ax.scatter(g["率_1026"], yy, s=38, facecolor="white", edgecolor=C_A,
               linewidth=1.3, zorder=2)
    ax.scatter(g["率_1579"], yy, s=38, color=C_B, zorder=3)
    ax.set_yticks(yy)
    ax.set_yticklabels(list(g["English"]), fontsize=7.4)
    ax.set_xlabel("Selection rate", fontsize=8.2)
    ax.set_title({"Q6 优势": "Q6 advantages", "Q8 反应": "Q8 reactions",
                  "Q9 应对": "Q9 coping"}[lbl], fontsize=9.4, loc="left", pad=3)
    ax.tick_params(labelsize=7.6)
    ax.grid(axis="x", ls=":", color="#D6D6D2", lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
axes[0].legend([mpatches.Rectangle((0, 0), 1, 1, facecolor="white", edgecolor=C_A, lw=1.3),
                mpatches.Rectangle((0, 0), 1, 1, color=C_B)],
               ["1026", "1579"], fontsize=8, loc="upper left", frameon=False)
fig.suptitle("Multiple-response items: selection rates before and after pooling",
             fontsize=10.2, y=0.985)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(os.path.join(FIG, "fig_RQ1_pooled_multiresp.png"),
            dpi=600, bbox_inches="tight", facecolor="white")
plt.close(fig)

# ======================================================================
# 图 5 · Q7 双口径 + Q11 七项
# ======================================================================
log("[图 5] Q7 / Q11 …")
fig, axes = plt.subplots(1, 2, figsize=(6.85, 3.9))
ax = axes[0]
g = t_q7[t_q7["集"] != "553"]
_labs = [INTENT_EN[x] for x in INTENT_ORDER]
xx = np.arange(len(_labs))
_w = 0.20
_off = {"1026": (-1.5, C_A, "1026"), "1579": (0.5, C_B, "1579")}
for k, (nm, (o, col, lab)) in enumerate(_off.items()):
    sub = g[g["集"] == nm].set_index("类别").reindex(INTENT_ORDER)
    ax.bar(xx + o * _w, sub["单选_%"], width=_w, color=col, alpha=0.35,
           edgecolor=col, linewidth=1.0)
    ax.bar(xx + o * _w, sub["多选_%"], width=_w, color="white",
           edgecolor=col, hatch="///", linewidth=1.0)
ax.set_xticks(xx)
ax.set_xticklabels(_labs, fontsize=7.6, rotation=18, ha="right")
ax.set_ylabel("%", fontsize=8.4)
ax.set_title("Q7 single- vs multiple-response", fontsize=9.6, loc="left", pad=3)
ax.legend([mpatches.Patch(color=C_A, alpha=0.35),
           mpatches.Patch(facecolor="white", edgecolor="#5F5E5A", hatch="///")],
          ["Single choice", "Multiple response"], fontsize=7.8, frameon=False)
ax.tick_params(labelsize=8)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)

ax = axes[1]
g = piv11.sort_values("均值_1579", ascending=True).reset_index(drop=True)
yy = np.arange(len(g))
ax.plot([g["均值_1026"], g["均值_1579"]], [yy, yy], color="#B8B8B4", lw=1.1, zorder=1)
ax.scatter(g["均值_1026"], yy, s=40, facecolor="white", edgecolor=C_A,
           linewidth=1.3, zorder=2)
ax.scatter(g["均值_1579"], yy, s=40, color=C_B, zorder=3)
ax.set_yticks(yy)
ax.set_yticklabels(list(g["English"]), fontsize=7.8)
ax.set_xlabel("Mean (0–2)", fontsize=8.4)
ax.set_title("Q11 expectations affected", fontsize=9.6, loc="left", pad=3)
ax.tick_params(labelsize=8)
ax.grid(axis="x", ls=":", color="#D6D6D2", lw=0.6)
ax.set_axisbelow(True)
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig_RQ1_pooled_intent.png"),
            dpi=600, bbox_inches="tight", facecolor="white")
plt.close(fig)

# ======================================================================
# 判定
# ======================================================================
_stable_rank = (_rho4 >= 0.90 and _rho10 >= 0.90
                and _top3_4 == 3 and _top3_10 == 3)
_stable_level = _max_core_d < 0.20 and _max_d4 < 0.20 and _max_d10 < 0.20
_big_shift = _max_shift["Δpp"].abs().max()

if _stable_rank and _stable_level and _big_shift >= 5:
    VERDICT = ("维持 1026 作 RQ1 主口径，1579 作敏感性分析（附录）"
               "——合并后画像几乎不变，但样本构成被明显拉偏，"
               "把它当主口径等于用一个代表性更差的样本换 54% 的样本量")
elif _stable_rank and _stable_level:
    VERDICT = ("两口径可互换：画像稳定且构成偏移不大，"
               "合并可提精度，但须在正文交代批次构成")
else:
    VERDICT = ("⚠ 合并后画像发生了实质性变化，需逐项核查；"
               "在查清前维持 1026 主口径")

# ======================================================================
# 报告
# ======================================================================
log("写报告 …")














os.makedirs(OUT, exist_ok=True)

log("写 Excel …")
with pd.ExcelWriter(XL, engine="openpyxl") as w:
    for k, v in sheets.items():
        v.to_excel(w, sheet_name=k[:31], index=False)
log(f"  ✅ {XL}（{len(sheets)} sheet）")
log("RQ1 合并口径敏感性分析完成。")
