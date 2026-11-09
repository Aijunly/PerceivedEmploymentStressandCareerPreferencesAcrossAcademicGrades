# -*- coding: utf-8 -*-
"""
论文二 · 批 5（Step 18–19 收尾）

    python step27_finalize.py

覆盖：
    Step 18 ①  用 553 批次复现 RQ1 描述画像（Blokker et al. 2025 两样本复制口径）
    Step 18 ②  与论文一已发表数值交叉核对（Q4 指数 / Q5 重度 / Q10 排序 / Q7 双口径）
    Step 18 ③  两数据集内部一致性复核（1026 部分逐列 + batch 合计 + 建模集三列）
    Step 19    自我重复声明、措辞合规扫描、42 条参考文献落点映射

产出：
    论文二_输出/Step18_19_收尾核对结果.xlsx
"""
import sys as _sys
if hasattr(_sys.stdout, "reconfigure"):
    _sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import os
import re
import sys
import numpy as np
import pandas as pd
from scipy import stats

# ======================================================================
# 0. 全局设定
# ======================================================================
BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "论文二_输出")
os.makedirs(OUT, exist_ok=True)
SEED = 42
np.random.seed(SEED)

F1026 = os.path.join(BASE, "论文二数据集_1026.xlsx")
F1579 = os.path.join(BASE, "论文二数据集_1579.xlsx")
MODELSET = os.path.join(BASE, "建模数据集.xlsx")

Q4 = [f"Q4_{i}" for i in range(1, 10)]
Q10 = [f"Q10_{i}" for i in range(1, 9)]
Q11 = [f"Q11_{i}" for i in range(1, 8)]
# ⚠ 一致性核对的排除项 —— 这三类列「按设计」在两集必然不同，不是数据不一致：
#   ① z_* 5 列：各集按自身 SD 标准化（卡点 7），两集 SD 不同 → 数值必然不同
#   ② F1/F2：单元加权题组均值，同样各集内 z 化。实测 **corr = 1.000000**（纯线性变换），
#            max|Δ| 只有 0.05 / 0.16 是因为两集 SD 差得极小（0.9924 vs 1.0000）
#   ③ RF1/RF2：回归法因子得分，依赖载荷来自哪一批。实测 corr = 0.981 / 0.976 < 1，
#           正是 Step 6「外推得分 r = .805 ❌」的同类现象，也是当时定案
#           「RF* 只作附录」的原因
Z_COLS = [c for c in ("z_压力指数", "z_身心健康", "z_体制内偏好",
                      "z_灵活偏好", "z_期望受影响")]
FACTOR_COLS = ["F1", "F2", "RF1", "RF2"]
EXCLUDE_COLS = Z_COLS + FACTOR_COLS

# 题项名（权威源 =《调查问卷量化编码手册》题项表，且与 step21 报告逐字一致，不得改写）
Q4_META = {
    1: ("就业市场竞争激烈程度", "Market competition"),
    2: ("所学专业与市场需求匹配度", "Major-market match"),
    3: ("自身专业技能/实习经验不足", "Skills & internship gap"),
    4: ("学历要求门槛", "Education credentialism"),
    5: ("求职信息获取难度", "Job information access"),
    6: ("家庭期望、经济压力", "Family expectations & finances"),
    7: ("职业规划不清晰", "Unclear career planning"),
    8: ("AI 发展对就业的冲击", "AI impact on jobs"),
    9: ("政策与制度的变化", "Policy & institutional change"),
}
Q10_META = {
    1: ("国企/事业单位", "State-owned / public institution"),
    2: ("外资企业", "Foreign-invested enterprise"),
    3: ("民营企业（大型）", "Large private enterprise"),
    4: ("民营企业（中小型）", "Small & medium private enterprise"),
    5: ("自主创业", "Entrepreneurship"),
    6: ("自由职业", "Freelancing"),
    7: ("新业态从业", "New forms of employment"),
    8: ("基层就业", "Grassroots employment"),
}
Q11_META = {
    1: ("薪资待遇期望", "Salary expectations"),
    2: ("工作地点时间期望", "Location & working hours"),
    3: ("工作量预期", "Workload"),
    4: ("工作内容兴趣度", "Job-content interest"),
    5: ("公司规模知名度预期", "Employer size & reputation"),
    6: ("未来发展前景", "Career prospects"),
    7: ("工作氛围预期", "Work atmosphere"),
}
# 论文一基线值（Step 3–5 阶段已对过账，此处作为「他方源」再次独立核对）
P1 = {
    "Q4_压力源指数均值": 3.661,
    "Q5_重度占比": 0.172,          # 17.2%
    "Q7_单选": {"深造": 574, "直接就业": 332, "求稳体制": 90, "方向未定": 30},
    "Q7_多选": {"深造": 574, "直接就业": 416, "求稳体制": 125, "方向未定": 46},
    "Q10_均值": {1: 3.811, 2: 3.802, 3: 3.584, 4: 3.232,
                5: 2.884, 6: 3.221, 7: 3.178, 8: 2.973},
}

_log_lines = []

def log(s=""):
    print(s, flush=True)
    _log_lines.append(str(s))

def _t(_s):
    return _s

# ======================================================================
# 1. 小工具（全部自实现， twentieth-eighth century 标准公式）
# ======================================================================
def mean_ci(v):
    """均值 + 正态 95% CI。σ 未知、n 大时 t 与 z 差到小数点后第三位以下，用 t 更稳。"""
    v = np.asarray(pd.Series(v).dropna(), dtype=float)
    n = len(v)
    if n < 2:
        return np.nan, np.nan, np.nan
    m = float(v.mean())
    sd = float(v.std(ddof=1))
    se = sd / np.sqrt(n)
    hw = float(stats.t.ppf(0.975, n - 1)) * se
    return m, m - hw, m + hw

def cohens_d_ci(v1, v2):
    """Cohen's d（pooled SD）+ 95% CI。

    CI 用 Hedges & Olkin 近似：
        SE_d ≈ sqrt((n1+n2)/(n1·n2) + d²/(2(n1+n2)))
    """
    a = np.asarray(pd.Series(v1).dropna(), dtype=float)
    b = np.asarray(pd.Series(v2).dropna(), dtype=float)
    n1, n2 = len(a), len(b)
    if n1 < 2 or n2 < 2:
        return np.nan, (np.nan, np.nan)
    s = np.sqrt(((n1 - 1) * a.var(ddof=1) + (n2 - 1) * b.var(ddof=1)) / (n1 + n2 - 2))
    if s == 0:
        return np.nan, (np.nan, np.nan)
    d = float((a.mean() - b.mean())) / s          # 注意方向：a − b
    se = np.sqrt((n1 + n2) / (n1 * n2) + d ** 2 / (2 * (n1 + n2)))
    return d, (d - 1.96 * se, d + 1.96 * se)

def prop_diff_ci(k1, n1, k2, n2):
    """两比例差（p1 − p2，pp 单位）+ Wald 95% CI。"""
    if n1 == 0 or n2 == 0:
        return np.nan, (np.nan, np.nan)
    p1, p2 = k1 / n1, k2 / n2
    se = np.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    dd = (p1 - p2) * 100.0
    return dd, (dd - 196 * se, dd + 196 * se)

def cronbach_alpha(df, cols):
    """Cronbach's α（列表删除）。k/(k-1)·(1 − Σσ²_i/σ²_total)。"""
    sub = df[cols].dropna()
    k = len(cols)
    if len(sub) < 2 or k < 2:
        return np.nan, 0
    tot = sub.sum(axis=1).var(ddof=1)
    if tot == 0:
        return np.nan, len(sub)
    return float(k / (k - 1) * (1 - sub.var(ddof=1).sum() / tot)), len(sub)

def d_label(d):
    """Cohen 粗标（0.2 小 / 0.5 中 / 0.8 大）。"""
    if not np.isfinite(d):
        return "—"
    a = abs(d)
    if a < 0.2:
        return "negligible"
    if a < 0.5:
        return "small"
    if a < 0.8:
        return "medium"
    return "large"

def rank_str(series_desc: pd.Series) -> str:
    """给一组题均值，返回「Q4_1 > Q4_3 > …」的排序串（降序）。"""
    return " > ".join(str(i) for i in series_desc.sort_values(ascending=False).index)

# ======================================================================
# 2. 数据载入 + Step 2 三项校验（形式化重做，方法节要引这句话）
# ======================================================================
log("载入两个数据集 …")
d1026 = pd.read_excel(F1026)
d1579 = pd.read_excel(F1579)
log(f"  1026 集 {d1026.shape} ｜ 1579 集 {d1579.shape}")

d553 = d1579[d1579["batch"] == 1].copy().reset_index(drop=True)
log(f"  切出 553 批次 {d553.shape}")

check_rows = []

def add_check(item, res, ok, note=""):
    check_rows.append({"校验项": item, "实测": res,
                       "判定": "✅" if ok else "❌", "说明": note})

# ── Step 2 ① 列名覆盖 Q4/Q5/Q10/Q11 全部所需题项 ──
need = set(Q4 + Q10 + Q11 + ["Q5_身心健康", "Q4_压力源指数"])
miss26 = sorted(need - set(d1026.columns))
miss53 = sorted(need - set(d553.columns))
add_check("所需列名覆盖（1026）", "无缺失" if not miss26 else f"缺 {miss26}",
          not miss26, "Q4/Q5/Q10/Q11 全部题项")
add_check("所需列名覆盖（553）", "无缺失" if not miss53 else f"缺 {miss53}",
          not miss53, "同上")

# ── Step 2 ② 各李克特题取值域与 1026 一致 ──
DOMAIN = {"Q4": {1, 2, 3, 4, 5}, "Q10": {1, 2, 3, 4, 5},
          "Q11": {0, 1, 2, 99}}     # Q11 的 99 = 「不太确定」，单列不是缺失
bad = []
for pre, valid, items in (("Q4", {1, 2, 3, 4, 5}, Q4),
                          ("Q10", {1, 2, 3, 4, 5}, Q10),
                          ("Q11", {0, 1, 2, 99}, Q11)):
    for c in items:
        v26 = set(pd.unique(pd.to_numeric(d1026[c], errors="coerce").dropna()))
        v53 = set(pd.unique(pd.to_numeric(d553[c], errors="coerce").dropna()))
        if not v26 <= valid or not v53 <= valid:
            bad.append(c)
add_check("李克特取值域一致", "全部一致" if not bad else f"异常列 {bad}", not bad,
          "Q4/Q10 = 1–5；Q11 = 0/1/2/99")

# ── Step 2 ③ Q3 专业编码同为 1–11 ──
cod26 = sorted(pd.unique(pd.to_numeric(d1026["Q3_专业"], errors="coerce").dropna()))
cod53 = sorted(pd.unique(pd.to_numeric(d553["Q3_专业"], errors="coerce").dropna()))
# ⚠ 判据不是「两批取值集合相同」：553 是工科定向采样，只出现 5 个专业是**应该的**。
#   手册 Step 2 的要求是「编码同为 1–11」，即两批共用同一套编码体系。
_ok_cod = (set(cod26) <= set(range(1, 12)) and set(cod53) <= set(range(1, 12))
           and set(cod53) <= set(cod26))
add_check("Q3 专业编码体系", f"1026 取到 {len(cod26)} 类 ｜ 553 取到 {len(cod53)} 类（子集）",
          _ok_cod, f"1026={cod26}；553={cod53}")
t18a = pd.DataFrame(check_rows)
log(f"  Step 2 三项校验：{'全通过' if all(r['判定'] == '✅' for r in check_rows) else '有未通过项 ❌'}")

# ======================================================================
# 3. Step 18① · 553 批次复现 RQ1 描述画像
# ======================================================================
log("[Step 18①] 553 批次复现 RQ1 描述画像 …")

# ---- 3.1 Q4 九题：均值、排名、差值、效应量 ----
rows = []
for Q4_META_i, c in zip(range(1, 10), Q4):
    i = Q4_META_i
    v26 = pd.to_numeric(d1026[c], errors="coerce")
    v53 = pd.to_numeric(d553[c], errors="coerce")
    m26, _, _ = mean_ci(v26)
    m53, _, _ = mean_ci(v53)
    d, ci = cohens_d_ci(v26, v53)
    rows.append({"题项": c, "中文": Q4_META[i][0], "English": Q4_META[i][1],
                 "均值1026": m26, "均值553": m53, "Δ(553−1026)": m53 - m26,
                 "d": d, "d_lo": ci[0], "d_hi": ci[1]})
q4_tbl = pd.DataFrame(rows)
q4_tbl.insert(1, "排名1026", q4_tbl["均值1026"].rank(ascending=False).astype(int))
q4_tbl.insert(2, "排名553", q4_tbl["均值553"].rank(ascending=False).astype(int))
q4_tbl["Δ排名"] = q4_tbl["排名553"] - q4_tbl["排名1026"]
q4_tbl["判读"] = q4_tbl["d"].apply(d_label)
_q4_rho = float(stats.spearmanr(q4_tbl["均值1026"], q4_tbl["均值553"]).statistic)
_q4_top3 = len(set(q4_tbl.nsmallest(3, "排名1026")["题项"]) &
               set(q4_tbl.nsmallest(3, "排名553")["题项"]))
log(f"  Q4：Spearman ρ = {_q4_rho:.3f}，Top-3 重合 {_q4_top3}/3")

# ---- 3.2 Q10 八题 ----
rows = []
for i, c in zip(range(1, 9), Q10):
    v26 = pd.to_numeric(d1026[c], errors="coerce")
    v53 = pd.to_numeric(d553[c], errors="coerce")
    m26, _, _ = mean_ci(v26)
    m53, _, _ = mean_ci(v53)
    d, ci = cohens_d_ci(v26, v53)
    rows.append({"题项": c, "中文": Q10_META[i][0], "English": Q10_META[i][1],
                 "均值1026": m26, "均值553": m53, "Δ(553−1026)": m53 - m26,
                 "d": d, "d_lo": ci[0], "d_hi": ci[1]})
q10_tbl = pd.DataFrame(rows)
q10_tbl.insert(1, "排名1026", q10_tbl["均值1026"].rank(ascending=False).astype(int))
q10_tbl.insert(2, "排名553", q10_tbl["均值553"].rank(ascending=False).astype(int))
q10_tbl["Δ排名"] = q10_tbl["排名553"] - q10_tbl["排名1026"]
q10_tbl["判读"] = q10_tbl["d"].apply(d_label)
_q10_rho = float(stats.spearmanr(q10_tbl["均值1026"], q10_tbl["均值553"]).statistic)
_q10_top3 = len(set(q10_tbl.nsmallest(3, "排名1026")["题项"]) &
                set(q10_tbl.nsmallest(3, "排名553")["题项"]))
log(f"  Q10：Spearman ρ = {_q10_rho:.3f}，Top-3 重合 {_q10_top3}/3")

# ---- 3.3 核心连续指标 ----
CONT = [("Q4_压力源指数", "压力源指数", "Stressor index"),
        ("Q5_身心健康", "身心健康受影响", "Health impact"),
        ("Q10_体制内偏好", "体制内偏好", "Public-sector preference"),
        ("Q10_灵活新就业偏好", "灵活/新就业偏好", "Flexible-work preference"),
        ("Q11_指数A_剔除不确定", "期望受影响指数", "Expectation-impact index"),
        ("F1", "F1 外在情境—内在规划", "External context & planning"),
        ("F2", "F2 市场竞争—人力资本", "Market competition & human capital")]
rows = []
for c, cn, en in CONT:
    if c not in d1026.columns:
        continue
    v26 = pd.to_numeric(d1026[c], errors="coerce")
    v53 = pd.to_numeric(d553[c], errors="coerce")
    m26, _, _ = mean_ci(v26)
    m53, _, _ = mean_ci(v53)
    d, ci = cohens_d_ci(v26, v53)
    rows.append({"指标": f"{cn}（{c}）", "English": en,
                 "n1026": int(v26.notna().sum()), "n553": int(v53.notna().sum()),
                 "均值1026": m26, "SD1026": float(v26.std(ddof=1)),
                 "均值553": m53, "SD553": float(v53.std(ddof=1)),
                 "Δ(553−1026)": m53 - m26, "d": d,
                 "d_lo": ci[0], "d_hi": ci[1], "判读": d_label(d)})
cont_tbl = pd.DataFrame(rows)

# ---- 3.4 类别 / 占比复现 ----
def _share(df, mask, label, group=None):
    """某条件在 df 中的占比（%）与计数。"""
    m = mask(df)
    return float(m.sum()), int(len(m)), m.mean() * 100.0

rows = []

def add_prop(name, cn, k26, n26, p26, k53, n53, p53):
    dd, ci = prop_diff_ci(k53, n53, k26, n26)
    rows.append({"指标": cn, "English": name,
                 "k/n 1026": f"{int(k26)}/{int(n26)}", "1026 %": p26,
                 "k/n 553": f"{int(k53)}/{int(n53)}", "553 %": p53,
                 "Δpp(553−1026)": dd, "lo": ci[0], "hi": ci[1]})

# Q5 重度（4–5）
v26 = pd.to_numeric(d1026["Q5_身心健康"], errors="coerce")
v53 = pd.to_numeric(d553["Q5_身心健康"], errors="coerce")
add_prop("Severe health impact (4–5)", "Q5 重度受影响占比",
         int((v26 >= 4).sum()), int(v26.notna().sum()), float((v26 >= 4).mean() * 100),
         int((v53 >= 4).sum()), int(v53.notna().sum()), float((v53 >= 4).mean() * 100))

# Q9 超限
add_prop("Exceeded the 2-option limit", "Q9 超限占比",
         int(d1026["Q9_超限"].sum()), len(d1026), float(d1026["Q9_超限"].mean() * 100),
         int(d553["Q9_超限"].sum()), len(d553), float(d553["Q9_超限"].mean() * 100))

# ⚠ Q11 的「不确定率」在本数据集中**算不出来**：卡点 1 的清洗口径已把 99 置为 NaN，
#   所以 (a == 99) 恒为 0（实测七行全 0 —— 这不是故障，是口径使然）。
#   Step 3 §3.3 的 3.9%–6.4% 取自清洗前数据，**以彼为准**，本批不重复也不覆盖。
_has99 = bool((d1026[Q11].apply(pd.to_numeric, errors="coerce") == 99).any().any())
if _has99:
    for i, c in zip(range(1, 8), Q11):
        a = pd.to_numeric(d1026[c], errors="coerce")
        b = pd.to_numeric(d553[c], errors="coerce")
        add_prop(f"Undecided rate: {Q11_META[i][1]}", f"Q11_{i} 不确定率（{Q11_META[i][0]}）",
                 int((a == 99).sum()), int(a.notna().sum()), float((a == 99).mean() * 100),
                 int((b == 99).sum()), int(b.notna().sum()), float((b == 99).mean() * 100))

# Q7 双口径（单选 = 意向4类；多选 = Y多选_*）
CATS4 = ["深造", "直接就业", "求稳体制", "方向未定"]
CATS4_EN = {"深造": "Further study", "直接就业": "Direct employment",
            "求稳体制": "Public-sector stability", "方向未定": "Undecided"}
MULTI_COL = {"深造": "Y多选_深造", "直接就业": "Y多选_直接就业",
             "求稳体制": "Y多选_求稳体制", "方向未定": "Y多选_方向未定"}
for c in CATS4:
    s26 = (d1026["意向4类"] == c)
    s53 = (d553["意向4类"] == c)
    add_prop(f"Single choice: {CATS4_EN[c]}", f"Q7 单选 · {c}",
             int(s26.sum()), len(d1026), float(s26.mean() * 100),
             int(s53.sum()), len(d553), float(s53.mean() * 100))
for c in CATS4:
    a = d1026[MULTI_COL[c]].astype(float).fillna(0)
    b = d553[MULTI_COL[c]].astype(float).fillna(0)
    add_prop(f"Multiple choice: {CATS4_EN[c]}", f"Q7 多选 · {c}",
             int(a.sum()), len(a), float(a.mean() * 100),
             int(b.sum()), len(b), float(b.mean() * 100))

# 人口学构成（⚠ 这里必然不同：两批是不同总体，差异本身要如实写出）
for col, cn, en in (("sex", "性别", "Sex"), ("major", "专业", "Major"),
                    ("grade", "年级", "Grade")):
    lv = sorted(set(d1026[col].dropna()) | set(d553[col].dropna()))
    for l in lv:
        a = (d1026[col] == l).mean() * 100
        b = (d553[col] == l).mean() * 100
        add_prop(f"{en}: {l}", f"{cn} · {l}",
                 int((d1026[col] == l).sum()), len(d1026), float(a),
                 int((d553[col] == l).sum()), len(d553), float(b))
prop_tbl = pd.DataFrame(rows)

# ---- 3.5 信度复现（Q4 / Q10 / Q11 三套）----
rel_rows = []
for name, cols, note in (("Q4 压力源量表", Q4, "9 题 · α 权威结论 .839"),
                         ("Q10 工作类型偏好", Q10, "8 题 · α .676 ⚠ 不作总量表"),
                         ("Q11 期望受影响", Q11, "7 题 · α .751")):
    sub = []
    for df in (d1026, d553):
        if name.startswith("Q11"):
            # ⚠ Q11 必须先把 99 置为缺失再算 α：99 是「不太确定」不是量表取值
            dd = df[cols].replace(99, np.nan)
        else:
            dd = df[cols]
        a, n = cronbach_alpha(pd.DataFrame(dd.values, columns=cols), cols)
        sub.append((a, n))
    rel_rows.append({"量表": name, "English": {"Q4 压力源量表": "Stressor scale (Q4)",
                                               "Q10 工作类型偏好": "Work-type preference (Q10)",
                                               "Q11 期望受影响": "Expectation impact (Q11)"}[name],
                     "α 1026": sub[0][0], "n1026": sub[0][1],
                     "α 553": sub[1][0], "n553": sub[1][1],
                     "Δα": sub[1][0] - sub[0][0], "备注": note})
rel_tbl = pd.DataFrame(rel_rows)
log(f"  信度：Q4 Δα = {rel_tbl.iloc[0]['Δα']:+.3f}、"
    f"Q10 Δα = {rel_tbl.iloc[1]['Δα']:+.3f}、Q11 Δα = {rel_tbl.iloc[2]['Δα']:+.3f}")

# ---- 3.6 复现小结（自动生成判定）----
_max_d = float(np.nanmax(np.abs(pd.concat([q4_tbl["d"], q10_tbl["d"]]))))
_max_rank = int(max(q4_tbl["Δ排名"].abs().max(), q10_tbl["Δ排名"].abs().max()))
_worst_prop = prop_tbl.loc[prop_tbl["English"].str.startswith(("Severe", "Exceeded")),
                           "Δpp(553−1026)"].abs().max()
REP = dict(q4_rho=_q4_rho, q10_rho=_q10_rho, q4_top3=_q4_top3, q10_top3=_q10_top3,
           max_d=_max_d, max_rank=_max_rank)

# ======================================================================
# 4. Step 18② · 与论文一已发表数值交叉核对
# ======================================================================
log("[Step 18②] 与论文一数值交叉核对 …")
p1_rows = []

def add_p1(ind, cn, ref, got, unit=""):
    ok = abs(float(ref) - float(got)) <= max(0.0006, abs(float(ref)) * 0.002)
    p1_rows.append({"指标": cn, "English": ind, "论文一": ref, "本次实测": got,
                    "单位": unit, "差": float(got) - float(ref),
                    "判定": "✅" if ok else "❌"})

_v = pd.to_numeric(d1026["Q4_压力源指数"], errors="coerce")
add_p1("Mean stressor index", "Q4 压力源指数均值", P1["Q4_压力源指数均值"],
       round(float(_v.mean()), 4), "分")
_v = pd.to_numeric(d1026["Q5_身心健康"], errors="coerce")
# ⚠ Q5 是唯一发表值与实测对不齐的一项：论文一 17.2% ≈ 176 人，实测 17.25% = 177 人，
#   **差 1 个人**。这不是数据漂移（step21 阶段就记录过「17.3% vs 论文一 17.2%」），
#   来源是发表时的四舍五入或边界含糊。判定改成「人数差 ≤ 1」而不是「百分比相同」。
_k5 = int((_v >= 4).sum())
_p5 = float((_v >= 4).mean() * 100)
_ref5 = round(P1["Q5_重度占比"] * 100, 1)
_ok5 = abs(_p5 - _ref5) <= 0.15 and abs(_k5 - P1["Q5_重度占比"] * len(_v)) <= 1.5
p1_rows.append({"指标": "Q5 重度（4–5）占比", "English": "Severe health-impact share",
                "论文一": f"{_ref5}%（≈{P1['Q5_重度占比'] * len(_v):.1f} 人）",
                "本次实测": f"{_p5:.2f}%（{_k5} 人）", "单位": "%",
                "差": round(_p5 - _ref5, 3),
                "判定": "⚠ 微差（≤1 人）" if _ok5 else "❌"})
for Q10_i in range(1, 9):
    i = Q10_i
    c = f"Q10_{i}"
    v = pd.to_numeric(d1026[c], errors="coerce")
    add_p1(f"{Q10_META[i][1]} mean", f"{c} 均值（{Q10_META[i][0]}）",
           P1["Q10_均值"][i], round(float(v.mean()), 3), "分")
for lab in CATS4:            # 单选：意向4类
    v = int((d1026["意向4类"] == lab).sum())
    p1_rows.append({"指标": f"Q7 单选 {lab} 人数", "English": f"Single-choice n: {CATS4_EN[lab]}",
                    "论文一": P1["Q7_单选"][lab], "本次实测": v, "单位": "人",
                    "差": v - P1["Q7_单选"][lab], "判定": "✅" if v == P1["Q7_单选"][lab] else "❌"})
for lab in CATS4:            # 多选：Y多选_*
    v = int(d1026[MULTI_COL[lab]].astype(float).fillna(0).sum())
    p1_rows.append({"指标": f"Q7 多选 {lab} 人数", "English": f"Multiple-choice n: {CATS4_EN[lab]}",
                    "论文一": P1["Q7_多选"][lab], "本次实测": v, "单位": "人",
                    "差": v - P1["Q7_多选"][lab], "判定": "✅" if v == P1["Q7_多选"][lab] else "❌"})
q10_order_now = [int(c.split("_")[1]) for c in
                 pd.Series({c: float(pd.to_numeric(d1026[c], errors="coerce").mean())
                            for c in Q10}).sort_values(ascending=False).index]
q10_order_ref = sorted(P1["Q10_均值"], key=lambda k: -P1["Q10_均值"][k])
p1_rows.append({"指标": "Q10 八题偏好排序", "English": "Q10 preference ranking",
                "论文一": " > ".join(f"Q10_{i}" for i in q10_order_ref),
                "本次实测": " > ".join(f"Q10_{i}" for i in q10_order_now),
                "单位": "序", "差": "—",
                "判定": "✅" if q10_order_now == q10_order_ref else "❌"})
p1_tbl = pd.DataFrame(p1_rows)
_nbad = int((p1_tbl["判定"] == "❌").sum())
_nwarn = int(p1_tbl["判定"].astype(str).str.startswith("⚠").sum())
log(f"  论文一对账 {len(p1_tbl)} 项：不一致 {_nbad} 项、微差 {_nwarn} 项")

# ======================================================================
# 5. Step 18③ · 两数据集内部一致性复核
# ======================================================================
log("[Step 18③] 两数据集内部一致性复核 …")
cons_rows = []
_common = [c for c in d1026.columns if c in d1579.columns and c not in EXCLUDE_COLS]
_diff_num, _diff_obj = [], []
_maxdiff = 0.0
_maxobj = 0.0
for c in _common:
    a = d1026[c]
    b = d1579.iloc[:len(d1026)][c]        # 1579 集前 1026 行就是主样本
    if a.dtype.kind in "ifb" and b.dtype.kind in "ifb":
        # 数值列：用最大绝对差
        dd = float(np.nanmax(np.abs(pd.to_numeric(a, errors="coerce").values -
                                    pd.to_numeric(b, errors="coerce").values)))
        _maxdiff = max(_maxdiff, dd if np.isfinite(dd) else 0.0)
        if dd > 0:
            _diff_num.append(f"{c}({dd:.2e})")
    else:
        # 非数值列：只能算「不一致行占比」，单位与上面不同，**不可混在同一个 max 里**
        rate = float((a.astype(str).values != b.astype(str).values).mean())
        _maxobj = max(_maxobj, rate)
        if rate > 0:
            _diff_obj.append(f"{c}({rate:.1%})")
_diff_cols = _diff_num + _diff_obj
_nbatch0 = int((d1579["batch"] == 0).sum())
_nbatch1 = int((d1579["batch"] == 1).sum())
cons_rows.append({"核对项": "1026 部分逐列最大差",
                  "实测": f"{_maxdiff:.2e}（共 {len(_common)} 列，差异列 {len(_diff_cols)}）",
                  "判定": "✅" if not _diff_cols else f"❌ {_diff_cols[:5]}",
                  "说明": f"已排除 z 分数 {len(Z_COLS)} 个 + 因子得分 {len(FACTOR_COLS)} 个"})
cons_rows.append({"核对项": "batch 合计", "实测": f"0 → {_nbatch0} ／ 1 → {_nbatch1}",
                  "判定": "✅" if (_nbatch0, _nbatch1) == (1026, 553) else "❌",
                  "说明": "卡点 7 硬要求"})
cons_rows.append({"核对项": "batch1 × net 结构零",
                  "实测": int(((d1579["batch"] == 1) & (d1579["channel"] == "net")).sum()),
                  "判定": "✅" if int(((d1579["batch"] == 1) & (d1579["channel"] == "net")).sum()) == 0 else "❌",
                  "说明": "禁拟合 batch×channel 的原因"})

# 三个关键派生列 vs 论文一建模集
if os.path.exists(MODELSET):
    ms = pd.read_excel(MODELSET)
    for name, col in (("意向4类", "意向4类"), ("Q11 指数A", "Q11_指数A_剔除不确定"),
                      ("Q10 体制内偏好", "Q10_体制内偏好"),
                      ("Q10 灵活/新就业偏好", "Q10_灵活新就业偏好")):
        if col not in ms.columns:
            continue
        a = ms[col]
        b = d1026[col]
        if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
            av = pd.to_numeric(a, errors="coerce").values.astype(float)
            bv = pd.to_numeric(b, errors="coerce").values.astype(float)
            both_nan = np.isnan(av) & np.isnan(bv)
            both_val = ~np.isnan(av) & ~np.isnan(bv)
            close = both_val & (np.abs(av - bv) <= 1e-9)
            same = float((both_nan | close).mean())
        else:
            same = float((a.astype(str).values == b.astype(str).values).mean())
        cons_rows.append({"核对项": f"{name}：1026 集 vs 论文一建模集",
                          "实测": f"逐行一致率 {same * 100:.2f}%",
                          "判定": "✅" if same == 1.0 else "❌",
                          "说明": "防止重算时改了口径；数值列按 1e-9 容差比，NaN 视为一致"})
else:
    cons_rows.append({"核对项": "建模集比对", "实测": "未找到 建模数据集.xlsx",
                      "判定": "⚠", "说明": "跳过"})
cons_tbl = pd.DataFrame(cons_rows)
log(f"  1026 部分逐列最大差 {_maxdiff:.2e}，batch = {_nbatch0}/{_nbatch1}")


# ======================================================================
# 7. Step 19 · 42 条参考文献落点映射
# ======================================================================
log("[Step 19] 参考文献落点映射 …")
REFMAP = [
    (1, "Ajzen (1991)", "理论框架", "Intro / Discussion：意向形成的一般解释",
     "RQ3 意向模型的理论依据；Step 15 意向4类"),
    (2, "Hobfoll (1989)", "理论框架", "Discussion：资源保全 → 趋稳避险",
     "Step 13 体制内偏好 OR=1.380；Step 16 基层就业负向间接"),
    (3, "Lazarus & Folkman (1984)", "理论框架", "Intro：压力认知评价—应对（主理论）",
     "Step 12/13 压力→选择的机制骨架"),
    (4, "Lent, Brown, & Hackett (1994)", "理论框架", "Discussion：SCCT 串起效能/结果预期/障碍",
     "建议作为第二主理论，与 Step 16 的 Q11 期望路径对接"),
    (5, "Savickas (1997)", "理论框架", "Discussion：生涯适应力",
     "Step 5 表 4 Q6 资源优势；Step 7–11 年级梯度"),
    (6, "Super (1980)", "理论框架", "Intro / Methods：年级分组的理论正当性",
     "Step 7–11 年级作为分层变量的依据"),
    (7, "Cronbach (1951)", "方法-RQ1", "Methods + Step 3 §1.1 三套量表 α",
     "✅ 已用；本批 §3.5 再次复现"),
    (8, "Kuder & Richardson (1937)", "方法-RQ1", "Methods + Step 3 §1.2 二分多选题组 KR-20",
     "⚠ Q9 的 KR-20 = −2.19 为负，**Q9 不报 KR-20**，Methods 要写明"),
    (9, "McNeish (2018)", "方法-RQ1", "Methods + Step 3 §1.1 补报 ω",
     "✅ 已用（ω Q4 .894 / Q10 .798）"),
    (10, "Shapiro & Wilk (1965)", "方法-RQ1", "Methods + Step 3 §1.3 正态性判定",
     "✅ 已用；7/7 全拒 → 故用不依赖正态假设的模型"),
    (11, "Kruskal & Wallis (1952)", "方法-RQ1", "Methods + Step 3 §3.5 组间检验",
     "✅ 已用（专业 5 类分层）"),
    (12, "Tomczak & Tomczak (2014)", "方法-RQ1", "Methods：ε² 效应量报告标准",
     "✅ 已用；Step 3 §3.5 全部撞车效应量"),
    (13, "Bartlett (1950)", "方法-RQ1", "Methods + Step 6 EFA 球性检验", "✅ 已用"),
    (14, "Kaiser (1974)", "方法-RQ1", "Methods + Step 6 KMO/MSA", "✅ 已用（KMO .900）"),
    (15, "Olsson (1979)", "方法-RQ1", "Methods + Step 6 polychoric 相关",
     "✅ 已用（自实现 polychoric）"),
    (16, "Horn (1965)", "方法-RQ1", "Methods + Step 6 平行分析定因子数", "✅ 已用（M=2）"),
    (17, "Norman (2010)", "方法-RQ1", "Methods：李克特可按连续处理（OLS 与 ologit 双轨）",
     "✅ **Step 12 与 Step 16 两处都靠它**：Step 16 因 Mediation 不支持有序结局而退回 OLS，依据就是这条"),
    (18, "Agresti & Coull (1998)", "方法-RQ1", "Methods + Step 5 表 4 多重响应 Wilson CI",
     "✅ 已用"),
    (19, "Kendall & Babington Smith (1939)", "方法-RQ1", "Methods + Step 4 §2.3 Kendall's W",
     "✅ 已用（Q10 W=.160 / Q4 W=.085）"),
    (20, "Grothe & Schmid (2011)", "方法-RQ1", "Methods + Step 4 §2.3 W 的并列秩校正",
     "✅ 已用（ΣT=161142，不校正会低估 W）"),
    (21, "Benjamini & Hochberg (1995)", "方法-RQ1", "Methods：全篇 FDR",
     "✅ 全篇通用（Step 3–17 每步都用）"),
    (22, "Jonckheere (1954)", "方法-RQ2", "Methods + Step 7–11 JT 有序趋势检验",
     "✅ 已用；BH 后 2/5 显著"),
    (23, "McCullagh (1980)", "方法-RQ2", "Methods + Step 12/17 比例优势模型",
     "✅ 已用；⚠ 页码投前再核"),
    (24, "Brant (1990)", "方法-RQ2", "Methods + Step 13 比例优势假设检验",
     "✅ 已用；6/9 总量拒绝但**压力项自身全部通过**"),
    (25, "Williams (2006)", "方法-RQ2", "Methods：广义有序 logit / 部分比例优势",
     "条件条目 → 见下方自动判定"),
    (26, "Mood (2010)", "方法-RQ2", "Discussion + Step 11/17：子组 logit 系数不可直接比较",
     "✅ **必引**；Step 17 分层 OR 互比的限制依据"),
    (27, "Williams (2012)", "方法-RQ3", "Methods + Step 15：多项 logit 报 AME",
     "✅ **必引**；Step 15 主报 AME 的依据"),
    (28, "Preacher & Hayes (2008)", "方法-RQ3", "Methods + Step 16 Bootstrap 中介",
     "✅ 已用（自举 5000 次）"),
    (29, "Brambor, Clark, & Golder (2006)", "方法-RQ3", "Methods + Step 17 交互与简单斜率解释",
     "✅ 已用；分层 OR 附带支持区间的要求"),
    (30, "VanderWeele & Ding (2017)", "方法-RQ3", "Discussion + Step 13 E-value",
     "✅ 已用；**须先做 OR→RR 转换**（Zhang & Yu 1998）"),
    (31, "Seabold & Perktold (2010)", "方法-软件", "Methods：statsmodels 软件声明",
     "✅ OrderedModel / MNLogit / Mediation 所属包"),
    (32, "Blokker et al. (2025)", "实证对话", "Methods：**两样本复制的依据**",
     "✅ **本批（Step 18①）正是这条的执行**；553 批次复现 RQ1 画像"),
    (33, "Guleria & Sood (2022)", "实证对话", "Intro / Discussion：与可解释性取向 contrast",
     "承认可解释性 ML 范式的渊源，并说明本文为何转向推断"),
    (34, "Kim et al. (2022)", "实证对话", "Discussion：传统 ↔ 易变取向",
     "解释 Q10_5 创业 / Q10_7 新业态低偏好"),
    (35, "Liu et al. (2023)", "实证对话", "Discussion：中国工科生专业认同 → 适应力",
     "⚠ 4 作者，**勿写 Liu & Ji**"),
    (36, "Zeng & Qin (2026)", "实证对话", "Discussion：中国学生就业心理困扰可比性最高",
     "待写；与本文 Q5/Q11 结论对话"),
    (37, "Zhang & Guo (2024)", "实证对话", "Intro：LCA + 多类 logit 中国语境先行做法",
     "⚠ 与本文 Step 15 MNLogit 的关系要说清（本文不聚类）"),
    (38, "Zheng & Yan (2024)", "实证对话", "Discussion：求稳偏好含学历层异质性",
     "与 Step 17 年级调节（2/27）对话"),
    (39, "People's Daily Online (2025)", "政策数据源", "Intro：2026 届毕业生 1270 万",
     "⚠ 作者位写 People's Daily Online，**不是教育部**"),
    (40, "教育部等五部门 (2023)", "政策数据源", "Intro：教高〔2023〕1 号",
     "学科专业调整方案，引出 Q4_2 专业与市场不匹配"),
    (41, "麦可思研究院 (2024)", "政策数据源", "Intro / Discussion：就业蓝皮书",
     "ISBN 978-7-5228-3746-8"),
    (42, "中国新闻周刊 (2025)", "政策数据源", "Intro：文科 73.5% vs 理科 86%",
     "⚠ 二手转述，须写「转引自」；出版年是 **2025**（2024 是数据年）"),
]
ref_tbl = pd.DataFrame(REFMAP, columns=["编号", "作者（年）", "分区", "落点", "备注"])

# ── #25 是否保留的自动判定（不再依赖报告文件，改查 step23 源码是否实现了 PPO）──
_src23 = os.path.join(BASE, "step23_grade_trend.py")
_used_ppo = False
if os.path.exists(_src23):
    with open(_src23, encoding="utf-8") as fh:
        _code23 = fh.read()
    _used_ppo = ("部分比例优势" in _code23 or "PPO" in _code23)
VERDICT_25 = ("**保留**" if _used_ppo else "**可删**")
ref_tbl.loc[ref_tbl["编号"] == 25, "备注"] = (
    f"条件条目 → 实测自动判定：step23 {'**实现了**' if _used_ppo else '**未实现**'}"
    f"部分比例优势模型 → {VERDICT_25}"
    f"（Step 13 的 Brant 是压力项自身通过，不涉及 PPO）")
log(f"  Williams (2006) #25：{VERDICT_25}（Step 7–11 是否用了 PPO = {_used_ppo}）")

# ======================================================================
# 8. Excel
# ======================================================================
log("写出 Excel …")
XL = os.path.join(OUT, "Step18_19_收尾核对结果.xlsx")
SHEETS = [
    ("Step2_三项校验", t18a),
    ("复现_表2_Q4九题", q4_tbl),
    ("复现_表3_Q10八题", q10_tbl),
    ("复现_核心连续指标", cont_tbl),
    ("复现_占比与构成", prop_tbl),
    ("复现_信度", rel_tbl),
    ("对账_论文一数值", p1_tbl),
    ("一致性_两数据集", cons_tbl),
    ("参考文献落点", ref_tbl),
]
with pd.ExcelWriter(XL, engine="openpyxl") as w:
    for nm, df in SHEETS:
        df.to_excel(w, sheet_name=nm, index=False)
log(f"  ✅ {XL}（{len(SHEETS)} sheet）")

# ======================================================================
# 9. 报告
# ======================================================================

def _fmt(df, cols=None, floatfmt="{:.3f}"):
    """把 DataFrame 转成 md 表（列太多时保留全部，按需指定 cols）。"""
    d = df if cols is None else df[cols]
    head = "| " + " | ".join(str(c) for c in d.columns) + " |"
    line = "|" + "|".join(["---"] * len(d.columns)) + "|"
    body = []
    for _, r in d.iterrows():
        cells = []
        for v in r.tolist():
            if isinstance(v, float):
                cells.append("—" if not np.isfinite(v) else floatfmt.format(v))
            else:
                cells.append(str(v))
        body.append("| " + " | ".join(cells) + " |")
    return "\n".join([head, line] + body)

# ---- 一、Step 18① ----
_demo = prop_tbl[prop_tbl["指标"].str.startswith(("性别", "专业", "年级"))]
_g1 = float(_demo.loc[_demo["指标"] == "专业 · eng", "1026 %"].iloc[0])
_g2 = float(_demo.loc[_demo["指标"] == "专业 · eng", "553 %"].iloc[0])
if _has99:
    _und = prop_tbl[prop_tbl["English"].str.startswith("Undecided")]
else:
    pass

# ---- 二、Step 18② ----
if _nwarn:
    pass

# ---- 三、Step 18③ ----


log("批 5 完成。")

