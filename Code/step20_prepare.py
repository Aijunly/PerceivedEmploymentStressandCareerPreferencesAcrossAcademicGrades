# -*- coding: utf-8 -*-
"""
论文二 · Step 1  双数据底座组装  (step20_prepare.py)

输入
  合并问卷清洗数据_1579.xlsx   1579 x 80   （顺序：纸质779 + 网络1 247 + 纸质2 553）
  建模数据集.xlsx              1026 x 74   （与上式前 1026 行逐行对齐，已实测）

输出
  论文二数据集_1026.xlsx       RQ1 描述画像、RQ2 年级梯度主分析
  论文二数据集_1579.xlsx       RQ3 压力→选择、RQ2 的 pooled 子分析
  论文二_缺失诊断.xlsx         两个数据集各一个 sheet

口径依据：论文二_分析步骤.md 第 0 部分 卡点 1（Q11 分母）与 卡点 7（合并口径）
解释器：E:/Python/python.exe
"""
import sys as _sys
if hasattr(_sys.stdout, "reconfigure"):
    _sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np
import pandas as pd
from pathlib import Path

BASE = Path(__file__).resolve().parent
OUTDIR = BASE / "论文二_输出"
OUTDIR.mkdir(exist_ok=True)

# ----------------------------------------------------------------------
# 常量
# ----------------------------------------------------------------------
Q4_ITEMS = [f"Q4_{i}" for i in range(1, 10)]
Q6_ITEMS = [f"Q6_{i}" for i in range(1, 10)]
Q7_ITEMS = [f"Q7_{i}" for i in range(1, 9)]
Q8_ITEMS = [f"Q8_{i}" for i in range(1, 10)]
Q9_ITEMS = [f"Q9_{i}" for i in range(1, 8)]
Q10_ITEMS = [f"Q10_{i}" for i in range(1, 9)]
Q11_ITEMS = [f"Q11_{i}" for i in range(1, 8)]

# Q3 专业 11 档 -> 5 档（pooled 中 n<50 的稀有类并入 oth）
MAJOR_MAP = {1: "eng", 2: "sci", 6: "econ", 5: "hum",
             3: "oth", 4: "oth", 7: "oth", 8: "oth", 9: "oth", 10: "oth", 11: "oth"}
GRADE_MAP = {1: "y1", 2: "y2", 3: "y3", 4: "y4", 5: "grad"}
SEX_MAP = {1: "male", 2: "female"}

# Q7 多选 -> 意向 4 类（与 step12_dataset_label.py 完全一致）
CAT_OF_OPT = {2: "深造", 3: "深造", 4: "求稳体制", 6: "求稳体制",
              1: "直接就业", 5: "直接就业", 7: "直接就业", 8: "方向未定"}
PRIORITY = [2, 3, 4, 6, 1, 5, 7, 8]          # 考研>出国>考公>参军>就业>创业>灵活>慢就业
OPTS_OF_CAT = {"深造": [2, 3], "求稳体制": [4, 6], "直接就业": [1, 5, 7], "方向未定": [8]}
INTENT_CODE = {"深造": 0, "直接就业": 1, "求稳体制": 2, "方向未定": 3}


def make_label(row_vals):
    picked = [i + 1 for i in range(8) if row_vals[i] == 1]
    if not picked:
        return np.nan, np.nan, 0
    for opt in PRIORITY:
        if opt in picked:
            return CAT_OF_OPT[opt], opt, len(picked)
    return np.nan, np.nan, len(picked)


def zscore(s):
    return (s - s.mean()) / s.std(ddof=1)


# ----------------------------------------------------------------------
# 1. 读入
# ----------------------------------------------------------------------
print("=" * 72)
print("论文二 Step 1 · 双数据底座组装")
print("=" * 72)

df = pd.read_excel(BASE / "合并问卷清洗数据_1579.xlsx")
model = pd.read_excel(BASE / "建模数据集.xlsx")
print(f"读入 1579 集 {df.shape}   建模集 {model.shape}")

# ----------------------------------------------------------------------
# 2. ⚠ Q11 的 99 必须先转缺失（卡点 7 技术坑）
# ----------------------------------------------------------------------
n99 = int((df[Q11_ITEMS] == 99).sum().sum())
df[Q11_ITEMS] = df[Q11_ITEMS].replace(99, np.nan)
print(f"\n[1] Q11 的 99 编码 -> NaN：共替换 {n99} 个单元格")

# ----------------------------------------------------------------------
# 3. 重建 意向4类（1579 全量，与 step12 同规则）
# ----------------------------------------------------------------------
q7 = df[Q7_ITEMS].fillna(0).astype(int).values
lab = [make_label(r) for r in q7]
df["意向4类"] = [x[0] for x in lab]
df["意向_取首选项"] = [x[1] for x in lab]
df["Q7_勾选数"] = [x[2] for x in lab]
df["意向4类_编码"] = df["意向4类"].map(INTENT_CODE)
for cat, opts in OPTS_OF_CAT.items():
    df[f"Y多选_{cat}"] = df[[f"Q7_{o}" for o in opts]].max(axis=1)
print("[2] 意向4类 重建完成（1579 全量，无 Q7 未作答）")

# ----------------------------------------------------------------------
# 4. 批次与渠道（二选一体系：channel + batch，丢弃三分类来源）
# ----------------------------------------------------------------------
df["batch"] = (df["来源"] == "纸质2").astype(int)
df["channel"] = np.where(df["来源"] == "网络1", "net", "paper")
src_raw = df["来源"].copy()
df = df.drop(columns=["来源"])
print(f"[3] batch 0={(df['batch']==0).sum()} / 1={(df['batch']==1).sum()}；"
      f"channel paper={(df['channel']=='paper').sum()} / net={(df['channel']=='net').sum()}")

# ----------------------------------------------------------------------
# 5. 分类变量 5 档化与英文短码
# ----------------------------------------------------------------------
df["major"] = df["Q3_专业"].map(MAJOR_MAP)
df["grade"] = df["Q2_年级"].map(GRADE_MAP)
df["sex"] = df["Q1_性别"].map(SEX_MAP)
df["年级_合并毕业"] = df["grade"].map({"y4": "grad_group", "grad": "grad_group"}).fillna(df["grade"])
print("[4] 专业 11->5 档、年级/性别英文短码完成")

# ----------------------------------------------------------------------
# 6. 派生变量
# ----------------------------------------------------------------------
df["Q11_缺失数"] = df[Q11_ITEMS].isna().sum(axis=1)
df["Q11_有效题数"] = 7 - df["Q11_缺失数"]
df["Q11_指数A_剔除不确定"] = df[Q11_ITEMS].mean(axis=1)      # NaN 自动跳过
df["Q10_体制内偏好"] = df["Q10_1"]                            # 单题，不是 mean(Q10_1,Q10_2)
df["Q10_灵活新就业偏好"] = df[["Q10_5", "Q10_6", "Q10_7"]].mean(axis=1)
print("[5] Q11 指数 / 缺失数 / Q10 两个复合偏好 构造完成")

# z 分数：各数据集按自身 SD 标准化（此处按 1579 全集算，切出的 1026 集单独重算）
for col, tag in [("Q4_压力源指数", "压力指数"), ("Q5_身心健康", "身心健康"),
                 ("Q10_体制内偏好", "体制内偏好"), ("Q10_灵活新就业偏好", "灵活偏好"),
                 ("Q11_指数A_剔除不确定", "期望受影响")]:
    df[f"z_{tag}"] = zscore(df[col])

# ----------------------------------------------------------------------
# 7. 切出 1026 集（前 1026 行，与建模集逐行对齐）
# ----------------------------------------------------------------------
keep_cols = ([
    "id", "batch", "channel", "sex", "grade", "major", "年级_合并毕业",
    "Q1_性别", "Q2_年级", "Q3_专业",
] + Q4_ITEMS + ["Q4_压力源指数"] + ["Q5_身心健康"]
  + Q6_ITEMS + ["Q6_选中数"]
  + Q7_ITEMS + ["Q7_勾选数", "意向4类", "意向4类_编码", "意向_取首选项"]
  + [f"Y多选_{c}" for c in OPTS_OF_CAT]
  + Q8_ITEMS + ["Q8_选中数"]
  + Q9_ITEMS + ["Q9_选中数", "Q9_超限"]
  + Q10_ITEMS + ["Q10_体制内偏好", "Q10_灵活新就业偏好"]
  + Q11_ITEMS + ["Q11_指数A_剔除不确定", "Q11_缺失数", "Q11_有效题数"]
  + ["z_压力指数", "z_身心健康", "z_体制内偏好", "z_灵活偏好", "z_期望受影响"]
  + ["所用时间_秒"])

keep_cols = [c for c in keep_cols if c in df.columns]
d1579 = df[keep_cols].reset_index(drop=True)
# 去标识化：按行重新编号，保证 id 在 1579 与 1026 两个文件中唯一且同一人同号
d1579["id"] = np.arange(1, len(d1579) + 1)
d1026 = d1579.iloc[:1026].reset_index(drop=True).copy()

# 1026 集内按自身 SD 重算 z 分数
for col, tag in [("Q4_压力源指数", "压力指数"), ("Q5_身心健康", "身心健康"),
                 ("Q10_体制内偏好", "体制内偏好"), ("Q10_灵活新就业偏好", "灵活偏好"),
                 ("Q11_指数A_剔除不确定", "期望受影响")]:
    d1026[f"z_{tag}"] = zscore(d1026[col])
print("[6] 切出 1026 集，z 分数按各自 SD 重算")

# ----------------------------------------------------------------------
# 8. 验收
# ----------------------------------------------------------------------
print("\n" + "=" * 72)
print("验收")
print("=" * 72)

chk = {}

# (a) 意向4类在 1026 部分与建模集逐行一致
same = (d1026["意向4类"].values == model["意向4类"].values).all()
vc = d1579["意向4类"].value_counts()
chk["a_1026意向4类与建模集一致"] = bool(same)
print(f"(a) 1026 部分 意向4类 与建模集逐行一致：{same}")
print(f"    1579 分布：{ {k: int(v) for k, v in vc.items()} }")
print(f"    1026 分布：{ {k: int(v) for k, v in d1026['意向4类'].value_counts().items()} }")
print(f"    553  分布：{ {k: int(v) for k, v in d1579.iloc[1026:]['意向4类'].value_counts().items()} }")

# (b) Q11 指数与建模集一致
d1 = (d1026["Q11_指数A_剔除不确定"] - model["Q11_指数A_剔除不确定"]).abs().max()
chk["b_Q11指数一致_maxdiff"] = float(d1)
print(f"(b) Q11 指数A 与建模集最大绝对差：{d1:.2e}（应为 0）")

# (c) Q10 复合偏好一致
d2 = (d1026["Q10_体制内偏好"] - model["Q10_体制内偏好"]).abs().max()
d3 = (d1026["Q10_灵活新就业偏好"] - model["Q10_灵活新就业偏好"]).abs().max()
chk["c_Q10两偏好_maxdiff"] = (float(d2), float(d3))
print(f"(c) Q10 体制内/灵活偏好 与建模集最大绝对差：{d2:.2e} / {d3:.2e}（应为 0）")

# (d) Q11 三档 N
miss1026 = d1026[Q11_ITEMS].isna().sum(axis=1)
miss1579 = d1579[Q11_ITEMS].isna().sum(axis=1)
n3 = {"1579": [int((miss1579 <= 6).sum()), int((miss1579 <= 2).sum()), int((miss1579 == 0).sum())],
      "1026": [int((miss1026 <= 6).sum()), int((miss1026 <= 2).sum()), int((miss1026 == 0).sum())]}
chk["d_Q11三档N"] = n3
print(f"(d) Q11 三档 N（≥1题 / ≥5题 / 7题）：1579 = {n3['1579']}，1026 = {n3['1026']}")

# (e) 批次与核心均值
chk["e_batch合计"] = [int((d1579['batch'] == 0).sum()), int((d1579['batch'] == 1).sum())]
m_main = d1026["Q4_压力源指数"].mean()
m_2nd = d1579.iloc[1026:]["Q4_压力源指数"].mean()
m_all = d1579["Q4_压力源指数"].mean()
chk["e_Q4均值"] = [round(m_main, 4), round(m_2nd, 4), round(m_all, 4)]
print(f"(e) batch 合计 {chk['e_batch合计']}；Q4 均值 主 {m_main:.4f} / 二 {m_2nd:.4f} / 合 {m_all:.4f}")

# (f) 专业 5 档与年级 5 档
mv = d1579["major"].value_counts()
gv = d1579["grade"].value_counts()
chk["f_major5"] = {k: int(v) for k, v in mv.items()}
chk["f_grade5"] = {k: int(v) for k, v in gv.items()}
print(f"(f) 专业 5 档：{chk['f_major5']}")
print(f"    年级 5 档：{chk['f_grade5']}")

# (g) 结构零与共线体检
ct = pd.crosstab(d1579["batch"], d1579["channel"])
chk["g_结构零"] = int(ct.loc[1, "net"]) == 0 if "net" in ct.columns else True
print(f"(g) batch×channel 交叉表 net 格 = {int(ct.loc[1, 'net'])}（应为 0，即结构零成立）")

# (h) 两集在 1026 行上一致（z 分数按设计各自标准化，排除）
zonly = [c for c in d1026.columns if c.startswith("z_")]
shared = [c for c in d1026.columns if c in d1579.columns and c not in zonly]
bad = []
for c in shared:
    x1 = d1026[c].fillna(-999).astype(str).values
    x2 = d1579[c].iloc[:1026].fillna(-999).astype(str).values
    if not (x1 == x2).all():
        bad.append(c)
chk["h_两集1026行差异列"] = bad
print(f"(h) 两集在 1026 行上的差异列：{bad if bad else '无'}")
print(f"    注：z 分数 5 列按设计各集自身 SD 标准化，已排除在一致性核对之外；"
      f"1026 集 batch 全为 0（{int((d1026['batch']==0).sum())}/1026）")

# ----------------------------------------------------------------------
# 9. 缺失诊断
# ----------------------------------------------------------------------
diag_cols = (["Q1_性别", "Q2_年级", "Q3_专业"] + Q4_ITEMS + ["Q4_压力源指数", "Q5_身心健康"]
             + Q6_ITEMS + Q7_ITEMS + Q8_ITEMS + Q9_ITEMS + Q10_ITEMS + Q11_ITEMS
             + ["Q11_指数A_剔除不确定", "Q10_体制内偏好", "Q10_灵活新就业偏好", "意向4类"])
diag_cols = [c for c in diag_cols if c in d1579.columns]


def diagnose(d, cols, name):
    rows = []
    for c in cols:
        s = d[c]
        nn = int(s.notna().sum())
        is_num = pd.api.types.is_numeric_dtype(s)
        rows.append({
            "变量": c, "N": nn, "缺失数": int(s.isna().sum()),
            "缺失率%": round(100 * s.isna().mean(), 2),
            "最小值": (round(float(s.min()), 3) if is_num and nn > 0 else ""),
            "最大值": (round(float(s.max()), 3) if is_num and nn > 0 else ""),
            "均值": (round(float(s.mean()), 4) if is_num and nn > 0 else ""),
            "取值集合": (",".join(sorted(map(str, s.dropna().unique()))[:8])
                       if (not is_num) or s.dropna().nunique() <= 8 else ""),
        })
    out = pd.DataFrame(rows)
    out.insert(0, "数据集", name)
    return out


diag_all = pd.concat([diagnose(d1026, diag_cols, "1026"),
                      diagnose(d1579, diag_cols, "1579")], ignore_index=True)

# ----------------------------------------------------------------------
# 10. 写出
# ----------------------------------------------------------------------
p1 = BASE / "论文二数据集_1026.xlsx"
p2 = BASE / "论文二数据集_1579.xlsx"
p3 = BASE / "论文二_缺失诊断.xlsx"
d1026.to_excel(p1, index=False)
d1579.to_excel(p2, index=False)
with pd.ExcelWriter(p3, engine="openpyxl") as w:
    diagnose(d1026, diag_cols, "1026").to_excel(w, sheet_name="缺失诊断_1026", index=False)
    diagnose(d1579, diag_cols, "1579").to_excel(w, sheet_name="缺失诊断_1579", index=False)
    pd.crosstab(d1579["batch"], d1579["channel"]).to_excel(w, sheet_name="批次x渠道")
    pd.crosstab(d1579["grade"], d1579["major"]).to_excel(w, sheet_name="年级x专业")
    pd.crosstab(d1579["batch"], d1579["grade"]).to_excel(w, sheet_name="批次x年级")
    pd.crosstab(d1579["batch"], d1579["major"]).to_excel(w, sheet_name="批次x专业")

print(f"\n[7] 已写出：\n    {p1}  {d1026.shape}\n    {p2}  {d1579.shape}\n    {p3}  (6 sheet)")

# ----------------------------------------------------------------------
# 11. 验收报告
# ----------------------------------------------------------------------
r1026 = d1026["意向4类"].value_counts()
r1579 = d1579["意向4类"].value_counts()
r553 = d1579.iloc[1026:]["意向4类"].value_counts()
for nm, r, tot in [("1026", r1026, 1026), ("553", r553, 553), ("1579", r1579, 1579)]:
    pass
print("\n完成。")
