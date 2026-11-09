# -*- coding: utf-8 -*-
"""
论文二 · Chow / pooling 检验（正式版）
判断 1026 主样本 与 553 第二批 能否合并用于 RQ3（压力 → 就业选择）。

三级嵌套模型：
    M0  Y = X + Z                 完全合并
    M1  M0 + B                    放开批次截距差 δ
    M2  M1 + B:X                  放开批次斜率差 θ   ← 决定性的一步

判定：
    θ 显著            → 禁止合并（压力的作用方式在两批不同）
    θ 不显著 δ 显著   → 可合并，但须保留批次哑变量
    两者均不显著      → 完全可合并

输出：
    论文二_输出/Chow_pooling检验结果.xlsx
    figs2/fig_RQ3_pooling_theta_forest.png
"""
import sys as _sys
if hasattr(_sys.stdout, "reconfigure"):
    _sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.diagnostic import het_breuschpagan
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "论文二_输出")
FIG = os.path.join(HERE, "figs2")
os.makedirs(OUT, exist_ok=True)
os.makedirs(FIG, exist_ok=True)

plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["savefig.facecolor"] = "white"

GRADE = {1: "y1", 2: "y2", 3: "y3", 4: "y4", 5: "grad"}
SEX = {1: "male", 2: "female"}
MAJOR_RAW = {1: "eng", 2: "sci", 3: "med", 4: "agri", 5: "hum",
             6: "econ", 7: "law", 8: "edu", 9: "art", 10: "pe", 11: "oth"}

Q7_COLS = [f"Q7_{i}" for i in range(1, 9)]
CAT_OF_OPT = {2: "深造", 3: "深造", 4: "求稳体制", 6: "求稳体制",
              1: "直接就业", 5: "直接就业", 7: "直接就业", 8: "方向未定"}
PRIORITY = [2, 3, 4, 6, 1, 5, 7, 8]


# ----------------------------------------------------------------------
# 1. 数据组装
# ----------------------------------------------------------------------
def make_label(row):
    picked = [i + 1 for i in range(8) if row[Q7_COLS[i]] == 1]
    if not picked:
        return np.nan
    for opt in PRIORITY:
        if opt in picked:
            return CAT_OF_OPT[opt]
    return np.nan


def load():
    main = pd.read_excel(os.path.join(HERE, "建模数据集.xlsx"))
    second = pd.read_excel(os.path.join(HERE, "网络问卷2_清洗数据.xlsx"))

    main = main.copy()
    second = second.copy()
    main["batch"] = 0
    second["batch"] = 1

    # 第二批没有「意向4类」，用与 step12_dataset_label.py 完全一致的规则重建
    second["意向4类"] = second.apply(make_label, axis=1)

    # 采集渠道：第二批全部为纸质（文件名带「网络」实为纸质）
    main["channel"] = main["来源"].map({"纸质": "paper", "网络": "net"}).fillna("paper")
    second["channel"] = "paper"

    keep = ["Q1_性别", "Q2_年级", "Q3_专业", "Q4_压力源指数", "Q5_身心健康",
            "Q10_体制内偏好", "Q10_灵活新就业偏好", "Q11_指数A_剔除不确定",
            "意向4类", "Q9_超限", "batch", "channel"]
    cols = {c: c for c in keep}
    a = main[[c for c in keep if c in main.columns]].rename(columns=cols)
    b = second[[c for c in keep if c in second.columns]].rename(columns=cols)
    df = pd.concat([a, b], ignore_index=True, sort=False)

    df["sex"] = df["Q1_性别"].map(SEX).fillna("unk")
    df["grade"] = df["Q2_年级"].map(GRADE).fillna("unk")
    df["major"] = df["Q3_专业"].map(MAJOR_RAW).fillna("oth")
    return df


def collapse_rare_major(df, min_n=50):
    """把 pooled 样本中人数 < min_n 的专业并入 oth，避免单例类别拉高方差。"""
    vc = df["major"].value_counts()
    rare = set(vc[vc < min_n].index)
    df = df.copy()
    df["major"] = df["major"].apply(lambda x: "oth" if x in rare else x)
    return df, sorted(rare)


def zscore(s):
    return (s - s.mean()) / s.std(ddof=1)


# ----------------------------------------------------------------------
# 2. 核心：三级嵌套模型
# ----------------------------------------------------------------------
COV = "C(sex) + C(grade) + C(major) + C(channel)"


def run_ols(df, ycol, xcol="stress"):
    d = df[[ycol, xcol, "sex", "grade", "major", "channel", "batch"]].dropna()
    f0 = f"{ycol} ~ {xcol} + {COV}"
    f1 = f"{ycol} ~ {xcol} + {COV} + batch"
    f2 = f"{ycol} ~ {xcol} + {COV} + batch + batch:{xcol}"

    m0 = smf.ols(f0, data=d).fit()
    m1 = smf.ols(f1, data=d).fit()
    m2 = smf.ols(f2, data=d).fit()

    f1stat, p_delta_classic, _ = m1.compare_f_test(m0)
    f2stat, p_theta_classic, _ = m2.compare_f_test(m1)

    # 异方差稳健版（Chow 的经典 F 假定同方差，这里以 HC3 Wald 为主结论）
    m2h = smf.ols(f2, data=d).fit(cov_type="HC3")
    m1h = smf.ols(f1, data=d).fit(cov_type="HC3")
    theta_name = [p for p in m2h.params.index if ":" in p and "batch" in p][0]
    delta_name = [p for p in m1h.params.index if p.startswith("batch")][0]

    wt_theta = m2h.wald_test(f"{theta_name} = 0", use_f=True)
    wt_delta = m1h.wald_test(f"{delta_name} = 0", use_f=True)

    ci_t = m2h.conf_int().loc[theta_name]
    ci_d = m1h.conf_int().loc[delta_name]

    # Breusch-Pagan 异方差诊断（在 M2 上）
    bp = het_breuschpagan(m2.resid, m2.model.exog)
    f_stat = m2.fvalue

    # 分样本斜率（各自内部回归，供直观对照）
    slopes = {}
    for bval, tag in [(0, "slope_main"), (1, "slope_second")]:
        sub = d[d["batch"] == bval]
        fit = smf.ols(f"{ycol} ~ {xcol} + {COV}", data=sub).fit()
        slopes[tag] = fit.params[xcol]

    return {
        "结局": ycol,
        "N": int(len(d)),
        "n_main": int((d["batch"] == 0).sum()),
        "n_second": int((d["batch"] == 1).sum()),
        "delta": round(m1h.params[delta_name], 4),
        "delta_CI": f"[{ci_d[0]:.3f}, {ci_d[1]:.3f}]",
        "p_delta_classic": p_delta_classic,
        "p_delta_HC3": float(wt_delta.pvalue),
        "theta": round(m2h.params[theta_name], 4),
        "theta_CI": f"[{ci_t[0]:.3f}, {ci_t[1]:.3f}]",
        "theta_lo": round(ci_t[0], 4),
        "theta_hi": round(ci_t[1], 4),
        "p_theta_classic": p_theta_classic,
        "p_theta_HC3": float(wt_theta.pvalue),
        "F_theta_classic": float(f2stat),
        "BP_p": bp[1],
        "slope_main": round(slopes["slope_main"], 4),
        "slope_second": round(slopes["slope_second"], 4),
        "slope_pooled": round(m2h.params[xcol], 4),
        "R2_M2": round(m2.rsquared, 4),
    }


CODE_MAP = {"深造": 0, "直接就业": 1, "求稳体制": 2, "方向未定": 3}


def run_mnlogit(df, ycol="意向4类", xcol="stress"):
    d = df[[ycol, xcol, "sex", "grade", "major", "channel", "batch"]].dropna()
    d = d.copy()
    d["cat"] = d[ycol].map(CODE_MAP)          # MNLogit 要求 endog 为数值
    d = d.dropna(subset=["cat"])
    d["cat"] = d["cat"].astype(int)
    print(f"  类别分布: {d['cat'].value_counts().sort_index().to_dict()} "
          f"(0=深造 1=直接就业 2=求稳体制 3=方向未定)")
    for bval, tag in [(0, "主"), (1, "二")]:
        v = d.loc[d["batch"] == bval, "cat"].value_counts().sort_index().to_dict()
        print(f"    {tag}样本: {v}")
    f0 = f"cat ~ {xcol} + {COV}"
    f1 = f"cat ~ {xcol} + {COV} + batch"
    f2 = f"cat ~ {xcol} + {COV} + batch + batch:{xcol}"
    kw = dict(disp=0, maxiter=1000, method="bfgs")
    m0 = sm.MNLogit.from_formula(f0, data=d).fit(**kw)
    m1 = sm.MNLogit.from_formula(f1, data=d).fit(**kw)
    m2 = sm.MNLogit.from_formula(f2, data=d).fit(**kw)
    for nm, m in [("M0", m0), ("M1", m1), ("M2", m2)]:
        ok = bool(m.mle_retvals.get("converged", False))
        print(f"    {nm} 收敛={ok}  logL={m.llf:.3f}")
        if not ok:
            print(f"      ⚠ {nm} 未收敛，LR 结果不可靠")

    def lr(ru, rr):
        stat = 2 * (ru.llf - rr.llf)
        # ⚠ MNLogit.params 形状为 (特征数, 类别数-1)，len() 只给特征数，
        #   必须用 .size 才是真实参数个数
        dof = int(ru.params.size - rr.params.size)
        return stat, dof, stats.chi2.sf(stat, dof)

    lr1 = lr(m1, m0)
    lr2 = lr(m2, m1)
    return {
        "结局": ycol + "（多项 logit，LR 检验）",
        "N": int(len(d)),
        "test": "LR",
        "p_delta_classic": lr1[2],
        "p_theta_classic": lr2[2],
        "chi2_theta": float(lr2[0]),
        "df_theta": int(lr2[1]),
        "note": f"LR δ={lr1[0]:.2f}(df={lr1[1]}), LR θ={lr2[0]:.2f}(df={lr2[1]})",
    }


# ----------------------------------------------------------------------
# 3. 主流程
# ----------------------------------------------------------------------
def main():
    print("=" * 78)
    print("论文二 · Chow / pooling 检验：1026 主样本 vs 553 第二批")
    print("=" * 78)

    df = load()
    print(f"\n合并后 N = {len(df)}  (主 {int((df['batch']==0).sum())} + 二 {int((df['batch']==1).sum())})")

    df, rare = collapse_rare_major(df, min_n=50)
    if rare:
        print(f"稀有专业已并入 oth: {rare}")

    df["stress"] = zscore(df["Q4_压力源指数"])
    print(f"stress 标准化: mean={df['stress'].mean():.3f}  sd={df['stress'].std(ddof=1):.3f}")

    ycols = ["Q10_体制内偏好", "Q10_灵活新就业偏好",
             "Q11_指数A_剔除不确定", "Q5_身心健康"]

    rows = []
    for y in ycols:
        print(f"\n--- {y} ---")
        r = run_ols(df, y)
        print(f"  δ = {r['delta']:+.4f}  95%CI {r['delta_CI']}   p_classic={r['p_delta_classic']:.4f}  p_HC3={r['p_delta_HC3']:.4f}")
        print(f"  θ = {r['theta']:+.4f}  95%CI {r['theta_CI']}   p_classic={r['p_theta_classic']:.4f}  p_HC3={r['p_theta_HC3']:.4f}")
        print(f"  分样本斜率: 主={r['slope_main']:+.4f}  二={r['slope_second']:+.4f}  pooled={r['slope_pooled']:+.4f}")
        print(f"  Breusch-Pagan p={r['BP_p']:.4f}  R²(M2)={r['R2_M2']}")
        rows.append(r)

    print("\n--- 意向4类（多分类结局，改用似然比检验）---")
    r_cat = run_mnlogit(df)
    print(f"  {r_cat['note']}")
    print(f"  p_δ={r_cat['p_delta_classic']:.4f}   p_θ={r_cat['p_theta_classic']:.4f}")

    # ---- BH 校正（以 θ 的检验为关注对象）----
    res = pd.DataFrame(rows)
    pvals = list(res["p_theta_HC3"]) + [r_cat["p_theta_classic"]]
    rej, padj, _, _ = multipletests(pvals, alpha=0.05, method="fdr_bh")
    res["p_theta_BH"] = padj[:len(res)]
    print("\n=== BH 校正后（θ 检验，'批次×压力'交互）===")
    for i, r in res.iterrows():
        verdict = "θ 显著→禁止合并" if padj[i] < 0.05 else "θ 不显著→可合并"
        flag = "" if r["p_delta_HC3"] < 0.05 else "（δ 亦不显著，完全可合并）"
        print(f"  {r['结局']:22s} θ={r['theta']:+.4f} CI{r['theta_CI']}  p_adj={padj[i]:.4f}  {verdict}{flag}")
    print(f"  {r_cat['结局']:22s} chi2={r_cat['chi2_theta']:.2f} df={r_cat['df_theta']}  p_adj={padj[-1]:.4f}  "
          f"{'θ 显著→禁止合并' if padj[-1] < 0.05 else 'θ 不显著→可合并'}")

    # ---- 敏感性 S1：剔除 Q9 超限 ----
    print("\n" + "=" * 78)
    print("敏感性 S1：剔除 Q9 超限样本（主 98 + 二 61）")
    print("=" * 78)
    df_s = df[df["Q9_超限"] != 1].copy()
    print(f"  N = {len(df_s)}")
    rows_s = []
    for y in ycols:
        r = run_ols(df_s, y)
        rows_s.append(r)
        print(f"  {y:22s} δ={r['delta']:+.4f} p={r['p_delta_HC3']:.4f} | "
              f"θ={r['theta']:+.4f} CI{r['theta_CI']} p={r['p_theta_HC3']:.4f}")
    res_s = pd.DataFrame(rows_s)

    # ---- 重点子群：研究生·工学的 Q5 ----
    print("\n" + "=" * 78)
    print("重点关注：研究生·工学 子群的 Q5 身心健康")
    print("=" * 78)
    sub = df[(df["grade"] == "grad") & (df["major"] == "eng")]
    print(f"  N = {len(sub)} (主 {int((sub['batch']==0).sum())} + 二 {int((sub['batch']==1).sum())})")
    if len(sub) > 30:
        r_sub = run_ols(sub, "Q5_身心健康")
        print(f"  δ={r_sub['delta']:+.4f} p={r_sub['p_delta_HC3']:.4f} | "
              f"θ={r_sub['theta']:+.4f} CI{r_sub['theta_CI']} p={r_sub['p_theta_HC3']:.4f}")
    else:
        print("  样本过小，跳过")

    # ---- 导出 ----
    xlsx = os.path.join(OUT, "Chow_pooling检验结果.xlsx")
    with pd.ExcelWriter(xlsx, engine="openpyxl") as w:
        res.to_excel(w, sheet_name="主结果_连续结局", index=False)
        pd.DataFrame([r_cat]).to_excel(w, sheet_name="主结果_意向4类", index=False)
        res_s.to_excel(w, sheet_name="敏感性_剔Q9超限", index=False)
        comp = df.groupby("batch")[["Q4_压力源指数", "Q5_身心健康", "Q10_体制内偏好",
                                    "Q10_灵活新就业偏好", "Q11_指数A_剔除不确定"]].mean().T
        comp.columns = ["主样本1026", "第二批553"]
        comp["差值"] = comp["第二批553"] - comp["主样本1026"]
        comp.to_excel(w, sheet_name="两批均值对照")
        pd.crosstab(df["batch"], df["grade"]).to_excel(w, sheet_name="年级构成")
        pd.crosstab(df["batch"], df["major"]).to_excel(w, sheet_name="专业构成")
    print(f"\n已写出: {xlsx}")

    # ---- 森林图：θ 点估计 + 95%CI ----
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    labels = ["Public-sector\npreference", "Flexible/new-form\npreference",
              "Expectation-impact\nindex", "Health impact"]
    ys = np.arange(len(labels))[::-1]
    for y, r in zip(ys, rows):
        ax.plot([r["theta_lo"], r["theta_hi"]], [y, y], lw=2, color="#378ADD")
        ax.plot([r["theta"]], [y], "o", ms=7, color="#185FA5")
    ax.axvline(0, color="#A32D2D", ls="--", lw=1)
    ax.set_yticks(ys)
    ax.set_yticklabels(labels, fontsize=11)
    ax.set_xlabel("Batch x stress interaction  (theta, with 95% CI)", fontsize=11)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.text(-0.02, 0.0, "", transform=ax.transAxes)
    fig.tight_layout()
    png = os.path.join(FIG, "fig_RQ3_pooling_theta_forest.png")
    fig.savefig(png, dpi=600, bbox_inches="tight", facecolor="white")
    print(f"已写出: {png}")

    return res, res_s, r_cat, padj


if __name__ == "__main__":
    main()
