# -*- coding: utf-8 -*-
"""共享工具：polychoric 相关矩阵与相关矩阵修正。

从 step22_efa.py 抽出，供 step21 / step22 共用，避免同一套数学在两份脚本里各写一遍。
算法：Olsson (1979) 两步法 —— 先由边际频数估阈值 τ = Φ^{-1}(cum p)，
再在给定阈值下用最大似然估 ρ；二元正态 CDF 用 Owen's T 公式。
与 scipy.multivariate_normal.cdf 对照最大误差 < 1e-16。
"""
import sys as _sys
if hasattr(_sys.stdout, "reconfigure"):
    _sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np
from scipy.stats import norm
from scipy.special import owens_t
from scipy.optimize import brentq, minimize_scalar

# ⚠ 上面四个都必须在本模块内 import：这些函数是在
#   poly_utils 的全局命名空间里查名字的，调用方（step21/step22）
#   即使自己也 import 了也不算。缺一个会抛 NameError，
#   而 polychoric_pair 内部有 try/except 兜底 → 静默返回 NaN，
#   症状是「相关矩阵退化」而不是报错，极难定位。

EPS = 1e-8
SMALL, BIG = -8.0, 8.0   # ±inf 哨兵阈值（Φ(±8) 误差 < 1e-14）


def thresholds_from_margins(counts):
    """由边际频数得累积比例 → 阈值 τ = Φ^{-1}(cum p)"""
    pr = np.asarray(counts, float)
    pr = pr / pr.sum()
    cum = np.cumsum(pr)[:-1]
    return norm.ppf(np.clip(cum, 1e-6, 1 - 1e-6))


def _protect(z):
    """避免 h=0 在 Owens T 的参数中除零"""
    z = np.asarray(z, float)
    out = z.copy()
    small = np.abs(out) < EPS
    out = np.where(small, np.where(out >= 0, EPS, -EPS), out)
    return out


def _bvn_cdf_grid(TA, TB, rho):
    """二元标准正态 CDF 网格（6×6，含哨兵行/列），Owen's T 向量化"""
    s = np.sqrt(1.0 - rho * rho)
    H = np.repeat(TA[:, None], TB.size, axis=1)
    K = np.repeat(TB[None, :], TA.size, axis=0)
    Hs, Ks = _protect(H), _protect(K)
    a1 = (K - rho * Hs) / (Hs * s)
    a2 = (Hs - rho * Ks) / (Ks * s)
    delta = np.where(H * K > 0, 0.0, 0.5)
    C = 0.5 * (norm.cdf(H) + norm.cdf(K)) - owens_t(Hs, a1) - owens_t(Ks, a2) - delta
    return np.clip(C, 0.0, 1.0)


def _cell_prob(TA, TB, rho):
    P = _bvn_cdf_grid(TA, TB, rho)
    return P[1:, 1:] - P[:-1, 1:] - P[1:, :-1] + P[:-1, :-1]


def _dloglik(tab, TA, TB, rho):
    """d L / d ρ：利用 dΦ2/dρ = φ2（闭式密度）"""
    P = _cell_prob(TA, TB, rho)
    s2 = 1.0 - rho * rho
    H = np.repeat(TA[:, None], TB.size, axis=1)
    K = np.repeat(TB[None, :], TA.size, axis=0)
    D = np.zeros_like(H)
    hi, ki = H[1:-1, 1:-1], K[1:-1, 1:-1]
    D[1:-1, 1:-1] = np.exp(-(hi ** 2 - 2 * rho * hi * ki + ki ** 2) / (2 * s2)) \
                    / (2 * np.pi * np.sqrt(s2))
    dP = D[1:, 1:] - D[:-1, 1:] - D[1:, :-1] + D[:-1, :-1]
    return float(np.sum(tab / np.clip(P, 1e-12, None) * dP))


def _loglik(tab, TA, TB, rho):
    P = _cell_prob(TA, TB, rho)
    return float(np.sum(tab * np.log(np.clip(P, 1e-300, None))))


def polychoric_pair(x, y):
    """单对 polychoric 相关系数"""
    ux = np.unique(x)
    uy = np.unique(y)
    tab = np.array([[np.sum((x == a) & (y == b)) for b in uy] for a in ux], float)
    cx = np.array([np.sum(x == a) for a in ux], float)
    cy = np.array([np.sum(y == b) for b in uy], float)
    if len(ux) < 2 or len(uy) < 2:
        return np.nan
    TA = np.array([SMALL, *thresholds_from_margins(cx), BIG])
    TB = np.array([SMALL, *thresholds_from_margins(cy), BIG])
    lo, hi = -0.985, 0.985
    try:
        if _dloglik(tab, TA, TB, lo) * _dloglik(tab, TA, TB, hi) > 0:
            # 端点同号：根在边界，退化为有界最大化
            return float(minimize_scalar(lambda r: -_loglik(tab, TA, TB, r),
                                         bounds=(lo, hi), method="bounded").x)
        return float(brentq(lambda r: _dloglik(tab, TA, TB, r), lo, hi, xtol=1e-7))
    except Exception:
        return np.nan


def polychoric_matrix(X):
    """X: (n, p) 整数矩阵 → polychoric 相关矩阵"""
    p = X.shape[1]
    R = np.eye(p)
    for i in range(p):
        for j in range(i + 1, p):
            r = polychoric_pair(X[:, i], X[:, j])
            if np.isnan(r):
                r = 0.0
            R[i, j] = R[j, i] = r
    return R


def nearest_pd(R, max_iter=100, tol=1e-9):
    """非正定修正：特征值裁剪 + 再归一化对角为 1（Higham 简化版）"""
    R = (R + R.T) / 2
    for _ in range(max_iter):
        ev, evec = np.linalg.eigh(R)
        if ev.min() > tol:
            break
        ev_clipped = np.clip(ev, tol, None)
        R = evec @ np.diag(ev_clipped) @ evec.T
        d = np.sqrt(np.diag(R))
        R = R / np.outer(d, d)
    np.fill_diagonal(R, 1.0)
    return R
