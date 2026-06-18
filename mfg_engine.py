"""
mfg_engine.py — Mean Field Game Nash Equilibrium Engine
=========================================================

Theory
------
We model N ETF agents, each choosing a trading intensity u_i(t) over
a finite horizon [0, T] to maximise:

  J_i = E[ ∫₀ᵀ (μ_i·u_i − γ/2·σ_i²·u_i² − λ·u_i² − κ·m(t)·u_i) dt ]

Where:
  μ_i   = individual ETF drift (estimated from recent returns)
  σ_i   = individual ETF volatility
  γ     = risk-aversion coefficient (CARA)
  λ     = quadratic execution cost (Almgren-Chriss)
  κ     = mean-field coupling strength
  m(t)  = mean field = (1/N) Σ u_j(t)  — average agent action

At Nash equilibrium the optimal control for agent i satisfies:

  u_i*(t) = (μ_i − κ·m*(t)) / (γ·σ_i² + 2λ)

And the mean field satisfies the fixed-point condition:

  m* = (1/N) Σ u_i*(m*)

We solve this via Picard iteration (fixed-point iteration) which
converges in O(20) iterations for typical parameter values.

Score Construction
------------------
For each ETF i, the composite MFG score is:

  score_i = w_drift · û_i  −  w_crowding · κ·m*  +  w_macro · ⟨u_i*, macro⟩

Where:
  û_i           = normalised equilibrium control (z-scored cross-sectionally)
  κ·m*          = crowding penalty term
  ⟨u_i*, macro⟩ = alignment of ETF's equilibrium action with macro direction

The final score is cross-sectionally z-scored per universe per window.

References
----------
- Lasry & Lions (2007). Mean field games. Japanese Journal of Mathematics.
- Carmona & Delarue (2018). Probabilistic Theory of Mean Field Games. Springer.
- Guéant, Lasry & Lions (2011). Mean Field Games and Applications. Springer LNM.
- Cardaliaguet et al. (2019). The Master Equation and the Convergence Problem
  in Mean Field Games. Princeton University Press.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple

import config


def _estimate_drift(log_returns: pd.Series, window: int) -> float:
    """Annualised mean log return over the last `window` bars."""
    r = log_returns.iloc[-window:]
    if len(r) < max(window // 2, 10):
        return 0.0
    return float(r.mean() * 252)


def _estimate_vol(log_returns: pd.Series, window: int) -> float:
    """Annualised volatility over the vol estimation window."""
    r = log_returns.iloc[-config.VOL_WINDOW:]
    if len(r) < 10:
        return 0.20  # fallback: 20% annualised
    return float(r.std() * np.sqrt(252)) + 1e-8  # avoid zero


def _macro_signal(macro_df: pd.DataFrame, window: int) -> np.ndarray:
    """
    Construct a scalar macro direction signal from recent changes.
    Returns a vector of length = len(MACRO_COLS) with z-scored recent changes.
    """
    if macro_df.empty:
        return np.zeros(len(config.MACRO_COLS))

    available = [c for c in config.MACRO_COLS if c in macro_df.columns]
    if not available:
        return np.zeros(len(config.MACRO_COLS))

    recent = macro_df[available].iloc[-window:]
    if len(recent) < 2:
        return np.zeros(len(config.MACRO_COLS))

    # Normalised pct-change over the window
    start = recent.iloc[0]
    end   = recent.iloc[-1]
    pct   = ((end - start) / (start.abs() + 1e-8)).values

    # z-score across macro columns
    mu  = pct.mean()
    std = pct.std() + 1e-8
    return (pct - mu) / std


def _macro_alignment(u_i: float, macro_signal: np.ndarray,
                     drift: float) -> float:
    """
    Alignment of agent i's equilibrium action with macro direction.
    Positive = agent is trading in the direction macro signals favour.
    """
    # Simplified: sign agreement between drift and aggregate macro signal
    macro_agg = float(macro_signal.mean())
    return float(np.sign(drift) * macro_agg * abs(u_i))


def solve_mfg_equilibrium(
    drifts:  np.ndarray,   # shape (N,) — annualised drift per ETF
    vols:    np.ndarray,   # shape (N,) — annualised vol per ETF
    kappa:   float,        # mean-field coupling
    gamma:   float,        # risk aversion
    lam:     float,        # execution cost
    n_iter:  int,
    tol:     float,
) -> Tuple[np.ndarray, float, int]:
    """
    Solve MFG fixed-point via Picard iteration.

    Returns
    -------
    u_star  : np.ndarray shape (N,)  — equilibrium controls
    m_star  : float                  — equilibrium mean field
    n_iters : int                    — iterations to convergence
    """
    N = len(drifts)
    denominator = gamma * vols**2 + 2 * lam   # (γσ²+2λ) per agent

    # Initialise mean field at zero
    m = 0.0

    for it in range(n_iter):
        # Individual best responses given current mean field
        u = (drifts - kappa * m) / denominator

        # Update mean field
        m_new = u.mean()

        if abs(m_new - m) < tol:
            return u, m_new, it + 1

        m = m_new

    return u, m, n_iter


def compute_mfg_scores(
    prices:    pd.DataFrame,
    macro_df:  pd.DataFrame,
    tickers:   List[str],
    window:    int,
) -> pd.Series:
    """
    Compute MFG Nash equilibrium scores for all ETFs in the universe.

    Parameters
    ----------
    prices   : DataFrame of closing prices, DatetimeIndex
    macro_df : DataFrame of macro signal levels, DatetimeIndex
    tickers  : list of ETF tickers in this universe
    window   : lookback window in trading days

    Returns
    -------
    pd.Series indexed by ticker, values = composite MFG z-score
    """
    avail = [t for t in tickers if t in prices.columns]
    if len(avail) < 2:
        return pd.Series(dtype=float)

    # Need enough data
    min_rows = window + config.VOL_WINDOW + 10
    if len(prices) < min_rows:
        return pd.Series(dtype=float)

    # ── Compute log returns ───────────────────────────────────────────────────
    log_ret = np.log(prices[avail] / prices[avail].shift(1)).dropna()

    if len(log_ret) < window:
        return pd.Series(dtype=float)

    # ── Estimate drifts and vols per ETF ─────────────────────────────────────
    drifts = np.array([_estimate_drift(log_ret[t], window) for t in avail])
    vols   = np.array([_estimate_vol(log_ret[t],   window) for t in avail])

    # ── Solve MFG Nash equilibrium ────────────────────────────────────────────
    u_star, m_star, n_iters = solve_mfg_equilibrium(
        drifts  = drifts,
        vols    = vols,
        kappa   = config.MF_COUPLING,
        gamma   = config.GAMMA_RISK,
        lam     = config.LAMBDA_COST,
        n_iter  = config.N_ITER,
        tol     = config.MF_TOL,
    )

    # ── Macro signal ──────────────────────────────────────────────────────────
    macro_sig = _macro_signal(macro_df, window)

    # ── Build composite scores ────────────────────────────────────────────────
    crowding_penalty = config.MF_COUPLING * m_star  # scalar, same for all agents

    raw_scores = {}
    for i, ticker in enumerate(avail):
        drift_component    = float(u_star[i])
        crowding_component = crowding_penalty
        macro_component    = _macro_alignment(u_star[i], macro_sig, drifts[i])

        raw_scores[ticker] = (
            config.WEIGHT_DRIFT    * drift_component
            - config.WEIGHT_CROWDING * crowding_component
            + config.WEIGHT_MACRO  * macro_component
        )

    scores = pd.Series(raw_scores)

    # ── Cross-sectional z-score ───────────────────────────────────────────────
    mu  = scores.mean()
    std = scores.std()
    if std < 1e-10:
        return pd.Series(0.0, index=scores.index)

    return (scores - mu) / std
