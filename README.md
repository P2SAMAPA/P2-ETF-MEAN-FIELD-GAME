# ♟️ P2-ETF-MEAN-FIELD-GAME

**Nash Equilibrium ETF Ranking Engine — Mean Field Game Framework**

Part of the **P2Quant Engine Suite** · [P2SAMAPA](https://github.com/P2SAMAPA)

---

## What This Engine Does

This engine models ETF market participants as rational agents who simultaneously
optimise their trading strategies, accounting for the crowding effect of all
other agents. The Nash equilibrium of this game determines which ETFs offer
the best risk-adjusted opportunity *after* factoring in crowding externalities.

### Why MFG for ETFs?

Standard signal engines (momentum, carry, PLV) measure properties of prices in
isolation. MFG asks: *if all rational agents optimised simultaneously, what would
the equilibrium look like — and which ETFs still offer positive opportunity at
that equilibrium?*

This directly models crowding — one of the most important and least-modelled
risks in systematic ETF strategies.

---

## Theory

### Agent Objective

Each agent i controls trading intensity u_i(t) over horizon [0, T]:

```
J_i = ∫₀ᵀ [ μᵢ·uᵢ  −  (γ/2)·σᵢ²·uᵢ²  −  λ·uᵢ²  −  κ·m(t)·uᵢ ] dt
```

| Term | Parameter | Meaning |
|------|-----------|---------|
| `μᵢ·uᵢ` | drift | Profit from trading in direction of estimated return |
| `(γ/2)·σᵢ²·uᵢ²` | γ = GAMMA_RISK | CARA risk aversion weighted by variance |
| `λ·uᵢ²` | λ = LAMBDA_COST | Quadratic execution cost (Almgren-Chriss) |
| `κ·m(t)·uᵢ` | κ = MF_COUPLING | **Crowding penalty** — mean field interaction |

### Nash Equilibrium

The optimal control for agent i, given mean field m, is:

```
uᵢ*(t) = (μᵢ − κ·m*) / (γ·σᵢ² + 2λ)
```

The mean field satisfies the fixed-point condition:

```
m* = (1/N) Σ uᵢ*(m*)
```

Solved via **Picard (fixed-point) iteration** — converges in ~10–20 iterations.

### Score Construction

```
score_i = w_drift · uᵢ*  −  w_crowding · κ·m*  +  w_macro · ⟨uᵢ*, macro⟩
```

| Component | Weight | Meaning |
|-----------|--------|---------|
| Equilibrium drift | 50% | Optimal action at Nash equilibrium |
| Crowding penalty | 30% | Cost of mean-field interaction (negative = crowded) |
| Macro alignment | 20% | Agreement of equilibrium action with macro direction |

Final score: **cross-sectional z-score** per universe per window.

---

## Hyperparameters

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `MF_COUPLING` κ | 0.6 | Mean-field interaction strength (0 = no crowding, 1 = full) |
| `GAMMA_RISK` γ | 0.5 | CARA risk aversion |
| `LAMBDA_COST` λ | 0.02 | Quadratic execution cost |
| `N_ITER` | 50 | Max Picard iterations |
| `MF_TOL` | 1e-6 | Convergence tolerance |

---

## Universes

| Universe | Tickers |
|---|---|
| FI_COMMODITIES | TLT, VCIT, LQD, HYG, VNQ, GLD, SLV |
| EQUITY_SECTORS | SPY, QQQ, XLK, XLF, XLE, XLV, XLI, XLY, XLP, XLU, GDX, XME, IWF, XSD, XBI, IWM, IWD, IWO, XLB, XLRE |
| COMBINED | All of the above |

---

## Rolling Windows

```
63d · 126d · 252d · 504d
```

---

## Repository Structure

```
P2-ETF-MEAN-FIELD-GAME/
├── config.py          # Universes, MFG hyperparameters, weights
├── data_manager.py    # HuggingFace loader → (prices, macro) DataFrames
├── mfg_engine.py      # Core MFG solver: drift estimation → Picard → scores
├── trainer.py         # Orchestrator: load → solve → build JSON → upload
├── push_results.py    # HfApi.upload_file wrapper
├── streamlit_app.py   # Two-tab Streamlit dashboard
├── us_calendar.py     # US trading calendar helper
├── requirements.txt
└── .github/
    └── workflows/
        └── daily.yml  # Scheduled run 23:30 UTC Mon–Fri
```

---

## Output JSON Schemas

### Tab 1 — `mfg_engine_YYYY-MM-DD.json`

```json
{
  "run_date": "2026-06-18",
  "universes": {
    "FI_COMMODITIES": {
      "top_etfs": [
        {"ticker": "TLT", "mfg_score": 1.23, "best_window": 252}
      ],
      "full_scores": {
        "TLT": {"score": 1.23, "best_window": 252}
      }
    }
  }
}
```

### Tab 2 — `mfg_engine_windows_YYYY-MM-DD.json`

```json
{
  "run_date": "2026-06-18",
  "universes": {
    "FI_COMMODITIES": {
      "windows": {
        "63":  {"top_etfs": [...], "full_ranking": [["TLT", 1.23], ...]},
        "252": {"top_etfs": [...], "full_ranking": [...]}
      }
    }
  }
}
```

---

## Setup

```bash
git clone https://github.com/P2SAMAPA/P2-ETF-MEAN-FIELD-GAME
cd P2-ETF-MEAN-FIELD-GAME
pip install -r requirements.txt

export HF_TOKEN=hf_...
python trainer.py
streamlit run streamlit_app.py
```

**Required GitHub secret:** `HF_TOKEN`

**Required HuggingFace dataset repo:** `P2SAMAPA/p2-etf-mean-field-game-results`

---

## Relationship to MEAN-FIELD-CROWDING Engine

| Engine | Focus |
|--------|-------|
| MEAN-FIELD-CROWDING | *Measures* how crowded ETFs are using volume, momentum, and macro correlation proxies |
| **MEAN-FIELD-GAME** | *Models* the Nash equilibrium of rational agents under crowding — what the optimal strategy is *given* that crowding exists |

They are complementary: use CROWDING to detect overcrowded ETFs, and MFG to find ETFs where the Nash equilibrium drift is positive even after crowding discounts.

---

## References

- Lasry, J.M. & Lions, P.L. (2007). Mean field games. *Japanese Journal of Mathematics*, 2(1), 229–260.
- Carmona, R. & Delarue, F. (2018). *Probabilistic Theory of Mean Field Games with Applications*. Springer.
- Guéant, O., Lasry, J.M. & Lions, P.L. (2011). Mean field games and applications. *Paris-Princeton Lectures on Mathematical Finance*.
- Almgren, R. & Chriss, N. (2001). Optimal execution of portfolio transactions. *Journal of Risk*, 3, 5–39.
- Cardaliaguet, P. et al. (2019). *The Master Equation and the Convergence Problem in Mean Field Games*. Princeton University Press.
