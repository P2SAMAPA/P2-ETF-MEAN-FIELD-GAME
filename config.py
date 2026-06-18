import os

HF_TOKEN    = os.environ.get("HF_TOKEN", "")
DATA_REPO   = "P2SAMAPA/fi-etf-macro-signal-master-data"
OUTPUT_REPO = "P2SAMAPA/p2-etf-mean-field-game-results"

UNIVERSES = {
    "FI_COMMODITIES": ["TLT", "VCIT", "LQD", "HYG", "VNQ", "GLD", "SLV"],
    "EQUITY_SECTORS": [
        "SPY", "QQQ", "XLK", "XLF", "XLE", "XLV", "XLI", "XLY",
        "XLP", "XLU", "GDX", "XME", "IWF", "XSD", "XBI",
        "IWM", "IWD", "IWO", "XLB", "XLRE",
    ],
    "COMBINED": [
        "TLT", "VCIT", "LQD", "HYG", "VNQ", "GLD", "SLV",
        "SPY", "QQQ", "XLK", "XLF", "XLE", "XLV", "XLI", "XLY",
        "XLP", "XLU", "GDX", "XME", "IWF", "XSD", "XBI",
        "IWM", "IWD", "IWO", "XLB", "XLRE",
    ],
}

# ── Macro signals ──────────────────────────────────────────────────────────────
MACRO_COLS = ["VIX", "DXY", "T10Y2Y"]

# ── Rolling windows (trading days) ────────────────────────────────────────────
WINDOWS = [63, 126, 252, 504]

# ── MFG model hyperparameters ─────────────────────────────────────────────────

# Mean-field interaction strength: how much the average agent's position
# feeds back into each individual agent's drift.
# 0 = no interaction (pure individual optimisation)
# 1 = full crowding feedback
MF_COUPLING = 0.6

# Quadratic execution cost coefficient (Almgren-Chriss style)
# Higher = agents trade more slowly, less market impact
LAMBDA_COST = 0.02

# Risk-aversion coefficient (CARA utility)
# Higher = agents are more risk-averse, smaller positions
GAMMA_RISK = 0.5

# Number of fixed-point iterations for Nash equilibrium solver
N_ITER = 50

# Convergence tolerance for mean-field fixed point
MF_TOL = 1e-6

# Time discretisation steps within each window (MFG horizon)
N_TIME_STEPS = 20

# Volatility estimation window for individual ETF vol
VOL_WINDOW = 63

# Macro sensitivity estimation window
MACRO_WINDOW = 252

# ── Score construction ────────────────────────────────────────────────────────
# Equilibrium score components:
#   - Nash drift         : agent's optimal drift at equilibrium
#   - Crowding penalty   : mean-field interaction term (negative = crowded)
#   - Macro alignment    : dot product of equilibrium action with macro signal
WEIGHT_DRIFT    = 0.50
WEIGHT_CROWDING = 0.30
WEIGHT_MACRO    = 0.20

TOP_N = 3
