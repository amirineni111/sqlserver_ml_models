"""
NASDAQ ML Prediction Configuration

Centralized configuration for confidence thresholds and model parameters.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ============================================================================
# CONFIDENCE THRESHOLDS
# ============================================================================

# High confidence threshold (default: 58%)
# Recalibrated June 2026 from 60% to 58% after the Jun 21 retrain switched to
# sigmoid/Platt calibration, which compresses the model's confidence into ~0.50-0.60
# (Buy p99 = 0.599). The old 60% gate was unreachable -> high_confidence flag never
# fired. 58% reserves the flag for the tier approaching the 60%-accuracy band
# (ml_prediction_outcomes: Buy 60+% bucket = 60.1% acc). See [[nasdaq-accuracy-overhaul-june-2026]].
# NOTE: Layer 2 overrides this from data/derived_thresholds.json when present (see bottom).
HIGH_CONFIDENCE_THRESHOLD = float(os.getenv('HIGH_CONFIDENCE_THRESHOLD', '0.58'))

# Medium confidence threshold (default: 55%)
MEDIUM_CONFIDENCE_THRESHOLD = float(os.getenv('MEDIUM_CONFIDENCE_THRESHOLD', '0.55'))

# Low confidence threshold (below MEDIUM_CONFIDENCE_THRESHOLD)
# Signals below 55% are flagged as low confidence

# ============================================================================
# SIGNAL RELIABILITY FILTERS
# ============================================================================
# Applied post-prediction to suppress signals identified as systematically
# unreliable from accuracy analysis (May 2026):

# BUY dead-zone fix: Recalibrated June 2026 from 0.67 to 0.55. The 0.67 gate was tuned
# for the previous isotonic model's wider confidence spread; the Jun 21 sigmoid/Platt
# retrain compressed Buy confidence to ~0.50-0.60 (p99 = 0.599), so 0.67 suppressed 100%
# of buys (ml_prediction_summary buy_signals = 0 every day). Realized accuracy by bucket
# (ml_prediction_outcomes): Buy <52%->51.2%, 52-55%->50.9%, 55-60%->53.3%, 60+%->60.1%.
# 0.55 is the first tier meaningfully above coin-flip (balanced quality/volume choice).
# NOTE: Layer 2 overrides this from data/derived_thresholds.json when present (see bottom).
BUY_MIN_CONFIDENCE = float(os.getenv('BUY_MIN_CONFIDENCE', '0.55'))

# SELL calibration: the old rule suppressed Sells ABOVE 55% because the prior model was
# inversely calibrated. The Jun 23 2026 retrain FIXED this — held-out test shows Sell
# accuracy now RISES monotonically with confidence (54.7% @50-55% -> 60.1% @55-60% ->
# 67.6% @60-70%), so suppressing high-confidence sells is now backwards. Effectively
# disabled (0.99); Layer 2 (derive_thresholds.py) will set the real cap once current-model
# outcomes accrue. NOTE: overridden by data/derived_thresholds.json when present.
SELL_MAX_CONFIDENCE = float(os.getenv('SELL_MAX_CONFIDENCE', '0.99'))

# RSI overbought Buy block: Buying overbought stocks (RSI > 70) loses 58% of the time.
# Hard rule: suppress any Buy signal where RSI > this threshold.
RSI_OVERBOUGHT_BUY_BLOCK = float(os.getenv('RSI_OVERBOUGHT_BUY_BLOCK', '70'))

# Energy sector exclusion: Energy stocks have 39.3% 1-day accuracy (worse than random).
# Commodity/geopolitical drivers not in feature set. Set to False to re-enable.
ENERGY_SECTOR_EXCLUDED = os.getenv('ENERGY_SECTOR_EXCLUDED', 'true').lower() == 'true'

# Direction of the energy exclusion. The blanket rule was derived from COMBINED
# accuracy, but energy is strongly directional over the last 150d of outcomes:
#   energy Buy  suppressed rows: 67.1% win (n=1,514)  <- best bucket in the system
#   energy Sell suppressed rows: 40.3% win (n=3,524)  <- worst
# So the blanket rule throws away the good half. 'both' preserves the historical
# behaviour and stays the default because switching to 'sell' ENABLES new trades
# (a risk-increasing change, and a 150-day energy uptrend may not persist).
# Set ENERGY_EXCLUSION_SIDE=sell to act on the evidence.
ENERGY_EXCLUSION_SIDE = os.getenv('ENERGY_EXCLUSION_SIDE', 'both').lower()

# Industry-level reliability filters (Sep 2026). `sector` is too coarse to express
# these: 'Basic Materials' contains both the best and the worst industry buckets.
# Format matches SECTOR_CONFIDENCE_OVERRIDES: {industry: {side: min_confidence}}.
# Each entry below is backed by 150 days of ml_prediction_outcomes:
#   Other Industrial Metals & Mining  Buy  39.1% actionable win (n=302)   -> gate hard
#   Marine Shipping                   Sell 33.3% actionable win (n=168)   -> gate hard
# Deliberately NOT gated, despite being requested in review — the data says they
# are working and blocking them would remove good signal:
#   Gold/Silver/Precious Metals       Buy  59.2% actionable win (n=1,131) vs 52.9% baseline
#   Marine Shipping                   Buy  81.4% actionable win (n=70)
#   Biotechnology                     Buy  53.8% / Sell 49.2% — within noise of baseline
# 0.99 = effectively always suppress; a value inside the model's range gates instead.
_industry_overrides_env = os.getenv('INDUSTRY_CONFIDENCE_OVERRIDES', '')
if _industry_overrides_env:
    import json as _json_ind
    INDUSTRY_CONFIDENCE_OVERRIDES: dict = _json_ind.loads(_industry_overrides_env)
else:
    INDUSTRY_CONFIDENCE_OVERRIDES: dict = {
        'Other Industrial Metals & Mining': {'Up': 0.99},
        'Marine Shipping': {'Down': 0.99},
    }

# Minimum confidence for a Sell to be actionable. There has never been a Sell floor
# (only SELL_MAX_CONFIDENCE, a cap), which is the real source of the Buy/Sell
# suppression asymmetry: 82% of Buys are gated vs 11% of Sells. Left DISABLED (0.0)
# because a floor only helps if confidence is informative, and on the Sell side it
# is not — realized win rate is flat across every bucket (49.8/51.6/50.5/51.4/51.5)
# and the Brier skill score is negative (-0.0154). Gating a non-signal just cuts
# volume at random. The Sell side needs model work, not a threshold. Knob exists so
# the floor can be enabled the moment recalibration makes Sell confidence mean something.
SELL_MIN_CONFIDENCE = float(os.getenv('SELL_MIN_CONFIDENCE', '0.0'))

# Minimum stock price filter: exclude penny/micro-cap stocks from predictions and training.
# Stocks below $5 have erratic volume, wide bid-ask spreads, and thin liquidity that makes
# technical signals unreliable. Applied at SQL layer (training + prediction) and Python layer.
# Analysis: Buy acc below $5 is 41.9-46.2% vs 48.8% for stocks above $10.
MIN_STOCK_PRICE = float(os.getenv('MIN_STOCK_PRICE', '5.0'))

# Sector-specific confidence overrides: sectors with near-random accuracy require higher
# confidence thresholds to emit actionable signals.
# Technology 1d accuracy: 49.9% | Healthcare 1d accuracy: 49.4%
# Override format: JSON string e.g. '{"Technology": 0.57, "Healthcare": 0.57}'
# Recalibrated June 2026 from 0.72 to 0.57: 0.72 was unreachable under the current
# calibration (Buy p99 = 0.599) and double-suppressed the Tech-heavy NASDAQ 100 on top
# of buy_dead_zone. 0.57 keeps a premium over the 0.55 base buy gate while staying within
# the model's actual confidence range.
_sector_overrides_env = os.getenv('SECTOR_CONFIDENCE_OVERRIDES', '')
if _sector_overrides_env:
    import json as _json
    SECTOR_CONFIDENCE_OVERRIDES: dict = _json.loads(_sector_overrides_env)
else:
    SECTOR_CONFIDENCE_OVERRIDES: dict = {'Technology': 0.57, 'Healthcare': 0.57}


# ============================================================================
# MODEL PARAMETERS
# ============================================================================

# Minimum prediction confidence to save to database (default: 50%)
MIN_PREDICTION_CONFIDENCE = float(os.getenv('MIN_PREDICTION_CONFIDENCE', '0.50'))

# Model accuracy threshold for retraining trigger (default: 52%)
MIN_MODEL_ACCURACY = float(os.getenv('MIN_MODEL_ACCURACY', '0.52'))

# Live prediction accuracy threshold for retraining (default: 50%)
MIN_LIVE_ACCURACY = float(os.getenv('MIN_LIVE_ACCURACY', '0.50'))

# Label dead-zone (percent): training samples whose 5-day forward return is
# smaller than this in absolute value are excluded from train/calibration sets.
# A stock that moves +0.05% in 5 days is noise, not an "Up" — these near-zero
# labels blur the decision boundary. Test set is never filtered (production
# scores every stock). Set to 0 to disable.
LABEL_DEAD_ZONE_PCT = float(os.getenv('LABEL_DEAD_ZONE_PCT', '1.0'))

# Cooldown (days) between performance-triggered retrains, to prevent retrain
# churn when live accuracy oscillates around the threshold.
RETRAIN_COOLDOWN_DAYS = int(os.getenv('RETRAIN_COOLDOWN_DAYS', '5'))


# ============================================================================
# HISTORICAL DATA PARAMETERS
# ============================================================================

# Days of historical data to fetch for predictions (default: 365)
HISTORICAL_DAYS = int(os.getenv('HISTORICAL_DAYS', '365'))

# Days of historical data for indicators calculation (default: 80)
INDICATOR_LOOKBACK_DAYS = int(os.getenv('INDICATOR_LOOKBACK_DAYS', '80'))


# ============================================================================
# DERIVED THRESHOLD OVERRIDES (Layer 2 — outcome-driven, see derive_thresholds.py)
# ============================================================================
# If data/derived_thresholds.json exists it overrides the hardcoded defaults above.
# The file is regenerated each retrain from ml_prediction_outcomes so the gates can
# never silently drift out of the model's confidence range again (the bug that zeroed
# out buys in Jun 2026). Hardcoded values above are the fallback when the file is absent
# or malformed. Mirrors the SECTOR_CONFIDENCE_OVERRIDES env-load pattern above.
DERIVED_THRESHOLDS_PATH = os.getenv(
    'DERIVED_THRESHOLDS_PATH',
    os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'derived_thresholds.json'),
)
DERIVED_THRESHOLDS_LOADED = False
try:
    if os.path.exists(DERIVED_THRESHOLDS_PATH):
        import json as _json_dt
        with open(DERIVED_THRESHOLDS_PATH) as _f_dt:
            _derived = _json_dt.load(_f_dt)
        if isinstance(_derived.get('buy_min_confidence'), (int, float)):
            BUY_MIN_CONFIDENCE = float(_derived['buy_min_confidence'])
        if isinstance(_derived.get('high_confidence_threshold'), (int, float)):
            HIGH_CONFIDENCE_THRESHOLD = float(_derived['high_confidence_threshold'])
        if isinstance(_derived.get('sell_max_confidence'), (int, float)):
            SELL_MAX_CONFIDENCE = float(_derived['sell_max_confidence'])
        if isinstance(_derived.get('sector_confidence_overrides'), dict):
            SECTOR_CONFIDENCE_OVERRIDES = {
                str(_k): float(_v) for _k, _v in _derived['sector_confidence_overrides'].items()
            }
        DERIVED_THRESHOLDS_LOADED = True
except Exception as _e_dt:  # noqa: BLE001 — never let a bad file break predictions
    print(f"[CONFIG] Could not load derived thresholds ({_e_dt}); using hardcoded defaults")


# ============================================================================
# VALIDATION
# ============================================================================

# Validate thresholds
if not (0.0 <= HIGH_CONFIDENCE_THRESHOLD <= 1.0):
    raise ValueError(f"HIGH_CONFIDENCE_THRESHOLD must be between 0 and 1, got {HIGH_CONFIDENCE_THRESHOLD}")

if not (0.0 <= MEDIUM_CONFIDENCE_THRESHOLD <= HIGH_CONFIDENCE_THRESHOLD):
    raise ValueError(f"MEDIUM_CONFIDENCE_THRESHOLD must be between 0 and HIGH_CONFIDENCE_THRESHOLD, got {MEDIUM_CONFIDENCE_THRESHOLD}")

if not (0.0 <= MIN_PREDICTION_CONFIDENCE <= 1.0):
    raise ValueError(f"MIN_PREDICTION_CONFIDENCE must be between 0 and 1, got {MIN_PREDICTION_CONFIDENCE}")

if not (0.0 <= BUY_MIN_CONFIDENCE <= 1.0):
    raise ValueError(f"BUY_MIN_CONFIDENCE must be between 0 and 1, got {BUY_MIN_CONFIDENCE}")

if not (0.0 <= SELL_MAX_CONFIDENCE <= 1.0):
    raise ValueError(f"SELL_MAX_CONFIDENCE must be between 0 and 1, got {SELL_MAX_CONFIDENCE}")

if MIN_STOCK_PRICE < 0:
    raise ValueError(f"MIN_STOCK_PRICE must be >= 0, got {MIN_STOCK_PRICE}")

if LABEL_DEAD_ZONE_PCT < 0:
    raise ValueError(f"LABEL_DEAD_ZONE_PCT must be >= 0, got {LABEL_DEAD_ZONE_PCT}")

if RETRAIN_COOLDOWN_DAYS < 0:
    raise ValueError(f"RETRAIN_COOLDOWN_DAYS must be >= 0, got {RETRAIN_COOLDOWN_DAYS}")

for _sector, _thresh in SECTOR_CONFIDENCE_OVERRIDES.items():
    if not (0.0 <= _thresh <= 1.0):
        raise ValueError(f"SECTOR_CONFIDENCE_OVERRIDES['{_sector}'] must be 0-1, got {_thresh}")


# ============================================================================
# CONFIGURATION SUMMARY (for logging)
# ============================================================================

def print_config():
    """Print current configuration"""
    print("=" * 80)
    print("NASDAQ ML PREDICTION CONFIGURATION")
    print("=" * 80)
    print(f"High Confidence Threshold:    {HIGH_CONFIDENCE_THRESHOLD:.0%} (≥{HIGH_CONFIDENCE_THRESHOLD*100:.0f}%)")
    print(f"Medium Confidence Threshold:  {MEDIUM_CONFIDENCE_THRESHOLD:.0%} (≥{MEDIUM_CONFIDENCE_THRESHOLD*100:.0f}%)")
    print(f"Low Confidence:               <{MEDIUM_CONFIDENCE_THRESHOLD:.0%} (<{MEDIUM_CONFIDENCE_THRESHOLD*100:.0f}%)")
    print(f"Minimum Prediction Confidence: {MIN_PREDICTION_CONFIDENCE:.0%}")
    print(f"Model Accuracy Threshold:     {MIN_MODEL_ACCURACY:.0%}")
    print(f"Live Accuracy Threshold:      {MIN_LIVE_ACCURACY:.0%}")
    print(f"Label Dead-Zone:              |5d return| < {LABEL_DEAD_ZONE_PCT}% excluded from training")
    print(f"Retrain Cooldown:             {RETRAIN_COOLDOWN_DAYS} days")
    print(f"Historical Days:              {HISTORICAL_DAYS}")
    print(f"--- Reliability Filters ---")
    print(f"Buy Min Confidence:           {BUY_MIN_CONFIDENCE:.0%} (dead-zone suppression)")
    print(f"Sell Max Confidence:          {SELL_MAX_CONFIDENCE:.0%} (inverted calibration filter)")
    print(f"RSI Overbought Buy Block:     RSI > {RSI_OVERBOUGHT_BUY_BLOCK:.0f}")
    print(f"Energy Sector Excluded:       {ENERGY_SECTOR_EXCLUDED}")
    print(f"Min Stock Price:              ${MIN_STOCK_PRICE:.2f}")
    print(f"Sector Confidence Overrides:  {SECTOR_CONFIDENCE_OVERRIDES}")
    print(f"Derived Thresholds Loaded:    {DERIVED_THRESHOLDS_LOADED} ({DERIVED_THRESHOLDS_PATH})")
    print(f"Indicator Lookback Days:      {INDICATOR_LOOKBACK_DAYS}")
    print("=" * 80)
    print("Model uses sigmoid/Platt calibration (Jun 21 2026 retrain)")
    print("Thresholds recalibrated Jun 2026 to match compressed confidence range (Buy p99=0.599)")
    print("=" * 80)


if __name__ == "__main__":
    print_config()
