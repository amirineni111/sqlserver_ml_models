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

# High confidence threshold (default: 60%)
# Lowered from 70% to 60% based on April 2026 analysis showing:
# - NASDAQ high-confidence (>70%) = 50% accuracy (coin flip)
# - Model already has isotonic calibration (CalibratedClassifierCV)
# - Lower threshold needed to match calibrated confidence output range
# - Aligned with NSE threshold for consistency across prediction systems
HIGH_CONFIDENCE_THRESHOLD = float(os.getenv('HIGH_CONFIDENCE_THRESHOLD', '0.60'))

# Medium confidence threshold (default: 55%)
MEDIUM_CONFIDENCE_THRESHOLD = float(os.getenv('MEDIUM_CONFIDENCE_THRESHOLD', '0.55'))

# Low confidence threshold (below MEDIUM_CONFIDENCE_THRESHOLD)
# Signals below 55% are flagged as low confidence

# ============================================================================
# SIGNAL RELIABILITY FILTERS
# ============================================================================
# Applied post-prediction to suppress signals identified as systematically
# unreliable from accuracy analysis (May 2026):

# BUY dead-zone fix: Buy signals at 55-65% confidence are inversely correlated
# with accuracy (34-48% acc). Only Buy signals >= 67% are reliable (67-83% acc).
BUY_MIN_CONFIDENCE = float(os.getenv('BUY_MIN_CONFIDENCE', '0.67'))

# SELL inverted calibration fix: High-confidence Sell signals are LESS accurate
# than low-confidence ones (70% confidence = 43% acc vs 50-54% confidence = 53-57%).
# Short-term filter: only emit Sell signals at confidence <= 55%.
# Long-term fix: recalibrate sell probabilities (requires model retrain).
SELL_MAX_CONFIDENCE = float(os.getenv('SELL_MAX_CONFIDENCE', '0.55'))

# RSI overbought Buy block: Buying overbought stocks (RSI > 70) loses 58% of the time.
# Hard rule: suppress any Buy signal where RSI > this threshold.
RSI_OVERBOUGHT_BUY_BLOCK = float(os.getenv('RSI_OVERBOUGHT_BUY_BLOCK', '70'))

# Energy sector exclusion: Energy stocks have 39.3% 1-day accuracy (worse than random).
# Commodity/geopolitical drivers not in feature set. Set to False to re-enable.
ENERGY_SECTOR_EXCLUDED = os.getenv('ENERGY_SECTOR_EXCLUDED', 'true').lower() == 'true'

# Minimum stock price filter: exclude penny/micro-cap stocks from predictions and training.
# Stocks below $5 have erratic volume, wide bid-ask spreads, and thin liquidity that makes
# technical signals unreliable. Applied at SQL layer (training + prediction) and Python layer.
# Analysis: Buy acc below $5 is 41.9-46.2% vs 48.8% for stocks above $10.
MIN_STOCK_PRICE = float(os.getenv('MIN_STOCK_PRICE', '5.0'))

# Sector-specific confidence overrides: sectors with near-random accuracy require higher
# confidence thresholds to emit actionable signals.
# Technology 1d accuracy: 49.9% | Healthcare 1d accuracy: 49.4%
# Override format: JSON string e.g. '{"Technology": 0.72, "Healthcare": 0.72}'
# Default: raise both to 72% (only very high-conviction signals survive).
_sector_overrides_env = os.getenv('SECTOR_CONFIDENCE_OVERRIDES', '')
if _sector_overrides_env:
    import json as _json
    SECTOR_CONFIDENCE_OVERRIDES: dict = _json.loads(_sector_overrides_env)
else:
    SECTOR_CONFIDENCE_OVERRIDES: dict = {'Technology': 0.72, 'Healthcare': 0.72}


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
    print(f"Indicator Lookback Days:      {INDICATOR_LOOKBACK_DAYS}")
    print("=" * 80)
    print("Model has isotonic calibration (CalibratedClassifierCV)")
    print("Threshold lowered from 70% to 60% to match calibrated output range")
    print("=" * 80)


if __name__ == "__main__":
    print_config()
