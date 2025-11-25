# SQL Server ML Trading Signals Analysis Report

**Generated:** November 25, 2025  
**Project:** NASDAQ 100 RSI Trading Signal Prediction  
**Database:** SQL Server (localhost\MSSQLSERVER01, stockdata_db)

## Executive Summary

This analysis successfully developed machine learning models to predict RSI-based trading signals (Buy/Sell) using NASDAQ 100 stock market data. The **Gradient Boosting Classifier** emerged as the best performing model with strong prediction capabilities.

## Dataset Overview

- **Total Records:** 41,346 stock market observations
- **Time Period:** March 2024 - November 2025
- **Target Variable:** RSI Trading Signals
  - Overbought (Sell): 6,562 samples (58.1%)
  - Oversold (Buy): 4,731 samples (41.9%)
- **Feature Count:** 18 engineered features
- **Train/Test Split:** 80/20 temporal split (33,076 train / 8,270 test)

## Model Performance Comparison

| Model               | Accuracy | Precision | Recall | F1-Score | CV Mean | CV Std |
|---------------------|----------|-----------|---------|----------|---------|---------|
| **Gradient Boosting** | 0.6631   | 0.6787    | 0.6631  | **0.6510** | 0.7174  | 0.0483  |
| Random Forest       | 0.6612   | 0.6785    | 0.6612  | 0.6479   | 0.7167  | 0.0469  |
| Extra Trees         | 0.6482   | 0.6485    | 0.6482  | 0.6463   | 0.7135  | 0.0247  |
| Logistic Regression | 0.6412   | 0.6414    | 0.6412  | 0.6413   | 0.7286  | 0.0235  |

## Best Model: Gradient Boosting Classifier

### Performance Metrics
- **Overall Accuracy:** 66.31%
- **Weighted F1-Score:** 65.10%
- **Cross-Validation:** 71.74% (±4.83%)

### Class-Specific Performance
**Overbought (Sell) Signals:**
- Precision: 63.4%
- Recall: 83.6%
- F1-Score: 72.1%
- Support: 4,305 samples

**Oversold (Buy) Signals:**
- Precision: 72.8%
- Recall: 47.5%
- F1-Score: 57.5%
- Support: 3,965 samples

### Trading Performance Analysis

**Signal Prediction Accuracy:**
- **Sell Signal Detection:** 83.6% accuracy (3,599 out of 4,305 actual sell signals)
- **Buy Signal Detection:** 47.5% accuracy (1,885 out of 3,965 actual buy signals)

**Prediction Confidence Distribution:**
- **High Confidence (>70%):** 47.8% of predictions with 79.7% accuracy
- **Medium Confidence (60-70%):** 27.2% of predictions
- **Low Confidence (≤60%):** 25.0% of predictions

## Feature Importance Analysis

**Top 10 Most Important Features:**

1. **RSI (62.75%)** - Primary technical indicator
2. **Month (14.75%)** - Seasonal patterns
3. **RSI Momentum (6.57%)** - Rate of RSI change
4. **Gap (3.47%)** - Price gap from previous close
5. **Close Price (1.45%)** - Stock price level
6. **Daily Volatility (1.38%)** - Intraday price range
7. **Low Price (1.36%)** - Support levels
8. **High Price (1.11%)** - Resistance levels
9. **Open Price (1.10%)** - Opening price
10. **Price Range (1.00%)** - High-low spread

## Key Insights

### Strengths
1. **Strong Sell Signal Detection:** Model excels at identifying overbought conditions (83.6% recall)
2. **High Confidence Predictions:** Nearly 80% accuracy for high-confidence predictions
3. **Feature Relevance:** RSI and temporal features provide strong predictive power
4. **Robust Cross-Validation:** Consistent performance across time periods

### Areas for Improvement
1. **Buy Signal Sensitivity:** Only 47.5% recall for oversold (buy) signals
2. **Class Imbalance:** Model biased toward sell signals due to class distribution
3. **Medium/Low Confidence:** 52.2% of predictions have moderate to low confidence

### Trading Strategy Implications
1. **Conservative Approach:** Model is more reliable for identifying selling opportunities
2. **Risk Management:** Use high-confidence predictions for trading decisions
3. **Signal Filtering:** Consider additional technical indicators for buy signals
4. **Seasonal Effects:** Month-based patterns should be factored into trading decisions

## Technical Implementation

### Data Pipeline
- **Database Connection:** SQLAlchemy with pyodbc for SQL Server
- **Feature Engineering:** 9 additional technical indicators created
- **Data Preprocessing:** StandardScaler normalization, temporal splits
- **Model Training:** Balanced class weights, time-series cross-validation

### Model Configuration
```python
GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6,
    random_state=42
)
```

### Saved Artifacts
- **Best Model:** `data/best_model_gradient_boosting.joblib`
- **Preprocessing:** `data/scaler.joblib`, `data/target_encoder.joblib`
- **Results:** `data/model_results.pkl`
- **Visualizations:** `reports/confusion_matrix.png`, `reports/feature_importance.png`

## Recommendations

### For Production Deployment
1. **Implement Confidence Thresholds:** Only act on predictions with >70% confidence
2. **Combine with Other Indicators:** Use additional technical analysis for buy signals
3. **Monitor Performance:** Set up real-time model monitoring and retraining schedules
4. **Risk Controls:** Implement position sizing based on prediction confidence

### For Model Improvement
1. **Feature Engineering:** Add moving averages, momentum indicators, volume patterns
2. **Ensemble Methods:** Combine multiple models for better buy signal detection
3. **Hyperparameter Tuning:** Optimize model parameters using grid search
4. **Extended Timeline:** Include more historical data for better generalization

### For Trading Strategy
1. **Focus on Sell Signals:** Leverage model's strength in overbought detection
2. **Manual Buy Decisions:** Use fundamental analysis for oversold opportunities
3. **Seasonal Adjustments:** Factor in monthly patterns for timing decisions
4. **Portfolio Management:** Scale positions based on prediction confidence levels

## Conclusion

The Gradient Boosting model successfully demonstrates the feasibility of using machine learning for RSI-based trading signal prediction. With 66.31% accuracy and strong sell signal detection capabilities, the model provides valuable insights for systematic trading strategies. The analysis reveals clear patterns in the data and establishes a solid foundation for automated trading signal generation.

**Next Steps:** Deploy the model for paper trading, monitor performance, and iteratively improve based on real-world results.

---
*Report generated from SQL Server ML project using Python, scikit-learn, and VS Code*
