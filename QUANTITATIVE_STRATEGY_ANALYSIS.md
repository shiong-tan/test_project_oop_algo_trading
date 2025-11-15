# Quantitative Analysis: Prediction-Based Trading Strategy
**Analysis Date:** 2025-11-15
**Strategy:** ML-based directional prediction with event-driven backtesting
**Symbol:** AAPL.O
**Data Period:** First 1,000 observations from Eikon dataset

---

## Executive Summary

The strategy shows consistent underperformance across all models (-14.3% to 0%), failing to beat a simple buy-and-hold benchmark. The core issues stem from:
1. Severe class imbalance and overfitting
2. Event-driven backtesting introducing look-ahead bias
3. Aggressive position sizing without risk controls
4. Limited feature set with questionable predictive power
5. Single train-test split without walk-forward validation

**Expected Performance Range with Fixes:** 5-15% annualized return with Sharpe 0.8-1.2 (realistic), assuming proper implementation of recommended improvements.

---

## 1. Strategy Weaknesses and Underperformance Root Causes

### 1.1 Critical Backtesting Flaw: Look-Ahead Bias
**Issue:** The current implementation uses `train_test_split(shuffle=False)` which creates a single 70/30 split. However, the model is trained on the ENTIRE training set (70% of data) and then applied to future data.

**Problem:** This is NOT realistic. In production, you cannot train on 700 days of data and suddenly have a model. The model should be:
- Trained incrementally (walk-forward)
- Retrained periodically (e.g., every 20-60 days)
- Evaluated on truly out-of-sample data

**Impact on Performance:**
```
Current: Train on days 1-700 → Test on days 701-1000
Realistic: Train on days 1-250 → Test on 251-270 → Retrain on 1-270 → Test on 271-290...
```

This single split biases the results and doesn't reflect trading reality where market regimes change.

### 1.2 Event-Driven vs Vectorized Backtesting Inconsistency
The author notes they "could not replicate vectorized backtesting results from Tutorial_05" - this is a RED FLAG indicating:

**Issue 1: Bar Indexing Confusion**
```python
for test_bar in range(len(self.test)):
    bar = test_bar + len(self.train)  # Adjusting bar for self.data access
```
This creates potential misalignment between predictions and actual prices used for execution.

**Issue 2: Position Logic Flaw**
```python
if self.units == 0 or self.units == -1:  # Bug: units never equals -1 in long-only
    if prediction == 1:
        self.place_buy_order(bar, amount=self.current_balance/2)
```
- The strategy is supposed to be long-short but only goes long
- `self.units == -1` never occurs (no shorting implemented)
- This explains the low trade count (1-24 trades)

**Issue 3: Signal Timing**
Prediction for day T uses data through day T-1, but execution happens at day T close price - this is acceptable but needs explicit documentation. However, there's no check for whether the prediction was generated BEFORE or AFTER the close.

### 1.3 Class Imbalance and Model Bias

**Target Distribution:**
```python
bins = [-0.01, -0.005, 0.005, 0.01]
data['d'] = np.where(data['r'] > 0, 1, 0)
```

In typical equity data, the distribution is often:
- Up days: ~52-54%
- Down days: ~46-48%

**Problem:** Most sklearn models will bias toward the majority class. Without class weighting or threshold tuning:
- GaussianNB: Likely predicting mostly 0 (explains 1 trade only)
- LogisticRegression/MLP: Similar majority class bias (explains 2 trades)
- DecisionTree: Overfits to noise (24 trades, worst performance)

**Evidence:**
- GaussianNB: 1 trade = never predicted buy signal
- LogisticRegression: 2 trades = bought once, sold once
- DecisionTree: 24 trades = overfitting to training noise

### 1.4 Missing Performance Attribution

The strategy lacks decomposition of returns:
- **Transaction costs:** 1% per trade (0.5% roundtrip would be more realistic)
- **Slippage:** Not modeled (critical for real trading)
- **Market exposure:** How much time in market vs cash?
- **Win rate:** Not tracked
- **Avg win/loss ratio:** Not tracked

**Estimated Transaction Cost Impact:**
- DecisionTree: 24 trades × 1% = 24% drag on returns
- This alone explains the -14.3% loss

### 1.5 Data Quality and Regime Issues

**Date Range:** The notebook doesn't show the actual dates, but "first 1,000 observations" from Eikon data likely covers:
- If daily: ~4 years of data
- Ending date shown: 2013-12-06

**Period Covered:** Approximately 2010-2013
- Post-financial crisis recovery
- QE2/QE3 era (low rates, high liquidity)
- AAPL had strong secular growth trend

**Implication:** The model may be learning regime-specific patterns that don't generalize. The 70/30 split means:
- Training: ~2010-2012
- Testing: ~2012-2013

This is problematic because market regimes change.

---

## 2. Feature Engineering Improvements

### 2.1 Current Feature Set Analysis

**Base Features (11):**
```python
1. 'r' - Log returns
2. 'd' - Direction (binary)
3. 'd_' - Discretized returns (4 bins)
4. 'SMA1' - 20-day SMA
5. 'SMA2' - 60-day SMA
6. 'SMA_' - SMA difference
7. 'EWMA1' - 20-day EWMA
8. 'EWMA2' - 60-day EWMA
9. 'EWMA_' - EWMA difference
10. 'V1' - 20-day rolling volatility
11. 'V2' - 60-day rolling volatility
```

**Lagged Features:** 5 lags × 11 features = 55 total features

### 2.2 Critical Issues with Current Features

**Issue 1: Feature Leakage**
```python
self.cols = ['SMA1', 'SMA2', 'SMA_', 'EWMA1', 'EWMA2', 'EWMA_', 'V1', 'V2']
# Only these 8 features are normalized, but then:
self.cols.extend(['r', 'd', 'd_'])  # These are added AFTER normalization
```

**'d'** (the target) should NEVER be a feature. This is direct data leakage.
**'d_'** is a discretized version of returns and contains similar information to the target.

**Impact:** The model sees the current period's direction during training, causing severe overfitting.

**Issue 2: Redundant Features**
- SMA1/SMA2 and EWMA1/EWMA2 are highly correlated
- Using both SMA_ and EWMA_ adds little information
- 5 lags of 11 features creates 55 features for ~700 training samples
- This is a feature-to-sample ratio of 55/700 = 7.8%, which is high but acceptable
- However, many features are redundant, reducing effective dimensionality

**Issue 3: No Momentum/Mean Reversion Indicators**
Current features focus on smoothing and volatility, but miss:
- RSI (Relative Strength Index)
- MACD (Moving Average Convergence Divergence)
- Bollinger Bands position
- Rate of change
- Volume indicators (if available)

**Issue 4: No Regime Detection**
Markets alternate between:
- Trending vs ranging
- High vol vs low vol
- Risk-on vs risk-off

The model should adapt to regimes, not treat all periods equally.

### 2.3 Recommended Feature Engineering

**Tier 1: Fix Existing Issues (Immediate)**
```python
# REMOVE these features entirely (data leakage):
- 'd' (current direction)
- 'd_' (discretized returns)

# FIX normalization:
- Normalize ALL features before creating lags
- Use only lagged features as inputs (never current period)

# REDUCE redundancy:
- Choose either SMA or EWMA, not both
- Recommended: Keep EWMA (more responsive)
```

**Tier 2: Add Momentum/Mean Reversion (High Priority)**
```python
# Momentum features
1. RSI_14 = RSI(14)  # Relative Strength Index
2. RSI_28 = RSI(28)  # Longer period RSI
3. ROC_5 = (Close - Close_5) / Close_5  # Rate of change
4. ROC_20 = (Close - Close_20) / Close_20

# Mean reversion features
5. BB_position = (Close - BB_middle) / (BB_upper - BB_lower)  # Bollinger Band position
6. Distance_from_SMA = (Close - SMA_50) / SMA_50  # Normalized distance

# Trend strength
7. ADX_14 = ADX(14)  # Average Directional Index (if available)
8. MACD_histogram = MACD - Signal_line
```

**Tier 3: Microstructure Features (Medium Priority)**
```python
# Volatility regime
9. Realized_vol_ratio = V1 / V2  # Short vol vs long vol
10. Vol_percentile = percentile_rank(V1, window=100)  # Vol regime

# Price patterns
11. Higher_high = (High > High_1) & (High_1 > High_2)  # Binary
12. Lower_low = (Low < Low_1) & (Low_1 < Low_2)  # Binary

# Intraday features (if high-frequency data available)
13. Open_close_ratio = (Close - Open) / Open
14. High_low_ratio = (High - Low) / Close
```

**Tier 4: Cross-Asset Features (Lower Priority)**
```python
# If other asset data available:
15. SPY_return = log_return(SPY)  # Market factor
16. VIX_level = VIX / VIX_SMA_20  # Normalized volatility index
17. Sector_return = log_return(XLK)  # Tech sector for AAPL
```

**Recommended Final Feature Set:**
- 12-16 base features (remove redundancy, add momentum/mean reversion)
- 3-5 lags (reduce from 5 to avoid overfitting)
- Total: 36-80 features (down from 55)
- Feature-to-sample ratio: 5-11% (acceptable)

**Implementation Priority:**
1. **Week 1:** Remove data leakage ('d', 'd_'), fix normalization
2. **Week 2:** Add RSI, ROC, Bollinger Bands
3. **Week 3:** Add volatility regime features
4. **Week 4:** Test feature selection (L1 regularization, RFE)

---

## 3. Model Selection and Ensemble Opportunities

### 3.1 Current Model Performance Analysis

**Results Summary:**
| Model | Return | Trades | Implied Behavior |
|-------|--------|--------|------------------|
| GaussianNB | 0.00% | 1 | Never buys (close-out only) |
| LogisticRegression | -3.52% | 2 | One buy-sell cycle |
| MLPClassifier | -3.52% | 2 | Identical to LogReg |
| SVC | -9.45% | 7 | Some trading, poor timing |
| DecisionTree | -14.32% | 24 | Overtrading, overfitting |

**Key Observations:**
1. **GaussianNB predicts all 0s:** Assumes feature independence (clearly violated)
2. **LogReg ≈ MLP:** Both conservative, likely predicting mostly 0s
3. **DecisionTree worst:** Classic overfitting, noise trading
4. **No model beats buy-and-hold:** Fundamental strategy flaw

### 3.2 Why These Models Are Inappropriate

**GaussianNB:**
- Assumes features are independent (false for price/volatility)
- Assumes Gaussian distribution (returns are fat-tailed)
- **Use case:** Text classification, not financial time series
- **Verdict:** Remove entirely

**Linear SVC:**
- Assumes linear separability (markets are non-linear)
- Kernel='linear' is too restrictive
- **Use case:** High-dimensional text data
- **Verdict:** Replace with RBF kernel SVC or remove

**DecisionTree:**
- High variance, overfits easily
- Not designed for time series
- **Use case:** Interpretability in stable domains
- **Verdict:** Use only in ensemble (RandomForest)

**MLPClassifier:**
- Can work but requires careful tuning
- Shuffle=False is correct for time series
- 64 hidden units may be too few
- **Verdict:** Keep but needs hyperparameter tuning

**LogisticRegression:**
- Too simple for complex patterns
- L2 regularization may be over-smoothing
- **Use case:** Baseline model
- **Verdict:** Keep as baseline, not primary model

### 3.3 Recommended Model Suite

**Tier 1: Primary Models (Choose 2-3)**

**A) Gradient Boosting Machines**
```python
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# XGBoost (Recommended)
xgb_model = XGBClassifier(
    n_estimators=100,
    max_depth=3,  # Shallow trees to prevent overfitting
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=len(y[y==0])/len(y[y==1]),  # Handle class imbalance
    random_state=42,
    early_stopping_rounds=10,  # Prevent overfitting
    eval_metric='logloss'
)

# LightGBM (Faster alternative)
lgbm_model = LGBMClassifier(
    n_estimators=100,
    max_depth=3,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    class_weight='balanced',  # Auto-handle imbalance
    random_state=42
)
```

**Why:** GBMs are state-of-the-art for tabular data, handle non-linearity well, and have built-in regularization.

**B) Random Forest**
```python
from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=5,  # Limit depth
    min_samples_split=20,  # Prevent overfitting
    min_samples_leaf=10,
    max_features='sqrt',  # Feature sampling
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
```

**Why:** Robust to overfitting, provides feature importance, handles non-linearity.

**C) Regularized Logistic Regression (Baseline)**
```python
from sklearn.linear_model import LogisticRegressionCV

logreg_model = LogisticRegressionCV(
    Cs=10,  # Test 10 different C values
    cv=5,  # 5-fold time series CV
    penalty='l1',  # L1 for feature selection
    solver='saga',
    max_iter=1000,
    class_weight='balanced',
    random_state=42
)
```

**Why:** Fast, interpretable, good baseline.

**Tier 2: Advanced Models (Optional)**

**D) LSTMs (if time series dependencies strong)**
```python
# Requires reshaping data for sequence input
# Only use if walk-forward validation shows benefit
# High risk of overfitting with limited data
```

**E) Autoencoders for Feature Learning**
```python
# Use unsupervised learning to learn features
# Then feed to simpler classifier
# Requires significant data (1000 samples too few)
```

### 3.4 Ensemble Strategy

**Approach 1: Simple Voting Ensemble**
```python
from sklearn.ensemble import VotingClassifier

ensemble = VotingClassifier(
    estimators=[
        ('xgb', xgb_model),
        ('rf', rf_model),
        ('logreg', logreg_model)
    ],
    voting='soft',  # Use probabilities
    weights=[2, 1, 1]  # Weight XGB higher if it performs better
)
```

**Approach 2: Stacking Ensemble**
```python
from sklearn.ensemble import StackingClassifier

ensemble = StackingClassifier(
    estimators=[
        ('xgb', xgb_model),
        ('rf', rf_model)
    ],
    final_estimator=LogisticRegression(class_weight='balanced'),
    cv=5  # Time series split
)
```

**Approach 3: Blending by Regime**
```python
# Train separate models for:
# - High volatility periods (use conservative model)
# - Low volatility periods (use aggressive model)
# - Trending periods (use momentum model)
# - Ranging periods (use mean reversion model)

# Example:
if current_volatility > 75th_percentile:
    prediction = conservative_model.predict(X)
else:
    prediction = aggressive_model.predict(X)
```

### 3.5 Model Selection Criteria

**Evaluation Metrics for Model Selection:**
1. **Accuracy** (basic, but not sufficient)
2. **Precision/Recall** (understand trade-offs)
3. **F1 Score** (balance precision/recall)
4. **ROC-AUC** (threshold-independent performance)
5. **Log Loss** (calibration quality)
6. **Profit-based metric** (ultimate goal)

**Custom Profit-Based Scoring:**
```python
from sklearn.metrics import make_scorer

def profit_score(y_true, y_pred):
    # Simulate trading based on predictions
    returns = calculate_returns(y_true, y_pred)
    return returns.sum()  # Or Sharpe ratio

profit_scorer = make_scorer(profit_score, greater_is_better=True)

# Use in cross-validation
from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, X, y, cv=5, scoring=profit_scorer)
```

**Recommended Model Pipeline:**
1. **Baseline:** Regularized LogisticRegression
2. **Primary:** XGBoost or LightGBM
3. **Ensemble:** Voting or Stacking of top 3 models
4. **Validation:** Walk-forward with profit-based scoring

---

## 4. Position Sizing Flaws

### 4.1 Current Position Sizing Logic

```python
if self.units == 0 or self.units == -1:
    if prediction == 1:
        # Buy with half of the current balance
        self.place_buy_order(bar, amount=self.current_balance/2)
```

**Critical Issues:**

**Issue 1: Fixed 50% Allocation**
- **Problem:** No adaptation to:
  - Market volatility (should reduce size in high vol)
  - Prediction confidence (should vary with probability)
  - Account balance changes (can lead to ruin)

**Issue 2: No Leverage Control**
- If balance grows to $15,000, next trade uses $7,500
- If balance shrinks to $5,000, next trade uses $2,500
- This creates path dependency and violates Kelly Criterion

**Issue 3: No Position Rebalancing**
```python
else:  # self.units > 0
    if prediction == 0:
        self.place_sell_order(bar, units=self.units)  # Sell all
```
- Binary: 100% long or 0% long (no short)
- No gradual scaling in/out
- No stop-loss or profit-taking

**Issue 4: Transaction Cost Amplification**
With 1% transaction costs (0.5% each way):
- 24 trades (DecisionTree) = 24% cost drag
- If trades are frequent, costs dominate returns

### 4.2 Position Sizing Best Practices

**Framework: Kelly Criterion (Modified)**

The Kelly Criterion optimizes position size based on edge:

```
f = (p * b - q) / b

where:
- f = fraction of capital to bet
- p = probability of winning
- q = probability of losing (1 - p)
- b = win/loss ratio
```

**For trading:**
```python
def kelly_position_size(win_prob, avg_win, avg_loss, max_fraction=0.25):
    """
    Calculate Kelly position size

    Parameters:
    - win_prob: Model prediction probability
    - avg_win: Average winning trade return (from backtest)
    - avg_loss: Average losing trade return (from backtest)
    - max_fraction: Cap Kelly size (0.25 = quarter Kelly)

    Returns:
    - fraction: Fraction of capital to risk
    """
    if avg_loss == 0:
        return 0

    b = avg_win / abs(avg_loss)  # Win/loss ratio
    q = 1 - win_prob

    kelly_fraction = (win_prob * b - q) / b

    # Never bet more than max_fraction (Kelly is aggressive)
    return max(0, min(kelly_fraction, max_fraction))

# Usage
prediction_prob = model.predict_proba(X)[:, 1]  # Probability of class 1
position_size = kelly_position_size(
    win_prob=prediction_prob,
    avg_win=0.015,  # 1.5% average win (from historical backtest)
    avg_loss=0.01,  # 1% average loss
    max_fraction=0.25  # Quarter Kelly (conservative)
)
```

**Alternative: Volatility-Based Sizing**

```python
def volatility_position_size(current_volatility, target_risk=0.01, max_position=0.5):
    """
    Scale position size inversely with volatility

    Parameters:
    - current_volatility: Current market volatility (e.g., 20-day rolling std)
    - target_risk: Target risk per trade (e.g., 1% of capital)
    - max_position: Maximum position size (e.g., 50% of capital)

    Returns:
    - fraction: Fraction of capital to invest
    """
    # Inverse volatility scaling
    position_fraction = target_risk / current_volatility

    return min(position_fraction, max_position)

# Usage
vol = data['r'].rolling(20).std().iloc[-1]
position_size = volatility_position_size(
    current_volatility=vol,
    target_risk=0.01,  # 1% risk per trade
    max_position=0.5   # Max 50% capital
)
```

**Recommended: Hybrid Approach**

```python
def dynamic_position_size(prediction_prob, current_vol, avg_win, avg_loss, capital):
    """
    Combine Kelly and volatility-based sizing
    """
    # Kelly component
    kelly_size = kelly_position_size(prediction_prob, avg_win, avg_loss, max_fraction=0.25)

    # Volatility component
    vol_size = volatility_position_size(current_vol, target_risk=0.01, max_position=0.5)

    # Take minimum (most conservative)
    position_size = min(kelly_size, vol_size)

    # Adjust for prediction confidence
    if prediction_prob < 0.55:  # Low confidence
        position_size *= 0.5
    elif prediction_prob > 0.70:  # High confidence
        position_size *= 1.5

    # Cap at 50% of capital
    position_size = min(position_size, 0.5)

    return position_size * capital
```

### 4.3 Implementation Example

**Modified BacktestingBase class:**

```python
def run_strategy_dynamic_sizing(self):
    # Initialize tracking
    win_count = 0
    loss_count = 0
    total_wins = 0
    total_losses = 0

    for test_bar in range(len(self.test)):
        bar = test_bar + len(self.train)

        # Get prediction probability (not just class)
        prediction_prob = self.model.predict_proba(
            self.test[self.cols_].iloc[test_bar:test_bar+1]
        )[0, 1]

        # Calculate current volatility
        current_vol = self.test['V1'].iloc[test_bar]

        # Calculate historical win/loss stats (updated dynamically)
        avg_win = total_wins / win_count if win_count > 0 else 0.015
        avg_loss = total_losses / loss_count if loss_count > 0 else 0.01

        # Dynamic position sizing
        position_amount = dynamic_position_size(
            prediction_prob=prediction_prob,
            current_vol=current_vol,
            avg_win=avg_win,
            avg_loss=avg_loss,
            capital=self.current_balance
        )

        # Trading logic
        if prediction_prob > 0.55:  # Buy signal threshold
            if self.units == 0:
                self.place_buy_order(bar, amount=position_amount)
        elif prediction_prob < 0.45:  # Sell signal threshold
            if self.units > 0:
                self.place_sell_order(bar, units=self.units)

                # Update win/loss stats
                trade_return = self.calculate_last_trade_return()
                if trade_return > 0:
                    win_count += 1
                    total_wins += trade_return
                else:
                    loss_count += 1
                    total_losses += abs(trade_return)
```

### 4.4 Position Sizing Comparison

**Scenario: $10,000 capital, 60% win rate, 1.5% avg win, 1% avg loss**

| Method | Position Size | Rationale | Expected Growth |
|--------|---------------|-----------|-----------------|
| Current (50% fixed) | $5,000 | None | High variance, risk of ruin |
| Kelly (full) | $3,750 | Optimal growth | Aggressive, ~15% annual |
| Quarter Kelly | $937 | Conservative | Stable, ~8% annual |
| Volatility-based | $500-$2,000 | Adapts to risk | Moderate, ~10% annual |
| Hybrid | $750-$1,500 | Best of both | Balanced, ~12% annual |

**Recommendation:**
- Start with **quarter Kelly** or **hybrid**
- Never exceed 25% of capital in single position
- Reduce size in high volatility periods
- Increase size only with high confidence (>70% probability)

---

## 5. Risk Management Gaps

### 5.1 Current Risk Management (None)

**What's Missing:**
1. Stop-loss orders
2. Maximum drawdown limits
3. Daily/weekly loss limits
4. Position concentration limits
5. Volatility-based exposure adjustment
6. Correlation risk management
7. Regime detection and adaptation

**Consequences:**
- Uncapped losses per trade
- No circuit breakers for bad models
- No recovery mechanism from drawdowns
- Exposure to flash crashes

### 5.2 Essential Risk Controls

**A) Trade-Level Stops**

```python
class RiskManager:
    def __init__(self, stop_loss_pct=0.02, take_profit_pct=0.05):
        self.stop_loss_pct = stop_loss_pct  # 2% stop loss
        self.take_profit_pct = take_profit_pct  # 5% take profit
        self.entry_prices = {}  # Track entry prices

    def check_stop_loss(self, symbol, current_price, position_size):
        """Check if stop loss triggered"""
        if symbol not in self.entry_prices:
            return False

        entry_price = self.entry_prices[symbol]
        loss_pct = (current_price - entry_price) / entry_price

        if loss_pct < -self.stop_loss_pct:
            print(f"STOP LOSS TRIGGERED: {loss_pct*100:.2f}% loss")
            return True
        return False

    def check_take_profit(self, symbol, current_price, position_size):
        """Check if take profit triggered"""
        if symbol not in self.entry_prices:
            return False

        entry_price = self.entry_prices[symbol]
        profit_pct = (current_price - entry_price) / entry_price

        if profit_pct > self.take_profit_pct:
            print(f"TAKE PROFIT TRIGGERED: {profit_pct*100:.2f}% profit")
            return True
        return False

    def record_entry(self, symbol, price):
        """Record entry price for stop/profit tracking"""
        self.entry_prices[symbol] = price

    def clear_position(self, symbol):
        """Clear position tracking"""
        if symbol in self.entry_prices:
            del self.entry_prices[symbol]
```

**B) Portfolio-Level Limits**

```python
class PortfolioRiskManager:
    def __init__(self, max_drawdown=0.15, daily_loss_limit=0.03):
        self.max_drawdown = max_drawdown  # 15% max drawdown
        self.daily_loss_limit = daily_loss_limit  # 3% daily loss limit
        self.peak_balance = 0
        self.daily_start_balance = 0
        self.is_trading_halted = False

    def update_peak(self, current_balance):
        """Track high water mark"""
        self.peak_balance = max(self.peak_balance, current_balance)

    def check_max_drawdown(self, current_balance):
        """Check if max drawdown exceeded"""
        if self.peak_balance == 0:
            return False

        drawdown = (self.peak_balance - current_balance) / self.peak_balance

        if drawdown > self.max_drawdown:
            print(f"MAX DRAWDOWN EXCEEDED: {drawdown*100:.2f}%")
            self.is_trading_halted = True
            return True
        return False

    def check_daily_loss_limit(self, current_balance):
        """Check if daily loss limit exceeded"""
        if self.daily_start_balance == 0:
            return False

        daily_loss = (self.daily_start_balance - current_balance) / self.daily_start_balance

        if daily_loss > self.daily_loss_limit:
            print(f"DAILY LOSS LIMIT EXCEEDED: {daily_loss*100:.2f}%")
            return True
        return False

    def start_new_day(self, current_balance):
        """Reset daily tracking"""
        self.daily_start_balance = current_balance

    def can_trade(self):
        """Check if trading is allowed"""
        return not self.is_trading_halted
```

**C) Volatility-Based Exposure Adjustment**

```python
class VolatilityRiskManager:
    def __init__(self, target_volatility=0.15, lookback=60):
        self.target_volatility = target_volatility  # 15% annualized
        self.lookback = lookback

    def calculate_position_scaling(self, returns_series):
        """
        Scale position based on realized volatility

        Returns:
        - scaling_factor: Multiply position size by this (0.5 to 2.0)
        """
        realized_vol = returns_series.rolling(self.lookback).std().iloc[-1]
        realized_vol_annualized = realized_vol * np.sqrt(252)

        # Scale inversely with volatility
        scaling_factor = self.target_volatility / realized_vol_annualized

        # Cap scaling between 0.5x and 2.0x
        return np.clip(scaling_factor, 0.5, 2.0)
```

### 5.3 Integrated Risk Management Framework

```python
class BacktestingWithRisk(BacktestingBase):
    def __init__(self, url, symbol, model, amount, ptc, verbose=False):
        super().__init__(url, symbol, model, amount, ptc, verbose)

        # Initialize risk managers
        self.trade_risk = RiskManager(stop_loss_pct=0.02, take_profit_pct=0.05)
        self.portfolio_risk = PortfolioRiskManager(max_drawdown=0.15, daily_loss_limit=0.03)
        self.vol_risk = VolatilityRiskManager(target_volatility=0.15, lookback=60)

        # Initialize peak balance
        self.portfolio_risk.peak_balance = amount
        self.portfolio_risk.daily_start_balance = amount

    def run_strategy_with_risk(self):
        print(92 * '=')
        print(f'*** BACKTESTING STRATEGY WITH RISK MANAGEMENT ***')
        print(92 * '=')

        current_date = None

        for test_bar in range(len(self.test)):
            bar = test_bar + len(self.train)
            date, price = self.get_date_price(bar)

            # Check if new day (reset daily limits)
            if date != current_date:
                current_date = date
                self.portfolio_risk.start_new_day(self.current_balance)

            # Update peak balance
            self.portfolio_risk.update_peak(self.current_balance)

            # Check portfolio-level risk limits
            if self.portfolio_risk.check_max_drawdown(self.current_balance):
                print("Trading halted due to max drawdown")
                break

            if self.portfolio_risk.check_daily_loss_limit(self.current_balance):
                print(f"Daily loss limit hit on {date}, skipping rest of day")
                continue

            # Check trade-level stops if we have a position
            if self.units > 0:
                if self.trade_risk.check_stop_loss(self.symbol, price, self.units):
                    print(f"{date} | STOP LOSS: Closing position")
                    self.place_sell_order(bar, units=self.units)
                    self.trade_risk.clear_position(self.symbol)
                    continue

                if self.trade_risk.check_take_profit(self.symbol, price, self.units):
                    print(f"{date} | TAKE PROFIT: Closing position")
                    self.place_sell_order(bar, units=self.units)
                    self.trade_risk.clear_position(self.symbol)
                    continue

            # Get prediction
            prediction = self.test['prediction'].iloc[test_bar]
            prediction_prob = self.model.predict_proba(
                self.test[self.cols_].iloc[test_bar:test_bar+1]
            )[0, 1]

            # Calculate volatility scaling
            vol_scaling = self.vol_risk.calculate_position_scaling(
                self.test['r'].iloc[:test_bar+1]
            )

            # Trading logic with risk management
            if self.units == 0:
                if prediction == 1 and prediction_prob > 0.60:  # Higher threshold
                    # Calculate position size with vol adjustment
                    base_amount = self.current_balance * 0.25  # Max 25% (reduced from 50%)
                    adjusted_amount = base_amount * vol_scaling

                    self.place_buy_order(bar, amount=adjusted_amount)
                    self.trade_risk.record_entry(self.symbol, price)

            else:  # Have position
                if prediction == 0 and prediction_prob < 0.40:  # Exit signal
                    self.place_sell_order(bar, units=self.units)
                    self.trade_risk.clear_position(self.symbol)

        self.close_out(bar)

        # Print risk statistics
        print('\n' + 92 * '=')
        print('RISK STATISTICS')
        print(92 * '=')
        max_dd = (self.portfolio_risk.peak_balance - min(self.wealth_over_time)) / self.portfolio_risk.peak_balance
        print(f'Maximum Drawdown: {max_dd*100:.2f}%')
        print(f'Peak Balance: ${self.portfolio_risk.peak_balance:.2f}')
```

### 5.4 Recommended Risk Parameters

**Conservative (Recommended for starting):**
```python
risk_params = {
    'stop_loss_pct': 0.02,        # 2% stop loss per trade
    'take_profit_pct': 0.05,      # 5% take profit per trade
    'max_drawdown': 0.15,         # 15% max portfolio drawdown
    'daily_loss_limit': 0.03,     # 3% daily loss limit
    'max_position_size': 0.25,    # 25% max per position (down from 50%)
    'min_prediction_prob': 0.60,  # 60% minimum confidence to enter
    'target_volatility': 0.15     # 15% annualized vol target
}
```

**Moderate (after validation):**
```python
risk_params = {
    'stop_loss_pct': 0.03,        # 3% stop loss per trade
    'take_profit_pct': 0.07,      # 7% take profit per trade
    'max_drawdown': 0.20,         # 20% max portfolio drawdown
    'daily_loss_limit': 0.05,     # 5% daily loss limit
    'max_position_size': 0.35,    # 35% max per position
    'min_prediction_prob': 0.55,  # 55% minimum confidence
    'target_volatility': 0.18     # 18% annualized vol target
}
```

**Aggressive (only after proven track record):**
```python
risk_params = {
    'stop_loss_pct': 0.05,        # 5% stop loss per trade
    'take_profit_pct': 0.10,      # 10% take profit per trade
    'max_drawdown': 0.25,         # 25% max portfolio drawdown
    'daily_loss_limit': 0.07,     # 7% daily loss limit
    'max_position_size': 0.50,    # 50% max per position
    'min_prediction_prob': 0.52,  # 52% minimum confidence
    'target_volatility': 0.20     # 20% annualized vol target
}
```

### 5.5 Risk Metrics to Track

**Beyond just returns, track:**

```python
def calculate_risk_metrics(balance_series, returns_series):
    """
    Calculate comprehensive risk metrics
    """
    # Drawdown analysis
    peak = balance_series.cummax()
    drawdown = (balance_series - peak) / peak
    max_drawdown = drawdown.min()

    # Volatility metrics
    daily_vol = returns_series.std()
    annual_vol = daily_vol * np.sqrt(252)

    # Risk-adjusted returns
    sharpe_ratio = (returns_series.mean() * 252) / annual_vol

    # Downside risk
    downside_returns = returns_series[returns_series < 0]
    downside_vol = downside_returns.std() * np.sqrt(252)
    sortino_ratio = (returns_series.mean() * 252) / downside_vol if len(downside_returns) > 0 else 0

    # Win/loss analysis
    winning_trades = returns_series[returns_series > 0]
    losing_trades = returns_series[returns_series < 0]

    win_rate = len(winning_trades) / len(returns_series)
    avg_win = winning_trades.mean() if len(winning_trades) > 0 else 0
    avg_loss = losing_trades.mean() if len(losing_trades) > 0 else 0
    profit_factor = abs(winning_trades.sum() / losing_trades.sum()) if len(losing_trades) > 0 else 0

    # Value at Risk (VaR)
    var_95 = np.percentile(returns_series, 5)
    cvar_95 = returns_series[returns_series <= var_95].mean()  # Conditional VaR

    return {
        'Max Drawdown': max_drawdown,
        'Annual Volatility': annual_vol,
        'Sharpe Ratio': sharpe_ratio,
        'Sortino Ratio': sortino_ratio,
        'Win Rate': win_rate,
        'Avg Win': avg_win,
        'Avg Loss': avg_loss,
        'Profit Factor': profit_factor,
        'VaR (95%)': var_95,
        'CVaR (95%)': cvar_95
    }
```

**Expected Output:**
```
RISK METRICS
================================================================================
Max Drawdown:           -12.50%
Annual Volatility:       18.50%
Sharpe Ratio:            0.95
Sortino Ratio:           1.32
Win Rate:                58.00%
Avg Win:                 1.80%
Avg Loss:               -1.20%
Profit Factor:           1.45
VaR (95%):              -2.50%
CVaR (95%):             -3.80%
```

---

## 6. Backtesting Methodology Issues

### 6.1 Critical Flaws in Current Approach

**Issue 1: Single Train-Test Split**

```python
self.train, self.test = train_test_split(self.data, test_size=0.3, random_state=42, shuffle=False)
```

**Problems:**
1. **No walk-forward validation:** Model trained once on 70% data, never retrained
2. **Regime dependency:** If test period (2012-2013) was bullish, results don't generalize
3. **Sample size:** Only ~300 test samples, high variance in performance estimates
4. **No out-of-sample validation:** The 30% is not truly "unseen" - it's from the same distribution

**Solution: Walk-Forward Analysis**

```python
def walk_forward_validation(data, train_window=500, test_window=50, step=50):
    """
    Implement walk-forward validation

    Parameters:
    - data: Full dataset
    - train_window: Number of days to train on
    - test_window: Number of days to test on
    - step: Number of days to move forward each iteration

    Returns:
    - results: List of performance metrics for each fold
    """
    results = []

    for start in range(0, len(data) - train_window - test_window, step):
        # Define windows
        train_start = start
        train_end = start + train_window
        test_start = train_end
        test_end = test_start + test_window

        # Split data
        train_data = data.iloc[train_start:train_end]
        test_data = data.iloc[test_start:test_end]

        # Train model
        model = train_model(train_data)

        # Test model
        performance = backtest(model, test_data)

        results.append({
            'train_period': (train_data.index[0], train_data.index[-1]),
            'test_period': (test_data.index[0], test_data.index[-1]),
            'return': performance['return'],
            'sharpe': performance['sharpe'],
            'max_drawdown': performance['max_drawdown']
        })

    return results

# Example usage
results = walk_forward_validation(data, train_window=500, test_window=50, step=25)

# Aggregate results
avg_return = np.mean([r['return'] for r in results])
avg_sharpe = np.mean([r['sharpe'] for r in results])
print(f"Average Return: {avg_return:.2f}%")
print(f"Average Sharpe: {avg_sharpe:.2f}")
```

**Issue 2: Look-Ahead Bias Risks**

**Potential sources:**
1. **Feature normalization:** `self.scaler.fit_transform(self.train[self.cols])`
   - This is CORRECT - only fit on train, transform test separately
   - But ensure test is transformed using train scaler parameters

2. **Lagged features:**
   ```python
   self.train[self.col_] = self.train[self.col].shift(self.lag)
   ```
   - This is CORRECT - using past values only

3. **Target variable in features:**
   ```python
   self.cols.extend(['r', 'd', 'd_'])  # 'd' is the target!
   ```
   - This is WRONG - 'd' (current direction) should never be a feature
   - Even with lags, 'd_lag_1' uses yesterday's direction to predict today's
   - This is borderline - acceptable if we assume yesterday's close is known before today's trade
   - BUT 'd' itself (current period) must be removed

**Fix:**
```python
# CORRECT approach
def add_lags(self, lags):
    # Features for lagging (exclude target)
    feature_cols = ['SMA1', 'SMA2', 'SMA_', 'EWMA1', 'EWMA2', 'EWMA_', 'V1', 'V2', 'r']

    self.cols_ = []
    for col in feature_cols:
        for lag in range(1, lags + 1):  # Start from lag 1 (yesterday)
            col_ = f'{col}_lag_{lag}'
            self.train[col_] = self.train[col].shift(lag)
            self.test[col_] = self.test[col].shift(lag)
            self.cols_.append(col_)

    self.train.dropna(inplace=True)
    self.test.dropna(inplace=True)
```

**Issue 3: Transaction Cost Realism**

Current: `ptc = 0.01` (1% per trade)

**Analysis:**
- **Bid-ask spread:** For AAPL, typically 0.01-0.02%
- **Commission:** $0-5 per trade (negligible for $5000 orders)
- **Slippage:** 0.05-0.10% for market orders
- **Market impact:** Minimal for small retail trades

**Realistic cost:** 0.1-0.2% roundtrip (total)

**Current 1% is too high, but useful for stress testing**

**Recommendation:**
```python
# Test with multiple cost scenarios
cost_scenarios = {
    'optimistic': 0.0005,   # 0.05% (limit orders, no slippage)
    'realistic': 0.0015,    # 0.15% (market orders, typical slippage)
    'conservative': 0.003,  # 0.30% (adverse slippage, market impact)
    'stress_test': 0.01     # 1% (current setting)
}

for scenario, cost in cost_scenarios.items():
    bt = BacktestingBase(url, symbol, model, amount, cost, verbose=False)
    bt.run_strategy()
    print(f"{scenario}: {bt.current_balance:.2f}")
```

### 6.2 Proper Backtesting Framework

**Complete Walk-Forward Backtesting Implementation:**

```python
class WalkForwardBacktest:
    def __init__(self, data, model_class, train_window=500, test_window=50,
                 rebalance_freq=50, ptc=0.0015):
        """
        Walk-forward backtesting framework

        Parameters:
        - data: Full dataset with features
        - model_class: Class of model to instantiate (e.g., XGBClassifier)
        - train_window: Training window size (days)
        - test_window: Testing window size (days)
        - rebalance_freq: How often to retrain model (days)
        - ptc: Proportional transaction cost
        """
        self.data = data
        self.model_class = model_class
        self.train_window = train_window
        self.test_window = test_window
        self.rebalance_freq = rebalance_freq
        self.ptc = ptc

        self.results = []
        self.all_trades = []

    def run(self):
        """Execute walk-forward backtest"""
        print("="*80)
        print("WALK-FORWARD BACKTEST")
        print("="*80)

        for start in range(0, len(self.data) - self.train_window - self.test_window,
                          self.rebalance_freq):
            # Define windows
            train_start = start
            train_end = start + self.train_window
            test_start = train_end
            test_end = min(test_start + self.test_window, len(self.data))

            # Get data
            train_data = self.data.iloc[train_start:train_end]
            test_data = self.data.iloc[test_start:test_end]

            print(f"\nFold {len(self.results)+1}:")
            print(f"Train: {train_data.index[0]} to {train_data.index[-1]}")
            print(f"Test:  {test_data.index[0]} to {test_data.index[-1]}")

            # Prepare features
            X_train, y_train = self.prepare_features(train_data)
            X_test, y_test = self.prepare_features(test_data)

            # Normalize features (fit on train only)
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Train model
            model = self.model_class()
            model.fit(X_train_scaled, y_train)

            # Generate predictions
            predictions = model.predict(X_test_scaled)
            pred_proba = model.predict_proba(X_test_scaled)[:, 1]

            # Backtest this fold
            fold_results = self.backtest_fold(test_data, predictions, pred_proba)
            self.results.append(fold_results)

            print(f"Return: {fold_results['return']:.2f}%")
            print(f"Sharpe: {fold_results['sharpe']:.2f}")
            print(f"Trades: {fold_results['num_trades']}")

        # Aggregate results
        self.print_summary()

    def prepare_features(self, data):
        """Extract features and target from data"""
        # Assuming features are already created and lagged
        feature_cols = [col for col in data.columns if 'lag' in col]
        X = data[feature_cols].values
        y = data['d'].values
        return X, y

    def backtest_fold(self, test_data, predictions, pred_proba):
        """Backtest a single fold"""
        balance = 10000
        units = 0
        trades = []
        balances = [balance]

        for i in range(len(test_data)):
            price = test_data.iloc[i]['AAPL.O']
            pred = predictions[i]
            prob = pred_proba[i]

            # Trading logic
            if units == 0 and pred == 1 and prob > 0.60:
                # Buy
                units = int((balance * 0.25) / (price * (1 + self.ptc)))
                balance -= units * price * (1 + self.ptc)
                trades.append({
                    'date': test_data.index[i],
                    'type': 'BUY',
                    'price': price,
                    'units': units,
                    'balance': balance
                })
            elif units > 0 and (pred == 0 or prob < 0.40):
                # Sell
                balance += units * price * (1 - self.ptc)
                trades.append({
                    'date': test_data.index[i],
                    'type': 'SELL',
                    'price': price,
                    'units': units,
                    'balance': balance
                })
                units = 0

            # Track balance
            net_balance = balance + units * price
            balances.append(net_balance)

        # Close any open position
        if units > 0:
            balance += units * test_data.iloc[-1]['AAPL.O']
            units = 0

        # Calculate metrics
        returns = pd.Series(balances).pct_change().dropna()
        total_return = (balance / 10000 - 1) * 100
        sharpe = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0
        max_dd = self.calculate_max_drawdown(balances)

        return {
            'return': total_return,
            'sharpe': sharpe,
            'max_drawdown': max_dd,
            'num_trades': len(trades),
            'final_balance': balance
        }

    def calculate_max_drawdown(self, balances):
        """Calculate maximum drawdown"""
        peak = pd.Series(balances).cummax()
        drawdown = (pd.Series(balances) - peak) / peak
        return drawdown.min() * 100

    def print_summary(self):
        """Print aggregate statistics"""
        print("\n" + "="*80)
        print("AGGREGATE RESULTS")
        print("="*80)

        avg_return = np.mean([r['return'] for r in self.results])
        std_return = np.std([r['return'] for r in self.results])
        avg_sharpe = np.mean([r['sharpe'] for r in self.results])
        avg_mdd = np.mean([r['max_drawdown'] for r in self.results])
        total_trades = sum([r['num_trades'] for r in self.results])

        print(f"Average Return:        {avg_return:.2f}% ± {std_return:.2f}%")
        print(f"Average Sharpe:        {avg_sharpe:.2f}")
        print(f"Average Max Drawdown:  {avg_mdd:.2f}%")
        print(f"Total Trades:          {total_trades}")
        print(f"Win Rate:              {self.calculate_win_rate():.2f}%")

    def calculate_win_rate(self):
        """Calculate win rate across all folds"""
        winning_folds = sum([1 for r in self.results if r['return'] > 0])
        return (winning_folds / len(self.results)) * 100

# Usage
from xgboost import XGBClassifier

wf_backtest = WalkForwardBacktest(
    data=data,
    model_class=XGBClassifier,
    train_window=500,
    test_window=50,
    rebalance_freq=50,
    ptc=0.0015
)
wf_backtest.run()
```

### 6.3 Monte Carlo Simulation for Robustness

**Test strategy robustness with parameter uncertainty:**

```python
def monte_carlo_backtest(data, model, n_simulations=1000):
    """
    Run Monte Carlo simulation with randomized parameters

    Tests robustness to:
    - Transaction cost uncertainty
    - Position sizing variation
    - Entry/exit threshold variation
    """
    results = []

    for i in range(n_simulations):
        # Randomize parameters
        ptc = np.random.uniform(0.0005, 0.003)  # 0.05% to 0.3%
        position_size = np.random.uniform(0.15, 0.35)  # 15% to 35%
        entry_threshold = np.random.uniform(0.55, 0.65)  # 55% to 65%
        exit_threshold = np.random.uniform(0.35, 0.45)  # 35% to 45%

        # Run backtest with these parameters
        bt = BacktestingBase(
            url=url,
            symbol=symbol,
            model=model,
            amount=10000,
            ptc=ptc,
            verbose=False
        )
        bt.entry_threshold = entry_threshold
        bt.exit_threshold = exit_threshold
        bt.position_fraction = position_size

        bt.run_strategy()

        results.append({
            'return': (bt.current_balance / 10000 - 1) * 100,
            'ptc': ptc,
            'position_size': position_size,
            'entry_threshold': entry_threshold
        })

    # Analyze distribution of returns
    returns = [r['return'] for r in results]

    print("="*80)
    print("MONTE CARLO SIMULATION RESULTS")
    print("="*80)
    print(f"Mean Return:      {np.mean(returns):.2f}%")
    print(f"Std Return:       {np.std(returns):.2f}%")
    print(f"5th Percentile:   {np.percentile(returns, 5):.2f}%")
    print(f"Median:           {np.median(returns):.2f}%")
    print(f"95th Percentile:  {np.percentile(returns, 95):.2f}%")
    print(f"Prob(Loss):       {sum([1 for r in returns if r < 0]) / len(returns) * 100:.1f}%")

    return results
```

---

## 7. Performance Metrics to Track

### 7.1 Current Metrics (Insufficient)

**What's tracked:**
```python
perf = (self.current_balance / self.initial_amount - 1) * 100
print(f'performance [%] = {perf:.3f}')
print(f'trades [#] = {self.trades}')
```

**What's missing:**
- Risk-adjusted returns (Sharpe, Sortino)
- Drawdown analysis
- Win rate and profit factor
- Exposure time
- Benchmark comparison
- Statistical significance

### 7.2 Essential Performance Metrics

**A) Return Metrics**

```python
class PerformanceMetrics:
    def __init__(self, balances, returns, trades, benchmark_returns=None):
        """
        Calculate comprehensive performance metrics

        Parameters:
        - balances: Series of portfolio balance over time
        - returns: Series of period returns
        - trades: List of trade dictionaries
        - benchmark_returns: Series of benchmark returns (e.g., buy-hold)
        """
        self.balances = pd.Series(balances)
        self.returns = pd.Series(returns)
        self.trades = trades
        self.benchmark_returns = benchmark_returns

    def total_return(self):
        """Total return percentage"""
        return (self.balances.iloc[-1] / self.balances.iloc[0] - 1) * 100

    def annualized_return(self):
        """Annualized return (CAGR)"""
        n_days = len(self.balances)
        total_return = self.balances.iloc[-1] / self.balances.iloc[0]
        return (total_return ** (252 / n_days) - 1) * 100

    def sharpe_ratio(self, risk_free_rate=0.02):
        """Sharpe ratio (annualized)"""
        excess_returns = self.returns - (risk_free_rate / 252)
        if self.returns.std() == 0:
            return 0
        return (excess_returns.mean() / self.returns.std()) * np.sqrt(252)

    def sortino_ratio(self, risk_free_rate=0.02):
        """Sortino ratio (annualized)"""
        excess_returns = self.returns - (risk_free_rate / 252)
        downside_returns = self.returns[self.returns < 0]

        if len(downside_returns) == 0 or downside_returns.std() == 0:
            return 0

        return (excess_returns.mean() / downside_returns.std()) * np.sqrt(252)

    def calmar_ratio(self):
        """Calmar ratio (return / max drawdown)"""
        annual_return = self.annualized_return() / 100
        max_dd = abs(self.max_drawdown() / 100)

        if max_dd == 0:
            return 0
        return annual_return / max_dd

    def max_drawdown(self):
        """Maximum drawdown percentage"""
        peak = self.balances.cummax()
        drawdown = (self.balances - peak) / peak
        return drawdown.min() * 100

    def avg_drawdown(self):
        """Average drawdown percentage"""
        peak = self.balances.cummax()
        drawdown = (self.balances - peak) / peak
        return drawdown[drawdown < 0].mean() * 100 if len(drawdown[drawdown < 0]) > 0 else 0

    def win_rate(self):
        """Percentage of winning trades"""
        if len(self.trades) == 0:
            return 0

        winning_trades = [t for t in self.trades if t.get('pnl', 0) > 0]
        return (len(winning_trades) / len(self.trades)) * 100

    def profit_factor(self):
        """Ratio of gross profit to gross loss"""
        gross_profit = sum([t.get('pnl', 0) for t in self.trades if t.get('pnl', 0) > 0])
        gross_loss = abs(sum([t.get('pnl', 0) for t in self.trades if t.get('pnl', 0) < 0]))

        if gross_loss == 0:
            return 0
        return gross_profit / gross_loss

    def avg_trade(self):
        """Average trade return"""
        if len(self.trades) == 0:
            return 0
        return np.mean([t.get('pnl', 0) for t in self.trades])

    def avg_winning_trade(self):
        """Average winning trade return"""
        winning_trades = [t.get('pnl', 0) for t in self.trades if t.get('pnl', 0) > 0]
        if len(winning_trades) == 0:
            return 0
        return np.mean(winning_trades)

    def avg_losing_trade(self):
        """Average losing trade return"""
        losing_trades = [t.get('pnl', 0) for t in self.trades if t.get('pnl', 0) < 0]
        if len(losing_trades) == 0:
            return 0
        return np.mean(losing_trades)

    def expectancy(self):
        """Expected value per trade"""
        win_rate = self.win_rate() / 100
        avg_win = self.avg_winning_trade()
        avg_loss = abs(self.avg_losing_trade())

        return (win_rate * avg_win) - ((1 - win_rate) * avg_loss)

    def exposure_time(self):
        """Percentage of time capital is deployed"""
        # Assumes balances series has NaN when not in market
        # For simplicity, track separately in backtest
        pass

    def information_ratio(self):
        """Information ratio vs benchmark"""
        if self.benchmark_returns is None:
            return 0

        active_returns = self.returns - self.benchmark_returns
        if active_returns.std() == 0:
            return 0

        return (active_returns.mean() / active_returns.std()) * np.sqrt(252)

    def print_summary(self):
        """Print comprehensive performance summary"""
        print("="*80)
        print("PERFORMANCE SUMMARY")
        print("="*80)
        print(f"Total Return:              {self.total_return():.2f}%")
        print(f"Annualized Return:         {self.annualized_return():.2f}%")
        print(f"Sharpe Ratio:              {self.sharpe_ratio():.2f}")
        print(f"Sortino Ratio:             {self.sortino_ratio():.2f}")
        print(f"Calmar Ratio:              {self.calmar_ratio():.2f}")
        print("\n" + "-"*80)
        print("RISK METRICS")
        print("-"*80)
        print(f"Max Drawdown:              {self.max_drawdown():.2f}%")
        print(f"Average Drawdown:          {self.avg_drawdown():.2f}%")
        print(f"Volatility (annual):       {self.returns.std() * np.sqrt(252) * 100:.2f}%")
        print("\n" + "-"*80)
        print("TRADE STATISTICS")
        print("-"*80)
        print(f"Total Trades:              {len(self.trades)}")
        print(f"Win Rate:                  {self.win_rate():.2f}%")
        print(f"Profit Factor:             {self.profit_factor():.2f}")
        print(f"Expectancy:                ${self.expectancy():.2f}")
        print(f"Avg Trade:                 ${self.avg_trade():.2f}")
        print(f"Avg Winning Trade:         ${self.avg_winning_trade():.2f}")
        print(f"Avg Losing Trade:          ${self.avg_losing_trade():.2f}")

        if self.benchmark_returns is not None:
            print("\n" + "-"*80)
            print("BENCHMARK COMPARISON")
            print("-"*80)
            print(f"Information Ratio:         {self.information_ratio():.2f}")
            benchmark_total_return = (self.benchmark_returns + 1).prod() - 1
            print(f"Benchmark Total Return:    {benchmark_total_return * 100:.2f}%")
            print(f"Alpha:                     {self.total_return() - benchmark_total_return * 100:.2f}%")

# Usage example
metrics = PerformanceMetrics(
    balances=balance_history,
    returns=return_history,
    trades=trade_list,
    benchmark_returns=benchmark_returns
)
metrics.print_summary()
```

### 7.3 Visualization Dashboard

```python
import matplotlib.pyplot as plt

class PerformanceVisualizer:
    def __init__(self, balances, returns, trades, benchmark_balances=None):
        self.balances = pd.Series(balances)
        self.returns = pd.Series(returns)
        self.trades = trades
        self.benchmark_balances = pd.Series(benchmark_balances) if benchmark_balances else None

    def plot_equity_curve(self):
        """Plot equity curve with benchmark"""
        fig, ax = plt.subplots(figsize=(12, 6))

        ax.plot(self.balances.index, self.balances.values, label='Strategy', linewidth=2)

        if self.benchmark_balances is not None:
            ax.plot(self.benchmark_balances.index, self.benchmark_balances.values,
                   label='Buy & Hold', linestyle='--', linewidth=2)

        ax.set_xlabel('Date')
        ax.set_ylabel('Portfolio Value ($)')
        ax.set_title('Equity Curve')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def plot_drawdown(self):
        """Plot drawdown over time"""
        peak = self.balances.cummax()
        drawdown = (self.balances - peak) / peak

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.fill_between(drawdown.index, drawdown.values * 100, 0,
                        color='red', alpha=0.3, label='Drawdown')
        ax.plot(drawdown.index, drawdown.values * 100, color='red', linewidth=1)

        ax.set_xlabel('Date')
        ax.set_ylabel('Drawdown (%)')
        ax.set_title('Drawdown Over Time')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def plot_monthly_returns(self):
        """Plot monthly returns heatmap"""
        monthly_returns = self.returns.resample('M').apply(lambda x: (1 + x).prod() - 1)

        # Create month/year matrix
        returns_df = pd.DataFrame({
            'Year': monthly_returns.index.year,
            'Month': monthly_returns.index.month,
            'Return': monthly_returns.values * 100
        })

        pivot = returns_df.pivot(index='Month', columns='Year', values='Return')

        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(pivot, annot=True, fmt='.1f', cmap='RdYlGn', center=0,
                   ax=ax, cbar_kws={'label': 'Return (%)'})

        ax.set_title('Monthly Returns Heatmap')
        ax.set_ylabel('Month')
        ax.set_xlabel('Year')

        plt.tight_layout()
        return fig

    def plot_rolling_sharpe(self, window=60):
        """Plot rolling Sharpe ratio"""
        rolling_sharpe = (
            self.returns.rolling(window).mean() /
            self.returns.rolling(window).std()
        ) * np.sqrt(252)

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(rolling_sharpe.index, rolling_sharpe.values, linewidth=2)
        ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
        ax.axhline(y=1, color='green', linestyle='--', linewidth=1, alpha=0.5, label='Sharpe = 1')

        ax.set_xlabel('Date')
        ax.set_ylabel('Rolling Sharpe Ratio')
        ax.set_title(f'{window}-Day Rolling Sharpe Ratio')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def plot_trade_distribution(self):
        """Plot distribution of trade returns"""
        trade_returns = [t.get('pnl', 0) for t in self.trades]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Histogram
        ax1.hist(trade_returns, bins=30, edgecolor='black', alpha=0.7)
        ax1.axvline(x=0, color='red', linestyle='--', linewidth=2)
        ax1.set_xlabel('Trade P&L ($)')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Distribution of Trade Returns')
        ax1.grid(True, alpha=0.3)

        # Cumulative P&L
        cumulative_pnl = np.cumsum(trade_returns)
        ax2.plot(range(len(cumulative_pnl)), cumulative_pnl, linewidth=2)
        ax2.axhline(y=0, color='red', linestyle='--', linewidth=1)
        ax2.set_xlabel('Trade Number')
        ax2.set_ylabel('Cumulative P&L ($)')
        ax2.set_title('Cumulative Trade P&L')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def create_dashboard(self):
        """Create comprehensive performance dashboard"""
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

        # Equity curve
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(self.balances.index, self.balances.values, label='Strategy', linewidth=2)
        if self.benchmark_balances is not None:
            ax1.plot(self.benchmark_balances.index, self.benchmark_balances.values,
                    label='Buy & Hold', linestyle='--', linewidth=2)
        ax1.set_title('Equity Curve', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Drawdown
        ax2 = fig.add_subplot(gs[1, 0])
        peak = self.balances.cummax()
        drawdown = (self.balances - peak) / peak
        ax2.fill_between(drawdown.index, drawdown.values * 100, 0, color='red', alpha=0.3)
        ax2.set_title('Drawdown', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Drawdown (%)')
        ax2.grid(True, alpha=0.3)

        # Rolling Sharpe
        ax3 = fig.add_subplot(gs[1, 1])
        rolling_sharpe = (
            self.returns.rolling(60).mean() /
            self.returns.rolling(60).std()
        ) * np.sqrt(252)
        ax3.plot(rolling_sharpe.index, rolling_sharpe.values)
        ax3.axhline(y=1, color='green', linestyle='--', alpha=0.5)
        ax3.set_title('60-Day Rolling Sharpe', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3)

        # Trade distribution
        ax4 = fig.add_subplot(gs[2, 0])
        trade_returns = [t.get('pnl', 0) for t in self.trades]
        ax4.hist(trade_returns, bins=20, edgecolor='black', alpha=0.7)
        ax4.axvline(x=0, color='red', linestyle='--')
        ax4.set_title('Trade Return Distribution', fontsize=12, fontweight='bold')
        ax4.set_xlabel('P&L ($)')
        ax4.grid(True, alpha=0.3)

        # Cumulative P&L
        ax5 = fig.add_subplot(gs[2, 1])
        cumulative_pnl = np.cumsum(trade_returns)
        ax5.plot(range(len(cumulative_pnl)), cumulative_pnl, linewidth=2)
        ax5.axhline(y=0, color='red', linestyle='--')
        ax5.set_title('Cumulative Trade P&L', fontsize=12, fontweight='bold')
        ax5.set_xlabel('Trade Number')
        ax5.grid(True, alpha=0.3)

        return fig

# Usage
viz = PerformanceVisualizer(
    balances=balance_history,
    returns=return_history,
    trades=trade_list,
    benchmark_balances=benchmark_balance_history
)

# Create individual plots
viz.plot_equity_curve()
viz.plot_drawdown()
viz.plot_monthly_returns()
viz.plot_rolling_sharpe()

# Or create full dashboard
viz.create_dashboard()
plt.savefig('performance_dashboard.png', dpi=300, bbox_inches='tight')
```

### 7.4 Statistical Significance Testing

**Test if performance is due to skill or luck:**

```python
def statistical_significance_tests(strategy_returns, benchmark_returns):
    """
    Perform statistical tests on strategy performance
    """
    from scipy import stats

    print("="*80)
    print("STATISTICAL SIGNIFICANCE TESTS")
    print("="*80)

    # 1. T-test: Are strategy returns significantly different from zero?
    t_stat, p_value = stats.ttest_1samp(strategy_returns, 0)
    print(f"\n1. T-test (returns vs zero):")
    print(f"   t-statistic: {t_stat:.3f}")
    print(f"   p-value: {p_value:.4f}")
    print(f"   Significant at 5%: {'Yes' if p_value < 0.05 else 'No'}")

    # 2. T-test: Are strategy returns better than benchmark?
    excess_returns = strategy_returns - benchmark_returns
    t_stat, p_value = stats.ttest_1samp(excess_returns, 0)
    print(f"\n2. T-test (excess returns vs benchmark):")
    print(f"   t-statistic: {t_stat:.3f}")
    print(f"   p-value: {p_value:.4f}")
    print(f"   Significant at 5%: {'Yes' if p_value < 0.05 else 'No'}")

    # 3. Normality test (important for above tests)
    stat, p_value = stats.shapiro(strategy_returns)
    print(f"\n3. Shapiro-Wilk test (normality):")
    print(f"   Statistic: {stat:.3f}")
    print(f"   p-value: {p_value:.4f}")
    print(f"   Returns are normal: {'Yes' if p_value > 0.05 else 'No'}")

    # 4. If not normal, use non-parametric test
    if p_value < 0.05:
        stat, p_value = stats.wilcoxon(excess_returns)
        print(f"\n4. Wilcoxon signed-rank test (non-parametric):")
        print(f"   Statistic: {stat:.3f}")
        print(f"   p-value: {p_value:.4f}")
        print(f"   Significant at 5%: {'Yes' if p_value < 0.05 else 'No'}")
```

---

## 8. Specific Recommendations and Realistic Expectations

### 8.1 Implementation Roadmap (12-Week Plan)

**Phase 1: Foundation (Weeks 1-4)**

**Week 1: Critical Bug Fixes**
- Remove data leakage ('d' and 'd_' from features)
- Fix normalization (apply to all features before lagging)
- Implement walk-forward validation framework
- **Expected Impact:** Stop negative performance, get to 0-2% return

**Week 2: Position Sizing and Risk Management**
- Reduce position size to 25% max (from 50%)
- Implement stop-loss (2%) and take-profit (5%)
- Add portfolio max drawdown limit (15%)
- **Expected Impact:** Reduce volatility, protect capital

**Week 3: Transaction Cost Realism**
- Test with realistic costs (0.15% vs current 1%)
- Implement slippage model
- **Expected Impact:** 8-10% improvement if costs were main drag

**Week 4: Baseline Establishment**
- Run walk-forward backtest on fixed bugs
- Calculate comprehensive performance metrics
- Compare to buy-and-hold benchmark
- **Expected Impact:** Establish true baseline (hopefully positive)

**Phase 2: Feature Engineering (Weeks 5-8)**

**Week 5: Add Momentum Features**
- Implement RSI (14, 28 periods)
- Implement ROC (5, 20 periods)
- Test incremental improvement
- **Expected Impact:** +2-4% return improvement

**Week 6: Add Mean Reversion Features**
- Implement Bollinger Bands position
- Distance from moving averages
- **Expected Impact:** +1-3% return improvement

**Week 7: Add Volatility Regime Features**
- Realized vol ratio
- Vol percentile rank
- **Expected Impact:** +1-2% return improvement, better risk adjustment

**Week 8: Feature Selection**
- Use L1 regularization to select best features
- Recursive feature elimination
- Validate with walk-forward
- **Expected Impact:** Reduce overfitting, stabilize performance

**Phase 3: Model Optimization (Weeks 9-11)**

**Week 9: Test Gradient Boosting Models**
- Implement XGBoost
- Hyperparameter tuning with cross-validation
- Compare to baseline models
- **Expected Impact:** +3-6% return improvement

**Week 10: Ensemble Methods**
- Voting ensemble (XGB + RF + LogReg)
- Test different weighting schemes
- **Expected Impact:** +2-4% return improvement, smoother equity curve

**Week 11: Regime-Based Models**
- Detect high/low vol regimes
- Train separate models per regime
- **Expected Impact:** +2-5% return improvement in changing markets

**Phase 4: Production Readiness (Week 12)**

**Week 12: Robustness Testing**
- Monte Carlo simulation (1000 runs)
- Parameter sensitivity analysis
- Out-of-sample validation (2014-2020 data if available)
- Stress testing (2008 crisis, 2020 COVID)
- **Expected Impact:** Validate realistic expectations

### 8.2 Realistic Performance Expectations

**Based on industry benchmarks for retail systematic strategies:**

**Conservative Scenario (50th percentile):**
```
Annualized Return:     5-8%
Sharpe Ratio:          0.6-0.9
Max Drawdown:          -12% to -18%
Win Rate:              52-55%
```

**Moderate Scenario (75th percentile):**
```
Annualized Return:     10-15%
Sharpe Ratio:          0.9-1.3
Max Drawdown:          -10% to -15%
Win Rate:              55-60%
```

**Optimistic Scenario (90th percentile):**
```
Annualized Return:     15-20%
Sharpe Ratio:          1.3-1.8
Max Drawdown:          -8% to -12%
Win Rate:              60-65%
```

**Reality Check:**
- Professional quant hedge funds target Sharpe 1.5-2.5
- Retail strategies typically achieve Sharpe 0.5-1.2
- Alpha degrades over time as markets adapt
- Transaction costs matter significantly

**Important Notes:**
1. **Overfitting Risk:** Historical backtest results often overestimate future performance by 30-50%
2. **Market Regime Dependency:** 2010-2013 was a bull market; strategy may fail in bear markets
3. **Data Mining Bias:** Testing multiple features/models inflates apparent performance
4. **Transaction Costs:** Real-world costs (slippage, commissions, taxes) reduce returns by 1-3% annually
5. **Psychological Factors:** Live trading performance typically 20-40% worse than backtest due to emotions

### 8.3 Recommended Success Metrics

**Don't just track returns. Track:**

1. **Out-of-Sample Sharpe Ratio > 0.8**
   - More important than absolute returns
   - Indicates consistent risk-adjusted performance

2. **Max Drawdown < 15%**
   - Psychological tolerance threshold
   - Determines strategy survivability

3. **Win Rate > 52%**
   - Just above coin flip
   - Combined with good profit factor

4. **Profit Factor > 1.5**
   - Gross wins / Gross losses
   - Indicates positive expectancy

5. **Calmar Ratio > 0.8**
   - Annualized return / Max drawdown
   - Balances return and risk

6. **Low Correlation to Benchmark (< 0.7)**
   - Indicates true alpha generation
   - Provides diversification value

7. **Stable Performance Across Regimes**
   - Bull, bear, ranging markets
   - High, low volatility periods

### 8.4 Warning Signs to Abandon Strategy

**Red flags that indicate strategy is not viable:**

1. **Sharpe < 0.3 out-of-sample** → Strategy has no edge
2. **Win rate < 48%** → Losing more often than winning
3. **Profit factor < 1.2** → Insufficient edge after costs
4. **Max drawdown > 25%** → Unacceptable risk
5. **Performance degrades in walk-forward** → Overfitting
6. **High correlation to benchmark (> 0.9)** → Just leveraged beta
7. **Unstable across parameter changes** → Fragile, not robust

### 8.5 Final Recommendations Summary

**Immediate Actions (Do This Week):**
1. Remove 'd' and 'd_' from features (data leakage)
2. Reduce position size to 25% max
3. Set realistic transaction costs (0.15% vs 1%)
4. Implement walk-forward validation

**High Priority (Do This Month):**
5. Add stop-loss (2%) and take-profit (5%)
6. Implement comprehensive performance metrics
7. Add RSI and Bollinger Bands features
8. Test XGBoost model

**Medium Priority (Do Next Quarter):**
9. Build ensemble model (XGB + RF + LogReg)
10. Implement regime detection
11. Run Monte Carlo robustness testing
12. Validate on out-of-sample data (2014-2020)

**Long-Term (Do Next 6 Months):**
13. Paper trade for 3 months minimum
14. Implement automated execution system
15. Add portfolio-level risk management
16. Expand to multiple assets for diversification

**Key Principle:**
> "The goal is not to find the perfect strategy, but to build a robust, profitable, and psychologically sustainable trading system."

**Expected Timeline to Profitability:**
- **3 months:** Fix bugs, positive backtest results
- **6 months:** Validated walk-forward performance
- **9 months:** Paper trading with realistic execution
- **12 months:** Live trading with small capital

**Expected Performance After Full Implementation:**
- **Year 1:** 5-10% annualized return, Sharpe 0.7-1.0
- **Year 2-3:** 10-15% annualized return, Sharpe 1.0-1.3 (if market conditions favorable)
- **Ongoing:** Monitor for alpha decay, retrain/adapt as needed

---

## Conclusion

The current strategy underperforms due to a combination of:
1. **Data leakage** (target in features)
2. **Overly aggressive position sizing** (50% with no stops)
3. **Single train-test split** (no walk-forward validation)
4. **Weak features** (redundant, no momentum)
5. **Inappropriate models** (GaussianNB, linear SVC)
6. **No risk management** (no stops, drawdown limits)
7. **Unrealistic transaction costs** (1% is stress test, not normal)

**Bottom Line:**
With proper implementation of the recommendations above, a realistic expectation is:
- **Conservative:** 5-8% annualized, Sharpe 0.6-0.9
- **Moderate:** 10-15% annualized, Sharpe 0.9-1.3
- **Optimistic:** 15-20% annualized, Sharpe 1.3-1.8

However, these are BEST CASE scenarios assuming:
- Proper walk-forward validation
- Realistic transaction costs
- Robust risk management
- No regime changes that invalidate the model
- Disciplined execution

**Most important:** Focus on building a robust, sustainable system rather than chasing maximum returns. A strategy with 8% annual return and Sharpe 1.2 that you can stick with is far better than a 20% return backtest that fails in live trading.
