"""Example usage of the algorithmic trading system.

This script demonstrates how to use all the components together:
1. Load and prepare data
2. Train a model
3. Create a trading strategy
4. Set up risk management
5. Run a backtest
6. Analyze results
"""

import logging
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from algo_trading.utils.config import Config
from algo_trading.utils.logger import setup_logger
from algo_trading.data.loader import DataLoader
from algo_trading.features.engineering import FeatureEngineer
from algo_trading.models.trainer import ModelTrainer
from algo_trading.strategies import MLPredictionStrategy
from algo_trading.risk import (
    RiskManager,
    create_position_sizer,
)
from algo_trading.backtesting import BacktestEngine

# Setup logging
logger = setup_logger("example", level="INFO", console=True)


def main():
    """Run complete trading system example."""

    logger.info("=" * 70)
    logger.info("Algorithmic Trading System - Complete Example")
    logger.info("=" * 70)

    # ========================================================================
    # 1. LOAD CONFIGURATION
    # ========================================================================
    logger.info("\n1. Loading configuration...")
    config = Config()
    logger.info(f"Loaded config from: config.yaml")

    # ========================================================================
    # 2. LOAD DATA
    # ========================================================================
    logger.info("\n2. Loading market data...")
    loader = DataLoader()

    data = loader.load_from_url(
        url=config.get("data.source_url"),
        symbol=config.get("data.symbol"),
        max_rows=config.get("data.max_rows"),
    )
    logger.info(f"Loaded {len(data)} rows of price data")

    # ========================================================================
    # 3. ENGINEER FEATURES
    # ========================================================================
    logger.info("\n3. Engineering features...")
    engineer = FeatureEngineer(config)

    # Add technical indicators
    data = engineer.add_moving_averages(data)
    data = engineer.add_volatility(data)
    data = engineer.add_momentum(data)
    data = engineer.add_lagged_features(data)

    # Add target variable (next day return direction)
    data["target"] = np.where(data[config.get("data.symbol")].pct_change().shift(-1) > 0, 1, 0)

    # Remove rows with NaN values
    data = data.dropna()
    logger.info(f"Generated features, {len(data)} rows after cleanup")

    # ========================================================================
    # 4. PREPARE DATA FOR MODELING
    # ========================================================================
    logger.info("\n4. Preparing data for modeling...")

    # Separate features and target
    exclude_cols = [config.get("data.symbol"), "target"] + config.get("features.exclude_features", [])
    feature_cols = [col for col in data.columns if col not in exclude_cols]

    X = data[feature_cols]
    y = data["target"]

    logger.info(f"Features: {len(feature_cols)} columns")
    logger.info(f"Target distribution: {y.value_counts().to_dict()}")

    # Train-test split (time-series aware)
    split_idx = int(len(data) * (1 - config.get("models.test_size")))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    logger.info(f"Train set: {len(X_train)} samples")
    logger.info(f"Test set: {len(X_test)} samples")

    # ========================================================================
    # 5. TRAIN MODEL
    # ========================================================================
    logger.info("\n5. Training ML model...")

    model_config = config.get("models.available_models.rf")
    model = RandomForestClassifier(**model_config["params"])

    trainer = ModelTrainer(model, model_name="RandomForest")
    trainer.train(X_train, y_train)

    # Evaluate model
    train_metrics = trainer.evaluate(X_train, y_train, set_name="train")
    test_metrics = trainer.evaluate(X_test, y_test, set_name="test")

    logger.info(f"Train accuracy: {train_metrics['accuracy']:.2%}")
    logger.info(f"Test accuracy: {test_metrics['accuracy']:.2%}")

    # ========================================================================
    # 6. CREATE TRADING STRATEGY
    # ========================================================================
    logger.info("\n6. Creating trading strategy...")

    strategy = MLPredictionStrategy(
        model=trainer.model,
        confidence_threshold=config.get("strategy.confidence_threshold"),
        feature_columns=feature_cols,
    )

    logger.info(f"Strategy: {strategy}")

    # Generate signals on test data
    test_data = data.iloc[split_idx:].copy()
    signals, confidences = strategy.generate_signals_simple(
        X_test,
        return_confidence=True,
    )

    # Get signal statistics
    signal_stats = strategy.get_signal_statistics(signals)
    logger.info(f"Buy signals: {signal_stats['buy_signals']} ({signal_stats['buy_pct']:.1%})")
    logger.info(f"Sell signals: {signal_stats['sell_signals']} ({signal_stats['sell_pct']:.1%})")

    # ========================================================================
    # 7. SETUP RISK MANAGEMENT
    # ========================================================================
    logger.info("\n7. Setting up risk management...")

    # Create position sizer
    position_sizer = create_position_sizer(
        method=config.get("risk_management.position_sizing.method"),
        risk_fraction=config.get("risk_management.position_sizing.risk_per_trade"),
        max_position_fraction=config.get("risk_management.position_sizing.max_position_size"),
    )
    logger.info(f"Position sizer: {position_sizer.__class__.__name__}")

    # Create risk manager
    risk_manager = RiskManager(
        initial_capital=config.get("backtesting.initial_capital"),
        max_drawdown_pct=config.get("risk_management.portfolio.max_drawdown_pct"),
        daily_loss_limit_pct=config.get("risk_management.portfolio.daily_loss_limit_pct"),
        max_portfolio_heat=config.get("risk_management.portfolio.max_portfolio_heat"),
        hard_stop_pct=config.get("risk_management.stop_loss.hard_stop_pct"),
        trailing_stop_pct=config.get("risk_management.stop_loss.trailing_stop_pct"),
        take_profit_pct=config.get("risk_management.stop_loss.take_profit_pct"),
        time_stop_days=config.get("risk_management.stop_loss.time_stop_days"),
    )

    # ========================================================================
    # 8. RUN BACKTEST
    # ========================================================================
    logger.info("\n8. Running backtest...")

    engine = BacktestEngine(
        initial_capital=config.get("backtesting.initial_capital"),
        risk_manager=risk_manager,
        position_sizer=position_sizer,
        commission_pct=config.get("backtesting.transaction_cost"),
        slippage_bps=config.get("backtesting.slippage_bps"),
        verbose=config.get("backtesting.verbose"),
    )

    # Prepare price data for backtest
    price_data = pd.DataFrame({
        "Close": test_data[config.get("data.symbol")]
    })

    # Run backtest
    results = engine.run_backtest(
        data=price_data,
        signals=signals,
        confidences=confidences,
    )

    # ========================================================================
    # 9. ANALYZE RESULTS
    # ========================================================================
    logger.info("\n9. Backtest Results")
    logger.info("=" * 70)

    # Performance metrics
    logger.info("\nPerformance Metrics:")
    logger.info(f"  Initial Capital:    ${results['initial_capital']:>12,.2f}")
    logger.info(f"  Final Capital:      ${results['final_capital']:>12,.2f}")
    logger.info(f"  Total Return:       {results['total_return']:>12.2%}")
    logger.info(f"  CAGR:              {results.get('cagr', 0):>12.2%}")
    logger.info(f"  Sharpe Ratio:      {results.get('sharpe_ratio', 0):>12.2f}")
    logger.info(f"  Sortino Ratio:     {results.get('sortino_ratio', 0):>12.2f}")
    logger.info(f"  Calmar Ratio:      {results.get('calmar_ratio', 0):>12.2f}")

    # Risk metrics
    logger.info("\nRisk Metrics:")
    logger.info(f"  Max Drawdown:      {results.get('max_drawdown', 0):>12.2%}")
    logger.info(f"  Max DD Duration:   {results.get('max_drawdown_duration', 0):>12} periods")
    logger.info(f"  VaR (95%):         ${results.get('var_95', 0):>12,.2f}")
    logger.info(f"  CVaR (95%):        ${results.get('cvar_95', 0):>12,.2f}")

    # Trading metrics
    logger.info("\nTrading Metrics:")
    logger.info(f"  Total Trades:      {results.get('total_trades', 0):>12}")
    logger.info(f"  Win Rate:          {results.get('win_rate', 0):>12.2%}")
    logger.info(f"  Profit Factor:     {results.get('profit_factor', 0):>12.2f}")
    logger.info(f"  R-Expectancy:      {results.get('r_expectancy', 0):>12.2f}")

    if results.get('total_trades', 0) > 0:
        logger.info(f"  Avg Win:           ${results.get('avg_win', 0):>12,.2f} ({results.get('avg_win_pct', 0):+.2%})")
        logger.info(f"  Avg Loss:          ${results.get('avg_loss', 0):>12,.2f} ({results.get('avg_loss_pct', 0):+.2%})")

    # Get equity curve
    equity_curve = engine.get_equity_curve()
    logger.info(f"\nEquity Curve: {len(equity_curve)} data points")

    # Get trade details
    trades = engine.get_trades()
    if not trades.empty:
        logger.info(f"\nTrade History: {len(trades)} trades")
        logger.info("\nFirst 5 trades:")
        logger.info(trades.head().to_string())

    # ========================================================================
    # 10. RISK MANAGEMENT SUMMARY
    # ========================================================================
    logger.info("\n10. Risk Management Summary")
    logger.info("=" * 70)
    logger.info(risk_manager.get_summary())

    logger.info("\n" + "=" * 70)
    logger.info("Example completed successfully!")
    logger.info("=" * 70)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\nInterrupted by user")
    except Exception as e:
        logger.error(f"\nError: {e}")
        import traceback
        traceback.print_exc()
