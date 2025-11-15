"""Simple backtest example demonstrating the trading system.

This is a minimal example showing how to:
1. Load data
2. Create features
3. Train a model
4. Run a backtest with risk management
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from algo_trading.data.loader import DataLoader
from algo_trading.features.engineering import FeatureEngineer
from algo_trading.models.trainer import ModelTrainer
from algo_trading.risk.position_sizing import create_position_sizer
from algo_trading.risk.manager import RiskManager
from algo_trading.backtesting.engine import BacktestEngine
from algo_trading.strategies.ml_strategy import MLPredictionStrategy
from algo_trading.utils.logger import setup_logger


def main():
    """Run simple backtest example."""
    # Setup logging
    logger = setup_logger("example", level="INFO", console=True)

    logger.info("Simple Backtest Example")
    logger.info("=" * 60)

    # 1. Load Data
    logger.info("\n1. Loading data...")
    loader = DataLoader()
    data = loader.load_from_url(
        url="http://hilpisch.com/ref_eikon_eod_data.csv",
        symbol="AAPL.O",
        max_rows=1000,
    )
    logger.info(f"   Loaded {len(data)} rows")

    # 2. Split data (70/30)
    logger.info("\n2. Splitting data...")
    split_idx = int(len(data) * 0.7)
    train_data = data.iloc[:split_idx]
    test_data = data.iloc[split_idx:]
    logger.info(f"   Train: {len(train_data)} rows, Test: {len(test_data)} rows")

    # 3. Create features
    logger.info("\n3. Engineering features...")
    engineer = FeatureEngineer(
        sma_windows=(20, 60),
        ewma_halflife=(20, 60),
        n_lags=5,
    )

    # Fit on training data
    train_features, feature_cols = engineer.fit_transform(train_data, price_col="AAPL.O")

    # Transform test data
    test_features = engineer.transform(test_data)
    logger.info(f"   Created {len(feature_cols)} features")

    # 4. Train model
    logger.info("\n4. Training model...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        class_weight="balanced",
        random_state=42,
    )

    trainer = ModelTrainer(model, model_name="RandomForest")
    trainer.train(train_features[feature_cols], train_features["direction"])

    # Evaluate
    test_metrics = trainer.evaluate(
        test_features[feature_cols],
        test_features["direction"],
        set_name="test",
    )
    logger.info(f"   Test Accuracy: {test_metrics['accuracy']:.2%}")
    logger.info(f"   Test F1 Score: {test_metrics['f1']:.2%}")

    # 5. Generate signals
    logger.info("\n5. Generating trading signals...")
    strategy = MLPredictionStrategy(
        model=trainer.model,
        confidence_threshold=0.55,
        feature_columns=feature_cols,
    )

    signals, confidences = strategy.generate_signals(
        test_features, return_confidence=True
    )

    signal_stats = strategy.get_signal_statistics(signals)
    logger.info(f"   Buy signals:  {signal_stats['buy_signals']}")
    logger.info(f"   Sell signals: {signal_stats['sell_signals']}")
    logger.info(f"   Hold signals: {signal_stats['hold_signals']}")

    # 6. Run backtest
    logger.info("\n6. Running backtest...")
    initial_capital = 10000

    # Create components
    risk_manager = RiskManager(
        initial_capital=initial_capital,
        max_drawdown_pct=0.15,
        daily_loss_limit_pct=0.03,
        hard_stop_pct=0.03,
        trailing_stop_pct=0.02,
        take_profit_pct=0.05,
    )

    position_sizer = create_position_sizer(
        method="confidence_based",
        base_risk_fraction=0.01,
        min_confidence=0.55,
    )

    engine = BacktestEngine(
        initial_capital=initial_capital,
        risk_manager=risk_manager,
        position_sizer=position_sizer,
        commission_pct=0.0015,
        slippage_bps=5,
    )

    # Prepare data
    backtest_data = test_data.copy()
    backtest_data.columns = ["Close"]

    # Run backtest
    results = engine.run_backtest(
        data=backtest_data,
        signals=signals,
        confidences=confidences,
    )

    # 7. Display results
    logger.info("\n7. Results:")
    logger.info("=" * 60)
    logger.info(f"   Initial Capital: ${results['initial_capital']:,.2f}")
    logger.info(f"   Final Capital:   ${results['final_capital']:,.2f}")
    logger.info(f"   Total Return:    {results['total_return']:+.2%}")
    logger.info(f"   Max Drawdown:    {results['max_drawdown']:.2%}")
    logger.info(f"   Sharpe Ratio:    {results['sharpe_ratio']:.2f}")
    logger.info(f"   Total Trades:    {results['total_trades']}")

    if results['total_trades'] > 0:
        logger.info(f"   Win Rate:        {results['win_rate']:.2%}")
        logger.info(f"   Profit Factor:   {results['profit_factor']:.2f}")

    logger.info("\n" + "=" * 60)
    logger.info("Example completed successfully!")

    return 0


if __name__ == "__main__":
    sys.exit(main())
