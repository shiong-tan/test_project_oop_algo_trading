"""Main execution script for the algorithmic trading system.

This script demonstrates the complete workflow:
1. Load and prepare data
2. Engineer features (preventing data leakage)
3. Train ML model
4. Backtest with risk management
5. Display results

Usage:
    python main.py
"""

import sys
import argparse
import logging
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime

# Import our modules
from algo_trading.data.loader import DataLoader
from algo_trading.features.engineering import FeatureEngineer
from algo_trading.models.trainer import ModelTrainer
from algo_trading.risk.position_sizing import create_position_sizer
from algo_trading.risk.manager import RiskManager
from algo_trading.backtesting.engine import BacktestEngine
from algo_trading.strategies.ml_strategy import MLPredictionStrategy
from algo_trading.utils.config import Config
from algo_trading.utils.logger import setup_logger


def main():
    """Main execution function."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Run algorithmic trading system backtest"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="rf",
        choices=["logreg", "rf", "xgb", "lgbm"],
        help="Model to use for predictions",
    )
    args = parser.parse_args()

    # Load configuration
    config = Config(args.config)

    # Setup logging
    log_level = "DEBUG" if args.verbose else config.get("logging.level", "INFO")
    logger = setup_logger(
        "trading",
        log_file=config.get("logging.file"),
        level=log_level,
        console=config.get("logging.console", True),
    )

    logger.info("=" * 80)
    logger.info("ALGORITHMIC TRADING SYSTEM - PRODUCTION BACKTEST")
    logger.info("=" * 80)

    try:
        # Step 1: Load Data
        logger.info("\n" + "=" * 80)
        logger.info("STEP 1: LOADING DATA")
        logger.info("=" * 80)

        loader = DataLoader(
            allowed_domains=["hilpisch.com"],
            timeout=30,
            max_retries=3,
        )

        data = loader.load_from_url(
            url=config.get("data.source_url"),
            symbol=config.get("data.symbol"),
            max_rows=config.get("data.max_rows"),
        )

        logger.info(f"Loaded {len(data)} rows of data")
        logger.info(f"Date range: {data.index[0]} to {data.index[-1]}")

        # Step 2: Train/Test Split
        logger.info("\n" + "=" * 80)
        logger.info("STEP 2: TRAIN/TEST SPLIT")
        logger.info("=" * 80)

        test_size = config.get("models.test_size", 0.3)
        split_idx = int(len(data) * (1 - test_size))

        train_data = data.iloc[:split_idx]
        test_data = data.iloc[split_idx:]

        logger.info(f"Training set: {len(train_data)} rows ({1-test_size:.0%})")
        logger.info(f"Test set:     {len(test_data)} rows ({test_size:.0%})")
        logger.info(f"Split date:   {data.index[split_idx]}")

        # Step 3: Feature Engineering
        logger.info("\n" + "=" * 80)
        logger.info("STEP 3: FEATURE ENGINEERING (LEAKAGE-FREE)")
        logger.info("=" * 80)

        feature_engineer = FeatureEngineer(
            sma_windows=(
                config.get("features.sma_short_window"),
                config.get("features.sma_long_window"),
            ),
            ewma_halflife=(
                config.get("features.ewma_short_halflife"),
                config.get("features.ewma_long_halflife"),
            ),
            volatility_windows=(
                config.get("features.volatility_short_window"),
                config.get("features.volatility_long_window"),
            ),
            rsi_period=config.get("features.rsi_period"),
            bollinger_period=config.get("features.bollinger_period"),
            bollinger_std=config.get("features.bollinger_std"),
            n_lags=config.get("features.n_lags"),
            exclude_features=config.get("features.exclude_features"),
        )

        # Fit on training data ONLY
        train_features, feature_cols = feature_engineer.fit_transform(
            train_data, price_col=config.get("data.symbol")
        )

        # Transform test data (no fitting!)
        test_features = feature_engineer.transform(test_data)

        logger.info(f"Created {len(feature_cols)} features")
        logger.info(f"Training features: {len(train_features)} rows")
        logger.info(f"Test features:     {len(test_features)} rows")

        # Step 4: Train Model
        logger.info("\n" + "=" * 80)
        logger.info("STEP 4: MODEL TRAINING")
        logger.info("=" * 80)

        # Get model configuration
        model_config = config.get(f"models.available_models.{args.model}")

        if model_config["type"] == "RandomForestClassifier":
            model = RandomForestClassifier(**model_config["params"])
        elif model_config["type"] == "LogisticRegression":
            from sklearn.linear_model import LogisticRegression
            model = LogisticRegression(**model_config["params"])
        elif model_config["type"] == "XGBClassifier":
            from xgboost import XGBClassifier
            model = XGBClassifier(**model_config["params"])
        elif model_config["type"] == "LGBMClassifier":
            from lightgbm import LGBMClassifier
            model = LGBMClassifier(**model_config["params"])
        else:
            raise ValueError(f"Unknown model type: {model_config['type']}")

        trainer = ModelTrainer(model, model_name=args.model.upper())

        # Train
        trainer.train(
            train_features[feature_cols],
            train_features["direction"],
        )

        # Evaluate on training set
        train_metrics = trainer.evaluate(
            train_features[feature_cols],
            train_features["direction"],
            set_name="train",
        )

        # Evaluate on test set
        test_metrics = trainer.evaluate(
            test_features[feature_cols],
            test_features["direction"],
            set_name="test",
        )

        logger.info(f"\nTRAINING SET PERFORMANCE:")
        logger.info(f"  Accuracy:  {train_metrics['accuracy']:.4f}")
        logger.info(f"  Precision: {train_metrics['precision']:.4f}")
        logger.info(f"  Recall:    {train_metrics['recall']:.4f}")
        logger.info(f"  F1 Score:  {train_metrics['f1']:.4f}")
        if train_metrics.get("roc_auc"):
            logger.info(f"  ROC AUC:   {train_metrics['roc_auc']:.4f}")

        logger.info(f"\nTEST SET PERFORMANCE:")
        logger.info(f"  Accuracy:  {test_metrics['accuracy']:.4f}")
        logger.info(f"  Precision: {test_metrics['precision']:.4f}")
        logger.info(f"  Recall:    {test_metrics['recall']:.4f}")
        logger.info(f"  F1 Score:  {test_metrics['f1']:.4f}")
        if test_metrics.get("roc_auc"):
            logger.info(f"  ROC AUC:   {test_metrics['roc_auc']:.4f}")

        # Feature importance
        if trainer.feature_importance is not None:
            logger.info(f"\nTOP 10 FEATURES:")
            for idx, row in trainer.feature_importance.head(10).iterrows():
                logger.info(f"  {row['feature']:20s}: {row['importance']:.4f}")

        # Step 5: Generate Trading Signals
        logger.info("\n" + "=" * 80)
        logger.info("STEP 5: GENERATING TRADING SIGNALS")
        logger.info("=" * 80)

        strategy = MLPredictionStrategy(
            model=trainer.model,
            confidence_threshold=config.get("strategy.confidence_threshold"),
            feature_columns=feature_cols,
        )

        signals, confidences = strategy.generate_signals(
            test_features, return_confidence=True
        )

        signal_stats = strategy.get_signal_statistics(signals)
        logger.info(f"Signal Statistics:")
        logger.info(f"  Total Periods:  {signal_stats['total_periods']}")
        logger.info(f"  Buy Signals:    {signal_stats['buy_signals']} ({signal_stats['buy_pct']:.1%})")
        logger.info(f"  Sell Signals:   {signal_stats['sell_signals']} ({signal_stats['sell_pct']:.1%})")
        logger.info(f"  Hold Signals:   {signal_stats['hold_signals']} ({signal_stats['hold_pct']:.1%})")
        logger.info(f"  Trade Frequency: {signal_stats['trade_frequency']:.1%}")

        # Step 6: Backtesting
        logger.info("\n" + "=" * 80)
        logger.info("STEP 6: BACKTESTING WITH RISK MANAGEMENT")
        logger.info("=" * 80)

        initial_capital = config.get("backtesting.initial_capital")

        # Create risk manager
        risk_manager = RiskManager(
            initial_capital=initial_capital,
            max_drawdown_pct=config.get("risk_management.portfolio.max_drawdown_pct"),
            daily_loss_limit_pct=config.get("risk_management.portfolio.daily_loss_limit_pct"),
            max_portfolio_heat=config.get("risk_management.portfolio.max_portfolio_heat"),
            hard_stop_pct=config.get("risk_management.stop_loss.hard_stop_pct"),
            trailing_stop_pct=config.get("risk_management.stop_loss.trailing_stop_pct"),
            take_profit_pct=config.get("risk_management.stop_loss.take_profit_pct"),
            time_stop_days=config.get("risk_management.stop_loss.time_stop_days"),
            drawdown_zones=config.get("risk_management.drawdown_zones"),
        )

        # Create position sizer
        position_sizer = create_position_sizer(
            method=config.get("risk_management.position_sizing.method"),
            risk_fraction=config.get("risk_management.position_sizing.risk_per_trade"),
            max_position_fraction=config.get("risk_management.position_sizing.max_position_size"),
        )

        # Create backtest engine
        engine = BacktestEngine(
            initial_capital=initial_capital,
            risk_manager=risk_manager,
            position_sizer=position_sizer,
            commission_pct=config.get("backtesting.transaction_cost"),
            slippage_bps=config.get("backtesting.slippage_bps"),
            verbose=args.verbose,
        )

        # Prepare data for backtesting
        backtest_data = test_data.copy()
        backtest_data.columns = ["Close"]  # Rename column for backtest engine

        # Run backtest
        results = engine.run_backtest(
            data=backtest_data,
            signals=signals,
            confidences=confidences,
        )

        # Step 7: Display Results
        logger.info("\n" + "=" * 80)
        logger.info("STEP 7: BACKTEST RESULTS")
        logger.info("=" * 80)

        logger.info(f"\nCAPITAL:")
        logger.info(f"  Initial:  ${results['initial_capital']:,.2f}")
        logger.info(f"  Final:    ${results['final_capital']:,.2f}")
        logger.info(f"  Return:   {results['total_return']:+.2%}")

        if "cagr" in results:
            logger.info(f"  CAGR:     {results['cagr']:+.2%}")

        logger.info(f"\nRISK METRICS:")
        logger.info(f"  Max Drawdown:     {results['max_drawdown']:.2%}")
        logger.info(f"  Max DD Duration:  {results['max_drawdown_duration']} periods")
        logger.info(f"  VaR (95%):        ${results['var_95']:,.2f}")
        logger.info(f"  CVaR (95%):       ${results['cvar_95']:,.2f}")

        logger.info(f"\nRISK-ADJUSTED RETURNS:")
        logger.info(f"  Sharpe Ratio:   {results['sharpe_ratio']:.3f}")
        logger.info(f"  Sortino Ratio:  {results['sortino_ratio']:.3f}")
        logger.info(f"  Calmar Ratio:   {results['calmar_ratio']:.3f}")

        logger.info(f"\nTRADING STATISTICS:")
        logger.info(f"  Total Trades:   {results['total_trades']}")
        if results['total_trades'] > 0:
            logger.info(f"  Win Rate:       {results['win_rate']:.2%}")
            logger.info(f"  Avg Win:        ${results['avg_win']:,.2f} ({results['avg_win_pct']:+.2%})")
            logger.info(f"  Avg Loss:       ${results['avg_loss']:,.2f} ({results['avg_loss_pct']:+.2%})")
            logger.info(f"  Profit Factor:  {results['profit_factor']:.2f}")
            logger.info(f"  R-Expectancy:   {results['r_expectancy']:.2f}")

        # Display risk management summary
        logger.info("\n" + risk_manager.get_summary())

        # Save results to file
        output_dir = Path("results")
        output_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = output_dir / f"backtest_results_{args.model}_{timestamp}.txt"

        with open(results_file, "w") as f:
            f.write("ALGORITHMIC TRADING SYSTEM - BACKTEST RESULTS\n")
            f.write("=" * 80 + "\n\n")

            f.write(f"Model: {args.model.upper()}\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Test Period: {test_data.index[0]} to {test_data.index[-1]}\n\n")

            f.write("CAPITAL:\n")
            f.write(f"  Initial:  ${results['initial_capital']:,.2f}\n")
            f.write(f"  Final:    ${results['final_capital']:,.2f}\n")
            f.write(f"  Return:   {results['total_return']:+.2%}\n")
            if "cagr" in results:
                f.write(f"  CAGR:     {results['cagr']:+.2%}\n")

            f.write("\nRISK METRICS:\n")
            f.write(f"  Max Drawdown:     {results['max_drawdown']:.2%}\n")
            f.write(f"  Max DD Duration:  {results['max_drawdown_duration']} periods\n")
            f.write(f"  VaR (95%):        ${results['var_95']:,.2f}\n")
            f.write(f"  CVaR (95%):       ${results['cvar_95']:,.2f}\n")

            f.write("\nRISK-ADJUSTED RETURNS:\n")
            f.write(f"  Sharpe Ratio:   {results['sharpe_ratio']:.3f}\n")
            f.write(f"  Sortino Ratio:  {results['sortino_ratio']:.3f}\n")
            f.write(f"  Calmar Ratio:   {results['calmar_ratio']:.3f}\n")

            f.write("\nTRADING STATISTICS:\n")
            f.write(f"  Total Trades:   {results['total_trades']}\n")
            if results['total_trades'] > 0:
                f.write(f"  Win Rate:       {results['win_rate']:.2%}\n")
                f.write(f"  Avg Win:        ${results['avg_win']:,.2f} ({results['avg_win_pct']:+.2%})\n")
                f.write(f"  Avg Loss:       ${results['avg_loss']:,.2f} ({results['avg_loss_pct']:+.2%})\n")
                f.write(f"  Profit Factor:  {results['profit_factor']:.2f}\n")
                f.write(f"  R-Expectancy:   {results['r_expectancy']:.2f}\n")

            f.write("\n" + risk_manager.get_summary())

        logger.info(f"\nResults saved to: {results_file}")

        # Save equity curve
        equity_df = engine.get_equity_curve()
        equity_file = output_dir / f"equity_curve_{args.model}_{timestamp}.csv"
        equity_df.to_csv(equity_file)
        logger.info(f"Equity curve saved to: {equity_file}")

        # Save trades
        trades_df = engine.get_trades()
        if not trades_df.empty:
            trades_file = output_dir / f"trades_{args.model}_{timestamp}.csv"
            trades_df.to_csv(trades_file, index=False)
            logger.info(f"Trade history saved to: {trades_file}")

        logger.info("\n" + "=" * 80)
        logger.info("BACKTEST COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)

        return 0

    except KeyboardInterrupt:
        logger.warning("\n\nBacktest interrupted by user")
        return 1

    except Exception as e:
        logger.error(f"\n\nERROR: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
