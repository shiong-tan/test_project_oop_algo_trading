"""Visualization module for trading system performance analysis.

This module provides comprehensive plotting capabilities for:
- Equity curves
- Drawdown charts
- Returns distribution
- Trade analysis
- Feature importance
- Confusion matrices
- ROC curves
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, List, Dict, Any, Tuple
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class PerformanceVisualizer:
    """Visualizer for trading system performance metrics."""

    def __init__(self, figsize: Tuple[int, int] = (12, 6), dpi: int = 100):
        """Initialize PerformanceVisualizer.

        Args:
            figsize: Default figure size (width, height)
            dpi: Resolution for saved figures
        """
        self.figsize = figsize
        self.dpi = dpi

    def plot_equity_curve(
        self,
        equity_curve: pd.DataFrame,
        benchmark: Optional[pd.Series] = None,
        title: str = "Equity Curve",
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Plot equity curve with optional benchmark comparison.

        Args:
            equity_curve: DataFrame with equity values (index=dates, column='equity')
            benchmark: Optional benchmark series for comparison
            title: Plot title
            save_path: Path to save figure (optional)

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)

        # Plot equity curve
        if isinstance(equity_curve, pd.DataFrame):
            equity = equity_curve['equity'] if 'equity' in equity_curve.columns else equity_curve.iloc[:, 0]
        else:
            equity = equity_curve

        ax.plot(equity.index, equity.values, label='Strategy', linewidth=2, color='#2E86AB')

        # Plot benchmark if provided
        if benchmark is not None:
            ax.plot(benchmark.index, benchmark.values, label='Benchmark',
                   linewidth=2, linestyle='--', color='#A23B72', alpha=0.7)

        # Formatting
        ax.set_xlabel('Date', fontsize=12, fontweight='bold')
        ax.set_ylabel('Portfolio Value ($)', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)

        # Add stats box
        initial = equity.iloc[0]
        final = equity.iloc[-1]
        total_return = (final - initial) / initial
        stats_text = f"Initial: ${initial:,.0f}\nFinal: ${final:,.0f}\nReturn: {total_return:+.2%}"
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
               fontsize=9)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Equity curve saved to {save_path}")

        return fig

    def plot_drawdown(
        self,
        equity_curve: pd.DataFrame,
        title: str = "Drawdown",
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Plot drawdown chart.

        Args:
            equity_curve: DataFrame with equity values
            title: Plot title
            save_path: Path to save figure (optional)

        Returns:
            Matplotlib figure
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.figsize, dpi=self.dpi,
                                       sharex=True, gridspec_kw={'height_ratios': [2, 1]})

        # Extract equity
        if isinstance(equity_curve, pd.DataFrame):
            equity = equity_curve['equity'] if 'equity' in equity_curve.columns else equity_curve.iloc[:, 0]
        else:
            equity = equity_curve

        # Calculate drawdown
        running_max = equity.expanding().max()
        drawdown = (equity - running_max) / running_max

        # Plot equity with running max
        ax1.plot(equity.index, equity.values, label='Equity', linewidth=2, color='#2E86AB')
        ax1.plot(running_max.index, running_max.values, label='Peak',
                linewidth=1.5, linestyle='--', color='#A23B72', alpha=0.7)
        ax1.set_ylabel('Portfolio Value ($)', fontsize=12, fontweight='bold')
        ax1.set_title(title, fontsize=14, fontweight='bold', pad=20)
        ax1.legend(loc='best', fontsize=10)
        ax1.grid(True, alpha=0.3)

        # Plot drawdown
        ax2.fill_between(drawdown.index, drawdown.values * 100, 0,
                         alpha=0.3, color='red', label='Drawdown')
        ax2.plot(drawdown.index, drawdown.values * 100, linewidth=2, color='darkred')
        ax2.set_xlabel('Date', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Drawdown (%)', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)

        # Add max drawdown line
        max_dd = drawdown.min() * 100
        ax2.axhline(y=max_dd, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
        ax2.text(0.02, 0.98, f"Max Drawdown: {max_dd:.2f}%", transform=ax2.transAxes,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                fontsize=9)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Drawdown chart saved to {save_path}")

        return fig

    def plot_returns_distribution(
        self,
        returns: pd.Series,
        title: str = "Returns Distribution",
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Plot returns distribution with statistics.

        Args:
            returns: Series of returns
            title: Plot title
            save_path: Path to save figure (optional)

        Returns:
            Matplotlib figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.figsize, dpi=self.dpi)

        # Histogram
        ax1.hist(returns.dropna() * 100, bins=50, alpha=0.7, color='#2E86AB', edgecolor='black')
        ax1.axvline(returns.mean() * 100, color='red', linestyle='--', linewidth=2, label='Mean')
        ax1.axvline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)
        ax1.set_xlabel('Returns (%)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax1.set_title('Returns Histogram', fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Q-Q plot
        from scipy import stats
        stats.probplot(returns.dropna(), dist="norm", plot=ax2)
        ax2.set_title('Q-Q Plot', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)

        # Add statistics
        stats_text = (
            f"Mean: {returns.mean()*100:.3f}%\n"
            f"Std: {returns.std()*100:.3f}%\n"
            f"Skew: {returns.skew():.3f}\n"
            f"Kurt: {returns.kurtosis():.3f}"
        )
        ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                fontsize=9)

        fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Returns distribution saved to {save_path}")

        return fig

    def plot_trade_analysis(
        self,
        trades: pd.DataFrame,
        title: str = "Trade Analysis",
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Plot trade analysis including P&L distribution and cumulative P&L.

        Args:
            trades: DataFrame with trade records (requires 'pnl' column)
            title: Plot title
            save_path: Path to save figure (optional)

        Returns:
            Matplotlib figure
        """
        if trades.empty:
            logger.warning("No trades to plot")
            fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
            ax.text(0.5, 0.5, 'No Trades', ha='center', va='center', fontsize=20)
            return fig

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10), dpi=self.dpi)

        # 1. P&L per trade
        colors = ['green' if x > 0 else 'red' for x in trades['pnl']]
        ax1.bar(range(len(trades)), trades['pnl'], color=colors, alpha=0.6, edgecolor='black')
        ax1.axhline(y=0, color='black', linestyle='-', linewidth=1)
        ax1.set_xlabel('Trade Number', fontsize=10, fontweight='bold')
        ax1.set_ylabel('P&L ($)', fontsize=10, fontweight='bold')
        ax1.set_title('P&L per Trade', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y')

        # 2. Cumulative P&L
        cumulative_pnl = trades['pnl'].cumsum()
        ax2.plot(cumulative_pnl.values, linewidth=2, color='#2E86AB')
        ax2.fill_between(range(len(cumulative_pnl)), cumulative_pnl.values, alpha=0.3, color='#2E86AB')
        ax2.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax2.set_xlabel('Trade Number', fontsize=10, fontweight='bold')
        ax2.set_ylabel('Cumulative P&L ($)', fontsize=10, fontweight='bold')
        ax2.set_title('Cumulative P&L', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)

        # 3. Win/Loss distribution
        wins = trades[trades['pnl'] > 0]['pnl']
        losses = trades[trades['pnl'] < 0]['pnl']

        ax3.hist([wins, losses], bins=20, label=['Wins', 'Losses'],
                color=['green', 'red'], alpha=0.6, edgecolor='black')
        ax3.set_xlabel('P&L ($)', fontsize=10, fontweight='bold')
        ax3.set_ylabel('Frequency', fontsize=10, fontweight='bold')
        ax3.set_title('Win/Loss Distribution', fontsize=12, fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 4. Trade statistics
        ax4.axis('off')
        total_trades = len(trades)
        winning_trades = len(wins)
        losing_trades = len(losses)
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        avg_win = wins.mean() if len(wins) > 0 else 0
        avg_loss = losses.mean() if len(losses) > 0 else 0
        profit_factor = abs(wins.sum() / losses.sum()) if len(losses) > 0 and losses.sum() != 0 else float('inf')

        stats_text = f"""
        TRADE STATISTICS
        {'='*40}
        Total Trades:     {total_trades}
        Winning Trades:   {winning_trades} ({win_rate:.1%})
        Losing Trades:    {losing_trades} ({(1-win_rate):.1%})

        Average Win:      ${avg_win:,.2f}
        Average Loss:     ${avg_loss:,.2f}
        Profit Factor:    {profit_factor:.2f}

        Total P&L:        ${trades['pnl'].sum():,.2f}
        Best Trade:       ${trades['pnl'].max():,.2f}
        Worst Trade:      ${trades['pnl'].min():,.2f}
        """

        ax4.text(0.1, 0.5, stats_text, transform=ax4.transAxes,
                verticalalignment='center', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3),
                fontsize=10)

        fig.suptitle(title, fontsize=14, fontweight='bold', y=0.995)
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Trade analysis saved to {save_path}")

        return fig

    def plot_feature_importance(
        self,
        feature_importance: pd.DataFrame,
        top_n: int = 20,
        title: str = "Feature Importance",
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Plot feature importance from model.

        Args:
            feature_importance: DataFrame with 'feature' and 'importance' columns
            top_n: Number of top features to display
            title: Plot title
            save_path: Path to save figure (optional)

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)

        # Get top N features
        top_features = feature_importance.head(top_n).sort_values('importance')

        # Plot horizontal bar chart
        colors = plt.cm.viridis(np.linspace(0, 1, len(top_features)))
        ax.barh(range(len(top_features)), top_features['importance'], color=colors, edgecolor='black')
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features['feature'])
        ax.set_xlabel('Importance', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3, axis='x')

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Feature importance plot saved to {save_path}")

        return fig

    def plot_confusion_matrix(
        self,
        confusion_matrix: np.ndarray,
        title: str = "Confusion Matrix",
        labels: Optional[List[str]] = None,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Plot confusion matrix heatmap.

        Args:
            confusion_matrix: 2D array of confusion matrix
            title: Plot title
            labels: Class labels (default: ['Down', 'Up'])
            save_path: Path to save figure (optional)

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(8, 6), dpi=self.dpi)

        if labels is None:
            labels = ['Down', 'Up']

        # Plot heatmap
        sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues',
                   xticklabels=labels, yticklabels=labels, ax=ax,
                   cbar_kws={'label': 'Count'})

        ax.set_xlabel('Predicted', fontsize=12, fontweight='bold')
        ax.set_ylabel('Actual', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Confusion matrix saved to {save_path}")

        return fig

    def plot_roc_curve(
        self,
        fpr: np.ndarray,
        tpr: np.ndarray,
        roc_auc: float,
        title: str = "ROC Curve",
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Plot ROC curve.

        Args:
            fpr: False positive rate array
            tpr: True positive rate array
            roc_auc: Area under ROC curve
            title: Plot title
            save_path: Path to save figure (optional)

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(8, 8), dpi=self.dpi)

        # Plot ROC curve
        ax.plot(fpr, tpr, color='#2E86AB', linewidth=2,
               label=f'ROC curve (AUC = {roc_auc:.3f})')

        # Plot diagonal line (random classifier)
        ax.plot([0, 1], [0, 1], color='gray', linestyle='--', linewidth=2,
               label='Random classifier')

        ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
        ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='lower right', fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"ROC curve saved to {save_path}")

        return fig

    def create_summary_report(
        self,
        equity_curve: pd.DataFrame,
        trades: pd.DataFrame,
        metrics: Dict[str, Any],
        feature_importance: Optional[pd.DataFrame] = None,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Create comprehensive summary report with multiple plots.

        Args:
            equity_curve: DataFrame with equity values
            trades: DataFrame with trade records
            metrics: Dictionary of performance metrics
            feature_importance: Optional feature importance DataFrame
            save_path: Path to save figure (optional)

        Returns:
            Matplotlib figure
        """
        # Determine grid layout based on available data
        if feature_importance is not None:
            fig = plt.figure(figsize=(16, 12), dpi=self.dpi)
            gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
            ax1 = fig.add_subplot(gs[0, :])
            ax2 = fig.add_subplot(gs[1, 0])
            ax3 = fig.add_subplot(gs[1, 1])
            ax4 = fig.add_subplot(gs[2, 0])
            ax5 = fig.add_subplot(gs[2, 1])
        else:
            fig = plt.figure(figsize=(16, 10), dpi=self.dpi)
            gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
            ax1 = fig.add_subplot(gs[0, :])
            ax2 = fig.add_subplot(gs[1, 0])
            ax3 = fig.add_subplot(gs[1, 1])

        # Extract equity
        if isinstance(equity_curve, pd.DataFrame):
            equity = equity_curve['equity'] if 'equity' in equity_curve.columns else equity_curve.iloc[:, 0]
        else:
            equity = equity_curve

        # 1. Equity curve
        ax1.plot(equity.index, equity.values, linewidth=2, color='#2E86AB')
        ax1.set_xlabel('Date', fontsize=10, fontweight='bold')
        ax1.set_ylabel('Portfolio Value ($)', fontsize=10, fontweight='bold')
        ax1.set_title('Equity Curve', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)

        # 2. Drawdown
        running_max = equity.expanding().max()
        drawdown = (equity - running_max) / running_max
        ax2.fill_between(drawdown.index, drawdown.values * 100, 0, alpha=0.3, color='red')
        ax2.plot(drawdown.index, drawdown.values * 100, linewidth=2, color='darkred')
        ax2.set_xlabel('Date', fontsize=10, fontweight='bold')
        ax2.set_ylabel('Drawdown (%)', fontsize=10, fontweight='bold')
        ax2.set_title('Drawdown', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)

        # 3. Performance metrics
        ax3.axis('off')
        metrics_text = f"""
        PERFORMANCE METRICS
        {'='*40}
        Total Return:     {metrics.get('total_return', 0)*100:+.2f}%
        CAGR:             {metrics.get('cagr', 0)*100:+.2f}%

        Max Drawdown:     {metrics.get('max_drawdown', 0)*100:.2f}%
        VaR (95%):        ${metrics.get('var_95', 0):,.2f}
        CVaR (95%):       ${metrics.get('cvar_95', 0):,.2f}

        Sharpe Ratio:     {metrics.get('sharpe_ratio', 0):.3f}
        Sortino Ratio:    {metrics.get('sortino_ratio', 0):.3f}
        Calmar Ratio:     {metrics.get('calmar_ratio', 0):.3f}

        Total Trades:     {metrics.get('total_trades', 0)}
        Win Rate:         {metrics.get('win_rate', 0)*100:.1f}%
        Profit Factor:    {metrics.get('profit_factor', 0):.2f}
        """
        ax3.text(0.1, 0.5, metrics_text, transform=ax3.transAxes,
                verticalalignment='center', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3),
                fontsize=10)

        # 4. Feature importance (if provided)
        if feature_importance is not None:
            top_features = feature_importance.head(15).sort_values('importance')
            colors = plt.cm.viridis(np.linspace(0, 1, len(top_features)))
            ax4.barh(range(len(top_features)), top_features['importance'], color=colors)
            ax4.set_yticks(range(len(top_features)))
            ax4.set_yticklabels(top_features['feature'], fontsize=8)
            ax4.set_xlabel('Importance', fontsize=10, fontweight='bold')
            ax4.set_title('Top 15 Features', fontsize=12, fontweight='bold')
            ax4.grid(True, alpha=0.3, axis='x')

            # 5. Trade P&L distribution
            if not trades.empty:
                colors_pnl = ['green' if x > 0 else 'red' for x in trades['pnl']]
                ax5.bar(range(len(trades)), trades['pnl'], color=colors_pnl, alpha=0.6)
                ax5.axhline(y=0, color='black', linestyle='-', linewidth=1)
                ax5.set_xlabel('Trade Number', fontsize=10, fontweight='bold')
                ax5.set_ylabel('P&L ($)', fontsize=10, fontweight='bold')
                ax5.set_title('P&L per Trade', fontsize=12, fontweight='bold')
                ax5.grid(True, alpha=0.3, axis='y')

        fig.suptitle('Trading System Performance Report', fontsize=16, fontweight='bold', y=0.995)

        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Summary report saved to {save_path}")

        return fig


def create_visualizer(figsize: Tuple[int, int] = (12, 6), dpi: int = 100) -> PerformanceVisualizer:
    """Factory function to create PerformanceVisualizer.

    Args:
        figsize: Default figure size
        dpi: Resolution for saved figures

    Returns:
        PerformanceVisualizer instance
    """
    return PerformanceVisualizer(figsize=figsize, dpi=dpi)
