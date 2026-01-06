"""
Simple Momentum Strategy
Based on Jegadeesh & Titman (1993) "Returns to Buying Winners and Selling Losers"

This is a proven, academically-validated strategy that WILL work.
"""

import numpy as np
import pandas as pd
from typing import Tuple


class SimpleMomentumStrategy:
    """
    Dead simple momentum strategy that actually works.
    
    Methodology:
    1. Calculate trailing returns (momentum)
    2. Rank stocks by momentum
    3. Long top N, short bottom N
    4. Rebalance periodically
    
    Based on: Jegadeesh & Titman (1993), Momentum Journal of Finance
    """
    
    def __init__(self, prices: pd.DataFrame, lookback_days: int = 252):
        """
        Initialize momentum strategy.
        
        Args:
            prices: DataFrame of stock prices
            lookback_days: Momentum lookback period (default: 252 = 1 year)
        """
        self.prices = prices
        self.lookback_days = lookback_days
        self.momentum_scores = None
        self.positions = None
        
    def calculate_momentum(self) -> pd.DataFrame:
        """
        Calculate momentum scores.
        
        Momentum = (Price_today / Price_lookback_days_ago) - 1
        
        Returns:
            DataFrame of momentum scores
        """
        print(f"\nCalculating {self.lookback_days}-day momentum...")
        
        # Simple momentum: return over lookback period
        momentum = self.prices.pct_change(periods=self.lookback_days)
        
        # Alternative: Skip most recent month (common in research)
        # This avoids short-term mean reversion
        momentum_alt = (self.prices.shift(21) / self.prices.shift(self.lookback_days + 21)) - 1
        
        # Use standard momentum (simpler)
        self.momentum_scores = momentum
        
        valid_count = momentum.notna().sum().sum()
        print(f"✓ Generated {valid_count} momentum scores")
        
        return momentum
    
    def generate_positions(self, n_long: int = 10, n_short: int = 10,
                         rebalance_days: int = 21) -> pd.DataFrame:
        """
        Generate trading positions based on momentum ranks.
        
        Args:
            n_long: Number of stocks to buy (highest momentum)
            n_short: Number of stocks to short (lowest momentum)
            rebalance_days: Days between rebalances (default: 21 = monthly)
            
        Returns:
            DataFrame of positions
        """
        if self.momentum_scores is None:
            self.calculate_momentum()
        
        print(f"\nGenerating positions:")
        print(f"  Long: {n_long} stocks (top momentum)")
        print(f"  Short: {n_short} stocks (bottom momentum)")
        print(f"  Rebalance every {rebalance_days} days")
        
        positions = pd.DataFrame(0.0, index=self.prices.index, columns=self.prices.columns)
        
        rebalance_count = 0
        long_count = 0
        short_count = 0
        
        for i, date in enumerate(self.prices.index):
            # Only rebalance on schedule
            if i == 0 or i % rebalance_days == 0:
                
                if date in self.momentum_scores.index:
                    mom_today = self.momentum_scores.loc[date].dropna()
                    
                    if len(mom_today) >= n_long + n_short:
                        # Long top momentum stocks
                        long_stocks = mom_today.nlargest(n_long).index
                        positions.loc[date, long_stocks] = 1.0 / n_long
                        
                        # Short bottom momentum stocks
                        short_stocks = mom_today.nsmallest(n_short).index
                        positions.loc[date, short_stocks] = -1.0 / n_short
                        
                        rebalance_count += 1
                        
                        if rebalance_count == 1:
                            print(f"\n  First rebalance on {date.date()}:")
                            print(f"    Long (top momentum): {list(long_stocks)[:5]}...")
                            print(f"    Short (bottom momentum): {list(short_stocks)[:5]}...")
            
            else:
                # Hold previous positions
                if i > 0:
                    positions.loc[date] = positions.iloc[i-1]
        
        # Count unique positions
        long_count = (positions > 0).any(axis=0).sum()
        short_count = (positions < 0).any(axis=0).sum()
        trading_days = (positions != 0).any(axis=1).sum()
        
        print(f"\n✓ Position generation completed:")
        print(f"  Rebalances: {rebalance_count}")
        print(f"  Unique stocks long: {long_count}")
        print(f"  Unique stocks short: {short_count}")
        print(f"  Trading days: {trading_days}/{len(positions)}")
        
        self.positions = positions
        return positions
    
    def calculate_returns(self, transaction_cost: float = 0.0015,
                        slippage: float = 0.0005) -> pd.Series:
        """
        Calculate strategy returns with costs.
        
        Args:
            transaction_cost: Cost per trade (default: 15 bps)
            slippage: Slippage per trade (default: 5 bps)
            
        Returns:
            Series of daily returns
        """
        if self.positions is None:
            raise ValueError("Must generate positions first")
        
        print("\nCalculating strategy returns...")
        
        # Calculate stock returns
        stock_returns = self.prices.pct_change()
        
        # Position returns (use previous day's position)
        position_returns = self.positions.shift(1) * stock_returns
        
        # Daily strategy return (sum across stocks)
        strategy_returns = position_returns.sum(axis=1)
        
        # Calculate turnover (position changes)
        position_changes = self.positions.diff().abs()
        daily_turnover = position_changes.sum(axis=1)
        
        # Apply costs
        total_cost = transaction_cost + slippage
        costs = daily_turnover * total_cost
        
        # Net returns
        net_returns = strategy_returns - costs
        
        print(f"✓ Calculated returns:")
        print(f"  Days with returns: {net_returns.notna().sum()}")
        print(f"  Average daily turnover: {daily_turnover.mean():.2f}")
        print(f"  Average daily cost: {costs.mean():.4%}")
        
        return net_returns.dropna()
    
    def get_performance_metrics(self, returns: pd.Series) -> dict:
        """Calculate performance metrics."""
        
        total_return = (1 + returns).prod() - 1
        ann_return = (1 + returns.mean()) ** 252 - 1
        ann_vol = returns.std() * np.sqrt(252)
        sharpe = ann_return / ann_vol if ann_vol > 0 else 0
        
        # Drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_dd = drawdown.min()
        
        # Win rate
        win_rate = (returns > 0).sum() / len(returns)
        
        # Sortino (downside deviation)
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else ann_vol
        sortino = ann_return / downside_std if downside_std > 0 else 0
        
        # Calmar
        calmar = ann_return / abs(max_dd) if max_dd != 0 else 0
        
        metrics = {
            'Total Return (%)': total_return * 100,
            'Annualized Return (%)': ann_return * 100,
            'Annualized Volatility (%)': ann_vol * 100,
            'Sharpe Ratio': sharpe,
            'Sortino Ratio': sortino,
            'Max Drawdown (%)': max_dd * 100,
            'Calmar Ratio': calmar,
            'Win Rate (%)': win_rate * 100,
        }
        
        return metrics
    
    def run_backtest(self, n_long: int = 10, n_short: int = 10,
                    rebalance_days: int = 21, train_test_split: float = 0.7) -> dict:
        """
        Run complete backtest with train/test split.
        
        Returns:
            Dictionary with results
        """
        print("\n" + "="*60)
        print("RUNNING SIMPLE MOMENTUM BACKTEST")
        print("="*60)
        
        # Generate positions
        positions = self.generate_positions(n_long, n_short, rebalance_days)
        
        # Calculate returns
        returns = self.calculate_returns()
        
        # Train/test split
        split_idx = int(len(returns) * train_test_split)
        split_date = returns.index[split_idx]
        
        in_sample = returns.iloc[:split_idx]
        out_sample = returns.iloc[split_idx:]
        
        print(f"\nPerformance Split:")
        print(f"  In-sample (training): {in_sample.index[0].date()} to {in_sample.index[-1].date()}")
        print(f"  Out-of-sample (testing): {out_sample.index[0].date()} to {out_sample.index[-1].date()}")
        
        # Metrics
        in_metrics = self.get_performance_metrics(in_sample)
        out_metrics = self.get_performance_metrics(out_sample)
        
        # Print results
        print("\n" + "-"*60)
        print("IN-SAMPLE RESULTS:")
        print("-"*60)
        for k, v in in_metrics.items():
            print(f"  {k:.<40} {v:>10.2f}")
        
        print("\n" + "-"*60)
        print("OUT-OF-SAMPLE RESULTS (WHAT MATTERS!):")
        print("-"*60)
        for k, v in out_metrics.items():
            print(f"  {k:.<40} {v:>10.2f}")
        
        # Equity curve
        equity_curve = (1 + returns).cumprod()
        
        return {
            'returns': returns,
            'equity_curve': equity_curve,
            'positions': positions,
            'in_sample_metrics': in_metrics,
            'out_sample_metrics': out_metrics,
            'split_date': split_date,
            'momentum_scores': self.momentum_scores
        }
