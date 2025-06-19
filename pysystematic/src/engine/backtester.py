import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class GenericBacktester:
    """Generic backtesting framework for multiple strategies"""
    def __init__(self, initial_capital=1000000, transaction_cost=0.001):
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.results = {}
    
    def backtest(self, signals, returns, strategy_name, position_sizing='equal_weight'):
        """
        Generic backtesting function
        Parameters:
        - signals: pd.Series with values -1, 0, 1
        - returns: pd.Series of asset returns
        - strategy_name: str
        - position_sizing: str ('equal_weight', 'volatility_scaled', 'kelly')
        """
        # Align signals and returns
        aligned_data = pd.concat([signals, returns], axis=1, join='inner')
        aligned_data.columns = ['signals', 'returns']
        aligned_data = aligned_data.dropna()
        if len(aligned_data) == 0:
            raise ValueError("No overlapping data between signals and returns")
        # Calculate position sizes
        positions = self._calculate_positions(
            aligned_data['signals'], aligned_data['returns'], method=position_sizing
        )
        # Calculate strategy returns
        strategy_returns = positions.shift(1) * aligned_data['returns']
        # Apply transaction costs
        position_changes = positions.diff().abs()
        transaction_costs = position_changes * self.transaction_cost
        strategy_returns = strategy_returns - transaction_costs
        # Calculate performance metrics
        performance = self._calculate_performance(strategy_returns, aligned_data['returns'])
        self.results[strategy_name] = {
            'returns': strategy_returns,
            'positions': positions,
            'performance': performance,
            'signals': aligned_data['signals']
        }
        return performance
    
    def _calculate_positions(self, signals, returns, method='equal_weight'):
        """Calculate position sizes based on method"""
        if method == 'equal_weight':
            return signals
        
        elif method == 'volatility_scaled':
            vol = returns.rolling(window=22).std() * np.sqrt(252)
            target_vol = 0.15  # 15% target volatility
            vol_scalar = target_vol / vol.replace(0, np.nan)
            vol_scalar = vol_scalar.clip(0.1, 2.0)  # Limit scaling
            return signals * vol_scalar
        elif method == 'kelly':
            # Simplified Kelly criterion
            win_rate = (returns[signals == 1] > 0).mean() if len(returns[signals == 1]) > 0 else 0.5
            avg_win = returns[returns > 0].mean() if len(returns[returns > 0]) > 0 else 0.01
            avg_loss = abs(returns[returns < 0].mean()) if len(returns[returns < 0]) > 0 else 0.01
            
            kelly_fraction = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
            kelly_fraction = np.clip(kelly_fraction, 0, 0.25)  # Cap at 25%
            return signals * kelly_fraction
        else:
            return signals
    
    def _calculate_performance(self, strategy_returns, benchmark_returns):
        """Calculate comprehensive performance metrics"""
        strategy_returns = strategy_returns.dropna()
        if len(strategy_returns) == 0:
            return self._empty_performance()
        
        total_return = (1 + strategy_returns).prod() - 1
        annualized_return = (1 + strategy_returns).prod() ** (252 / len(strategy_returns)) - 1
        annualized_vol = strategy_returns.std() * np.sqrt(252)
        sharpe_ratio = annualized_return / annualized_vol if annualized_vol > 0 else 0
        
        # Drawdown analysis
        cumulative_returns = (1 + strategy_returns).cumprod()
        rolling_max = cumulative_returns.cummax()
        drawdown = (cumulative_returns - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        # Additional metrics
        win_rate = (strategy_returns > 0).sum() / len(strategy_returns)
        profit_factor = abs(strategy_returns[strategy_returns > 0].sum() / 
                           strategy_returns[strategy_returns < 0].sum()) if strategy_returns[strategy_returns < 0].sum() != 0 else np.inf
        # Sortino ratio
        downside_returns = strategy_returns[strategy_returns < 0]
        downside_deviation = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
        sortino_ratio = annualized_return / downside_deviation if downside_deviation > 0 else np.inf
        # Calmar ratio
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else np.inf
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'annualized_volatility': annualized_vol,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'num_trades': len(strategy_returns),
            'avg_trade': strategy_returns.mean(),
            'std_trade': strategy_returns.std()
        }
    
    def _empty_performance(self):
        """Return empty performance dict for failed backtests"""
        return {key: 0.0 for key in [
            'total_return', 'annualized_return', 'annualized_volatility',
            'sharpe_ratio', 'sortino_ratio', 'calmar_ratio', 'max_drawdown',
            'win_rate', 'profit_factor', 'num_trades', 'avg_trade', 'std_trade'
        ]}
    
    def get_summary(self):
        """Get summary of all backtested strategies"""
        if not self.results:
            return pd.DataFrame()
        
        summary_data = {}
        for strategy_name, result in self.results.items():
            summary_data[strategy_name] = result['performance']
        
        return pd.DataFrame(summary_data).T
    
    def plot_results(self, strategies=None):
        """Plot cumulative returns for strategies"""
        if strategies is None:
            strategies = list(self.results.keys())
        
        plt.figure(figsize=(12, 6))
        
        for strategy in strategies:
            if strategy in self.results:
                returns = self.results[strategy]['returns']
                cumulative = (1 + returns).cumprod()
                plt.plot(cumulative.index, cumulative.values, label=strategy, linewidth=2)
        
        plt.title('Strategy Performance Comparison')
        plt.xlabel('Date')
        plt.ylabel('Cumulative Returns')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
