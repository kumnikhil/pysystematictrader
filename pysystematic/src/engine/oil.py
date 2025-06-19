import numpy as np 
from typing import List, Dict, Optional
import datetime as dt
import logging
from src.utils.data_objs import Position, RiskMetrics
from src.quant.models.BSM import TorchBlackScholes
from src.marketdata.feeds import MarketDataFeed
from src.quant.forward_curves.TwoFactor import TwoFactorStochasticVolatilityModel
import pandas as pd
from scipy.stats import norm
import torch 
from torch.autograd import grad
from scipy.optimize import minimize_scalar

class OilTradingEngine:
    """Enhanced trading engine with PyTorch Greeks computation"""
    
    def __init__(self):
        self.model = TwoFactorStochasticVolatilityModel()
        self.market_data = MarketDataFeed()
        self.portfolio: List[Position] = []
        self.last_calibration = None
        self.bs_model = TorchBlackScholes()
        
        # Risk-free rate (for options pricing)
        self.risk_free_rate = 0.05
        
        self.risk_limits = {
            'var_1d_limit': 1000000,      # $1M daily VaR limit
            'var_10d_limit': 3000000,     # $3M 10-day VaR limit
            'delta_limit': 5000,          # 5000 barrel delta limit
            'gamma_limit': 100,           # 100 gamma limit
            'vega_limit': 500000,         # $500k vega limit
            'theta_limit': 50000,         # $50k theta limit
        }
        
        # Setup logging
        logging.basicConfig(level=logging.INFO, 
                          format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger('OilTradingEngine')
    
    def morning_calibration(self) -> bool:
        """Calibrate model parameters using latest market data"""
        try:
            self.logger.info("Starting morning calibration...")
            
            # Fetch forward curve data
            forward_prices, maturities = self.market_data.get_forward_curve()
            
            if len(forward_prices) < 3:
                self.logger.warning("Insufficient market data for calibration")
                return False
            
            # Calibrate model
            self.model.set_parameters()  # Initialize with defaults
            
            # For production: implement sophisticated calibration to implied vols
            # For now, use market data structure to inform parameters
            historical_data = self._get_historical_data()
            if historical_data is not None:
                self._calibrate_to_historical_vol(historical_data, maturities)
            
            # Store initial curve for simulation
            self.initial_curve = forward_prices
            self.curve_maturities = maturities
            self.last_calibration = dt.datetime.now()
            
            self.logger.info(f"Calibration complete. Parameters: σ={self.model.sigma:.3f}, "
                           f"β₁={self.model.beta1:.3f}, β₂={self.model.beta2:.3f}")
            return True
            
        except Exception as e:
            self.logger.error(f"Calibration failed: {e}")
            return False
    
    def _get_historical_data(self, days: int = 30) -> Optional[pd.DataFrame]:
        """Get historical price data for volatility estimation"""
        try:
            end_date = dt.datetime.now()
            start_date = end_date - dt.timedelta(days=days)
            hist, ric_list = self.market_data.wti.forward_curve_history(
                start_date=start_date.strftime('%Y-%m-%d'),
                end_date=end_date.strftime('%Y-%m-%d'),
                forward_maturity=6
            )
            return self.market_data.wti.close_prices(hist, ric_list)
        except:
            return None
    
    def _calibrate_to_historical_vol(self, historical_data: pd.DataFrame, maturities: np.ndarray):
        """Calibrate model to historical volatilities"""
        if historical_data is None or historical_data.empty:
            return
            
        try:
            # Calculate realized volatilities
            returns = historical_data.pct_change().dropna()
            realized_vols = returns.std() * np.sqrt(252)  # Annualized
            
            # Adjust sigma to match front-month realized vol
            if len(realized_vols) > 0:
                front_vol = realized_vols.iloc[0]
                if 0.1 < front_vol < 1.0:  # Sanity check
                    self.model.sigma = front_vol * 0.8  # Scale down slightly
                    
            # Adjust beta parameters based on term structure shape
            if len(realized_vols) > 1:
                vol_slope = (realized_vols.iloc[-1] - realized_vols.iloc[0]) / len(realized_vols)
                if vol_slope < -0.05:  # Strong Samuelson effect
                    self.model.beta1 = max(0.05, self.model.beta1 * 1.2)
                elif vol_slope > 0:  # Weak or inverse Samuelson
                    self.model.beta1 = max(0.02, self.model.beta1 * 0.8)
                    
        except Exception as e:
            self.logger.warning(f"Historical calibration adjustment failed: {e}")
    
    def price_option(self, strike: float, maturity: float, option_type: str, 
                    underlying_maturity: float = None, use_model_vol: bool = True) -> float:
        """Price European option using calibrated model or Black-Scholes"""
        try:
            if underlying_maturity is None:
                underlying_maturity = maturity
                
            # Get current forward price
            F0 = self.market_data.get_current_price(underlying_maturity)
            
            if use_model_vol:
                # Use stochastic volatility model
                price = self.model.european_option_price(F0, strike, maturity, maturity, option_type)
            else:
                # Use Black-Scholes with market implied vol
                implied_vol = self.market_data.get_implied_volatility(maturity)
                price = self._black_scholes_price(F0, strike, maturity, implied_vol, option_type)
            
            self.logger.debug(f"Priced {option_type} option: K={strike}, T={maturity}, Price={price:.2f}")
            return price
            
        except Exception as e:
            self.logger.error(f"Option pricing failed: {e}")
            return 0.0
    
    def _black_scholes_price(self, F: float, K: float, T: float, sigma: float, option_type: str) -> float:
        """Black-Scholes pricing for comparison"""
        if T <= 0:
            if option_type == 'call':
                return max(0, F - K)
            else:
                return max(0, K - F)
        
        d1 = (np.log(F / K) + (self.risk_free_rate + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        if option_type == 'call':
            price = np.exp(-self.risk_free_rate * T) * (F * norm.cdf(d1) - K * norm.cdf(d2))
        else:
            price = np.exp(-self.risk_free_rate * T) * (K * norm.cdf(-d2) - F * norm.cdf(-d1))
        
        return price
    
    def calculate_option_greeks_torch(self, position: Position) -> Dict[str, float]:
        """Calculate option Greeks using PyTorch automatic differentiation"""
        if position.option_type is None:
            # Futures position
            return {
                'delta': 1.0,
                'gamma': 0.0,
                'vega': 0.0,
                'theta': 0.0,
                'rho': 0.0
            }
        
        try:
            # Get market parameters
            F0 = self.market_data.get_current_price(position.maturity)
            implied_vol = self.market_data.get_implied_volatility(position.maturity)
            
            # Create tensors with gradients enabled
            F = torch.tensor(F0, dtype=torch.float64, requires_grad=True)
            K = torch.tensor(position.strike, dtype=torch.float64)
            T = torch.tensor(position.maturity, dtype=torch.float64, requires_grad=True)
            sigma = torch.tensor(implied_vol, dtype=torch.float64, requires_grad=True)
            r = torch.tensor(self.risk_free_rate, dtype=torch.float64, requires_grad=True)
            
            # Calculate option price
            option_price = self.bs_model(F, K, T, sigma, r, position.option_type)
            
            # Calculate first-order Greeks
            delta = grad(option_price, F, create_graph=True, retain_graph=True)[0]
            vega = grad(option_price, sigma, create_graph=True, retain_graph=True)[0]
            theta = -grad(option_price, T, create_graph=True, retain_graph=True)[0]
            rho = grad(option_price, r, create_graph=True, retain_graph=True)[0]
            
            # Calculate second-order Greek (Gamma)
            gamma = grad(delta, F, create_graph=True, retain_graph=True)[0]
            
            return {
                'delta': float(delta.item()),
                'gamma': float(gamma.item()),
                'vega': float(vega.item()),
                'theta': float(theta.item()),
                'rho': float(rho.item())
            }
            
        except Exception as e:
            self.logger.error(f"Greeks calculation failed for {position.instrument}: {e}")
            return {'delta': 0.0, 'gamma': 0.0, 'vega': 0.0, 'theta': 0.0, 'rho': 0.0}
    
    def calculate_portfolio_greeks(self) -> Dict[str, float]:
        """Calculate portfolio Greeks using PyTorch autograd"""
        try:
            total_delta = 0.0
            total_gamma = 0.0
            total_vega = 0.0
            total_theta = 0.0
            total_rho = 0.0
            
            for position in self.portfolio:
                # Calculate Greeks for this position
                greeks = self.calculate_option_greeks_torch(position)
                
                # Scale by position size
                total_delta += position.quantity * greeks['delta']
                total_gamma += position.quantity * greeks['gamma']
                total_vega += position.quantity * greeks['vega']
                total_theta += position.quantity * greeks['theta']
                total_rho += position.quantity * greeks['rho']
                
                self.logger.debug(f"Position {position.instrument}: "
                                f"Delta={greeks['delta']:.3f}, "
                                f"Gamma={greeks['gamma']:.4f}, "
                                f"Vega={greeks['vega']:.2f}")
            
            return {
                'delta': total_delta,
                'gamma': total_gamma,
                'vega': total_vega,
                'theta': total_theta,
                'rho': total_rho
            }
            
        except Exception as e:
            self.logger.error(f"Portfolio Greeks calculation failed: {e}")
            return {'delta': 0, 'gamma': 0, 'vega': 0, 'theta': 0, 'rho': 0}
    
    def calculate_option_implied_vol(self, position: Position, market_price: float) -> float:
        """Calculate implied volatility from market price using optimization"""
        if position.option_type is None:
            return 0.0
        
        try:
            F0 = self.market_data.get_current_price(position.maturity)
            
            def objective(vol):
                try:
                    model_price = self._black_scholes_price(
                        F0, position.strike, position.maturity, vol, position.option_type
                    )
                    return (model_price - market_price) ** 2
                except:
                    return float('inf')
            
            # Optimize to find implied volatility
            result = minimize_scalar(objective, bounds=(0.01, 2.0), method='bounded')
            
            if result.success:
                return result.x
            else:
                return self.market_data.get_implied_volatility(position.maturity)
                
        except Exception as e:
            self.logger.error(f"Implied vol calculation failed: {e}")
            return 0.3  # Default fallback
    
    def add_position(self, position: Position):
        """Add position to portfolio"""
        # Update current price
        if position.option_type is None:  # Futures position
            position.current_price = self.market_data.get_current_price(position.maturity)
        else:  # Option position
            position.current_price = self.price_option(
                position.strike, position.maturity, position.option_type
            )
        
        self.portfolio.append(position)
        self.logger.info(f"Added position: {position.instrument}, Qty: {position.quantity}")
        
        # Calculate and log Greeks for this position
        if position.option_type is not None:
            greeks = self.calculate_option_greeks_torch(position)
            self.logger.info(f"Position Greeks - Delta: {greeks['delta']:.3f}, "
                           f"Gamma: {greeks['gamma']:.4f}, "
                           f"Vega: {greeks['vega']:.2f}")
    
    def calculate_portfolio_var(self, horizon_days: int = 1, confidence: float = 0.95, 
                              n_simulations: int = 10000) -> float:
        """Calculate portfolio Value-at-Risk using Monte Carlo simulation"""
        try:
            if not self.portfolio:
                return 0.0
            
            # Convert horizon to years
            horizon_years = horizon_days / 365.0
            
            # Generate price scenarios
            times, maturities, forward_paths, _ = self.model.simulate_forward_paths(
                T_max=horizon_years,
                n_steps=max(1, horizon_days),
                n_paths=n_simulations,
                n_maturities=len(self.curve_maturities),
                initial_curve=self.initial_curve,
                curve_maturities=self.curve_maturities
            )
            
            # Calculate P&L for each scenario
            portfolio_pnl = []
            
            for path_idx in range(n_simulations):
                scenario_pnl = 0.0
                
                for position in self.portfolio:
                    if position.option_type is None:  # Futures position
                        # Interpolate future price for this maturity
                        future_price = np.interp(position.maturity, maturities, 
                                               forward_paths[path_idx, -1, :])
                        position_pnl = position.quantity * (future_price - position.current_price)
                        
                    else:  # Option position
                        # Use Greeks for P&L approximation (Taylor expansion)
                        underlying_price = np.interp(position.maturity, maturities,
                                                   forward_paths[path_idx, -1, :])
                        
                        greeks = self.calculate_option_greeks_torch(position)
                        price_change = underlying_price - self.market_data.get_current_price(position.maturity)
                        
                        # First and second order approximation
                        delta_pnl = greeks['delta'] * price_change
                        gamma_pnl = 0.5 * greeks['gamma'] * price_change**2
                        theta_pnl = greeks['theta'] * horizon_years
                        
                        position_pnl = position.quantity * (delta_pnl + gamma_pnl + theta_pnl)
                    
                    scenario_pnl += position_pnl
                
                portfolio_pnl.append(scenario_pnl)
            
            # Calculate VaR
            portfolio_pnl = np.array(portfolio_pnl)
            var = -np.percentile(portfolio_pnl, (1 - confidence) * 100)
            
            return var
            
        except Exception as e:
            self.logger.error(f"VaR calculation failed: {e}")
            return 0.0
    
    def stress_test(self) -> Dict[str, float]:
        """Run stress test scenarios with proper Greeks-based calculation"""
        scenarios = {
            'opec_production_cut': {'price_shock': 15, 'vol_shock': 5},
            'global_recession': {'price_shock': -25, 'vol_shock': 10},
            'middle_east_conflict': {'price_shock': 30, 'vol_shock': 15},
            'renewable_breakthrough': {'price_shock': -20, 'vol_shock': 8},
            'china_demand_surge': {'price_shock': 18, 'vol_shock': 6}
        }
        
        results = {}
        
        try:
            # Calculate current portfolio Greeks
            portfolio_greeks = self.calculate_portfolio_greeks()
            
            for scenario_name, shocks in scenarios.items():
                scenario_pnl = 0.0
                price_shock_pct = shocks['price_shock'] / 100
                vol_shock_pct = shocks['vol_shock'] / 100
                
                for position in self.portfolio:
                    if position.option_type is None:  # Futures position
                        current_price = position.current_price
                        shocked_price = current_price * (1 + price_shock_pct)
                        position_pnl = position.quantity * (shocked_price - current_price)
                        
                    else:  # Option position
                        # Use Greeks for stress testing
                        greeks = self.calculate_option_greeks_torch(position)
                        current_underlying = self.market_data.get_current_price(position.maturity)
                        
                        # Price shock impact
                        price_change = current_underlying * price_shock_pct
                        delta_pnl = greeks['delta'] * price_change
                        gamma_pnl = 0.5 * greeks['gamma'] * price_change**2
                        
                        # Volatility shock impact
                        current_vol = self.market_data.get_implied_volatility(position.maturity)
                        vol_change = current_vol * vol_shock_pct
                        vega_pnl = greeks['vega'] * vol_change
                        
                        position_pnl = position.quantity * (delta_pnl + gamma_pnl + vega_pnl)
                    
                    scenario_pnl += position_pnl
                
                results[scenario_name] = scenario_pnl         
        except Exception as e:
            self.logger.error(f"Stress test failed: {e}")
            
        return results
    
    def calculate_risk_metrics(self) -> RiskMetrics:
        """Calculate comprehensive risk metrics"""
        try:
            # VaR calculations
            var_1d = self.calculate_portfolio_var(horizon_days=1, confidence=0.95)
            var_10d = self.calculate_portfolio_var(horizon_days=10, confidence=0.95)
            # Expected Shortfall (average loss beyond VaR)
            es_1d = self.calculate_portfolio_var(horizon_days=1, confidence=0.975)
            # Greeks using PyTorch
            greeks = self.calculate_portfolio_greeks()
            # Stress scenarios
            stress_results = self.stress_test()
            # Total exposure
            total_exposure = sum(abs(pos.market_value) for pos in self.portfolio)
            return RiskMetrics(
                var_1d=var_1d,
                var_10d=var_10d,
                expected_shortfall=es_1d,
                delta=greeks['delta'],
                gamma=greeks['gamma'],
                vega=greeks['vega'],
                theta=greeks['theta'],
                rho=greeks['rho'],
                total_exposure=total_exposure,
                stress_scenarios=stress_results
            )
            
        except Exception as e:
            self.logger.error(f"Risk metrics calculation failed: {e}")
            return RiskMetrics()
    
    def check_risk_limits(self, risk_metrics: RiskMetrics) -> List[str]:
        """Check for risk limit breaches"""
        breaches = []
        
        if risk_metrics.var_1d > self.risk_limits['var_1d_limit']:
            breaches.append(f"1-day VaR breach: ${risk_metrics.var_1d:,.0f} > ${self.risk_limits['var_1d_limit']:,.0f}")
        
        if risk_metrics.var_10d > self.risk_limits['var_10d_limit']:
            breaches.append(f"10-day VaR breach: ${risk_metrics.var_10d:,.0f} > ${self.risk_limits['var_10d_limit']:,.0f}")
        
        if abs(risk_metrics.delta) > self.risk_limits['delta_limit']:
            breaches.append(f"Delta limit breach: {risk_metrics.delta:,.0f} > {self.risk_limits['delta_limit']:,.0f}")
        
        if abs(risk_metrics.gamma) > self.risk_limits['gamma_limit']:
            breaches.append(f"Gamma limit breach: {risk_metrics.gamma:.2f} > {self.risk_limits['gamma_limit']:.2f}")
        
        if abs(risk_metrics.vega) > self.risk_limits['vega_limit']:
            breaches.append(f"Vega limit breach: ${risk_metrics.vega:,.0f} > ${self.risk_limits['vega_limit']:,.0f}")
        
        if abs(risk_metrics.theta) > self.risk_limits['theta_limit']:
            breaches.append(f"Theta limit breach: ${risk_metrics.theta:,.0f} > ${self.risk_limits['theta_limit']:,.0f}")
        
        return breaches
    
    def generate_trading_report(self) -> str:
        """Generate comprehensive trading report"""
        try:
            # Calculate current portfolio metrics
            risk_metrics = self.calculate_risk_metrics()
            breaches = self.check_risk_limits(risk_metrics)
            
            # Portfolio summary
            total_positions = len(self.portfolio)
            total_market_value = sum(pos.market_value for pos in self.portfolio)
            total_pnl = sum(pos.unrealized_pnl for pos in self.portfolio)
            
            # Count option vs futures positions
            option_positions = sum(1 for pos in self.portfolio if pos.option_type is not None)
            futures_positions = total_positions - option_positions
            
            # Calculate additional portfolio analytics
            long_positions = sum(1 for pos in self.portfolio if pos.quantity > 0)
            short_positions = total_positions - long_positions
            
            # Calculate exposure by maturity buckets
            exposure_buckets = {'<3M': 0, '3-6M': 0, '6-12M': 0, '>12M': 0}
            for pos in self.portfolio:
                if pos.maturity <= 0.25:
                    exposure_buckets['<3M'] += abs(pos.market_value)
                elif pos.maturity <= 0.5:
                    exposure_buckets['3-6M'] += abs(pos.market_value)
                elif pos.maturity <= 1.0:
                    exposure_buckets['6-12M'] += abs(pos.market_value)
                else:
                    exposure_buckets['>12M'] += abs(pos.market_value)
            
            # Calculate maximum single position exposure
            max_position_exposure = max((abs(pos.market_value) for pos in self.portfolio), default=0)
            
            report = f"""
=== OIL TRADING PORTFOLIO REPORT ===
Generated: {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Last Model Calibration: {self.last_calibration.strftime('%Y-%m-%d %H:%M:%S') if self.last_calibration else 'Never'}

PORTFOLIO SUMMARY:
- Total Positions: {total_positions} (Futures: {futures_positions}, Options: {option_positions})
- Position Direction: Long: {long_positions}, Short: {short_positions}
- Total Market Value: ${total_market_value:,.2f}
- Unrealized P&L: ${total_pnl:,.2f}
- Largest Single Position: ${max_position_exposure:,.2f}

EXPOSURE BY MATURITY:
- < 3 Months: ${exposure_buckets['<3M']:,.2f}
- 3-6 Months: ${exposure_buckets['3-6M']:,.2f}
- 6-12 Months: ${exposure_buckets['6-12M']:,.2f}
- > 12 Months: ${exposure_buckets['>12M']:,.2f}

RISK METRICS (PyTorch Greeks):
- 1-Day VaR (95%): ${risk_metrics.var_1d:,.2f}
- 10-Day VaR (95%): ${risk_metrics.var_10d:,.2f}
- Expected Shortfall: ${risk_metrics.expected_shortfall:,.2f}
- Portfolio Delta: {risk_metrics.delta:,.2f} barrels
- Portfolio Gamma: {risk_metrics.gamma:.4f}
- Portfolio Vega: ${risk_metrics.vega:,.2f}
- Portfolio Theta: ${risk_metrics.theta:,.2f} (daily decay)
- Portfolio Rho: ${risk_metrics.rho:,.2f}

STRESS TEST RESULTS:
"""
            
            # Add stress test results
            for scenario, pnl in risk_metrics.stress_scenarios.items():
                scenario_name = scenario.replace('_', ' ').title()
                pnl_pct = (pnl / total_market_value * 100) if total_market_value != 0 else 0
                status = "🔴" if pnl < -100000 else "🟡" if pnl < -50000 else "🟢"
                report += f"- {status} {scenario_name}: ${pnl:+,.2f} ({pnl_pct:+.1f}%)\n"
            
            # Risk limit status
            if breaches:
                report += "\n⚠️  RISK LIMIT BREACHES:\n"
                for breach in breaches:
                    report += f"- 🚨 {breach}\n"
            else:
                report += "\n✅ All risk limits within bounds\n"
            
            # Risk utilization percentages
            report += f"""
RISK LIMIT UTILIZATION:
- VaR (1D): {(risk_metrics.var_1d / self.risk_limits['var_1d_limit'] * 100):.1f}%
- VaR (10D): {(risk_metrics.var_10d / self.risk_limits['var_10d_limit'] * 100):.1f}%
- Delta: {(abs(risk_metrics.delta) / self.risk_limits['delta_limit'] * 100):.1f}%
- Gamma: {(abs(risk_metrics.gamma) / self.risk_limits['gamma_limit'] * 100):.1f}%
- Vega: {(abs(risk_metrics.vega) / self.risk_limits['vega_limit'] * 100):.1f}%
- Theta: {(abs(risk_metrics.theta) / self.risk_limits['theta_limit'] * 100):.1f}%

MODEL PARAMETERS:
- Base Volatility (σ): {self.model.sigma:.3f}
- Factor 1 Mean Reversion (β₁): {self.model.beta1:.3f}
- Factor 2 Mean Reversion (β₂): {self.model.beta2:.3f}
- Factor Weight (R): {self.model.R:.3f}
- Factor Correlation (ρ): {self.model.rho:.3f}
- Risk-Free Rate: {self.risk_free_rate:.3f}

POSITION DETAILS:
"""
            
            # Add detailed position information
            for i, pos in enumerate(self.portfolio, 1):
                pos_type = "FUTURE" if pos.option_type is None else f"{pos.option_type.upper()}"
                
                # Calculate position Greeks if it's an option
                if pos.option_type is not None:
                    pos_greeks = self.calculate_option_greeks_torch(pos)
                    greeks_str = f"Δ={pos_greeks['delta']:.3f}, Γ={pos_greeks['gamma']:.4f}, ν={pos_greeks['vega']:.1f}"
                    strike_str = f" K=${pos.strike:.2f},"
                else:
                    greeks_str = "Δ=1.000"
                    strike_str = ""
                
                # Position P&L percentage
                pnl_pct = (pos.unrealized_pnl / (abs(pos.quantity * pos.entry_price)) * 100) if pos.entry_price != 0 else 0
                
                # Time to maturity
                ttm_days = int(pos.maturity * 365)
                
                report += f"""
{i:2d}. {pos.instrument} [{pos_type}]
    Qty: {pos.quantity:+,.0f},{strike_str} TTM: {ttm_days}d
    Entry: ${pos.entry_price:.2f}, Current: ${pos.current_price:.2f}
    P&L: ${pos.unrealized_pnl:+,.2f} ({pnl_pct:+.1f}%)
    Greeks: {greeks_str}"""
            
            # Market data summary
            try:
                current_curve_prices, current_curve_mats = self.market_data.get_forward_curve()
                front_month = current_curve_prices[0] if len(current_curve_prices) > 0 else 0
                curve_slope = (current_curve_prices[-1] - current_curve_prices[0]) if len(current_curve_prices) > 1 else 0
                
                report += f"""

MARKET DATA SUMMARY:
- Front Month Price: ${front_month:.2f}
- Curve Shape: ${curve_slope:+.2f} (back-front)
- Curve Points: {len(current_curve_prices)}
- Last Update: {dt.datetime.now().strftime('%H:%M:%S')}

PORTFOLIO METRICS:
- Sharpe Ratio (annualized): {self._calculate_sharpe_ratio():.2f}
- Maximum Drawdown: {self._calculate_max_drawdown():.1%}
- Position Concentration: {(max_position_exposure / total_market_value * 100):.1f}%
"""
            except Exception as e:
                self.logger.warning(f"Market data summary failed: {e}")
                report += "\nMARKET DATA SUMMARY: [Data unavailable]\n"
            
            report += f"""
TRADING RECOMMENDATIONS:
{self._generate_trading_recommendations(risk_metrics)}

=== END OF REPORT ===
"""
            
            return report
            
        except Exception as e:
            self.logger.error(f"Report generation failed: {e}")
            return f"""
=== TRADING REPORT ERROR ===
Report generation failed: {e}
Timestamp: {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Please check system logs for detailed error information.
"""
    
    def _calculate_sharpe_ratio(self) -> float:
        """Calculate portfolio Sharpe ratio (simplified)"""
        try:
            if not self.portfolio:
                return 0.0
            
            # Simplified calculation based on current P&L
            total_pnl = sum(pos.unrealized_pnl for pos in self.portfolio)
            total_investment = sum(abs(pos.quantity * pos.entry_price) for pos in self.portfolio)
            
            if total_investment == 0:
                return 0.0
            
            # Annualized return estimate
            avg_maturity = np.mean([pos.maturity for pos in self.portfolio])
            annualized_return = (total_pnl / total_investment) * (1 / max(avg_maturity, 0.1))
            
            # Estimate volatility based on portfolio VaR
            risk_metrics = self.calculate_risk_metrics()
            portfolio_vol = (risk_metrics.var_1d / total_investment) * np.sqrt(252) if total_investment > 0 else 0.3
            
            return (annualized_return - self.risk_free_rate) / max(portfolio_vol, 0.01)
            
        except:
            return 0.0
    
    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown (simplified)"""
        try:
            # Simplified: use current unrealized P&L as proxy
            pnl_values = [pos.unrealized_pnl for pos in self.portfolio]
            if not pnl_values:
                return 0.0
            
            cumulative_pnl = np.cumsum(pnl_values)
            running_max = np.maximum.accumulate(cumulative_pnl)
            drawdown = (cumulative_pnl - running_max) / np.maximum(running_max, 1)
            
            return abs(np.min(drawdown)) if len(drawdown) > 0 else 0.0
            
        except:
            return 0.0
    
    def _generate_trading_recommendations(self, risk_metrics: RiskMetrics) -> str:
        """Generate automated trading recommendations"""
        recommendations = []
        
        try:
            # Risk-based recommendations
            if risk_metrics.var_1d > self.risk_limits['var_1d_limit'] * 0.8:
                recommendations.append("⚠️ Consider reducing position sizes - approaching VaR limit")
            
            if abs(risk_metrics.delta) > self.risk_limits['delta_limit'] * 0.7:
                recommendations.append(f"🔄 Portfolio delta ({risk_metrics.delta:.0f}) is high - consider hedging")
            
            if abs(risk_metrics.gamma) > self.risk_limits['gamma_limit'] * 0.6:
                recommendations.append("📈 High gamma exposure - monitor for large price moves")
            
            if abs(risk_metrics.vega) > self.risk_limits['vega_limit'] * 0.7:
                recommendations.append("📊 High vega exposure - volatile period may impact significantly")
            
            # Position concentration check
            total_value = sum(abs(pos.market_value) for pos in self.portfolio)
            max_position = max((abs(pos.market_value) for pos in self.portfolio), default=0)
            
            if max_position / total_value > 0.3 and total_value > 0:
                recommendations.append("🎯 High position concentration - consider diversification")
            
            # Maturity analysis
            short_term_exposure = sum(abs(pos.market_value) for pos in self.portfolio if pos.maturity < 0.25)
            if short_term_exposure / total_value > 0.6 and total_value > 0:
                recommendations.append("⏰ High short-term exposure - roll positions to avoid delivery")
            
            # Greeks-based recommendations
            if risk_metrics.theta < -10000:
                recommendations.append(f"⏳ High time decay (${risk_metrics.theta:,.0f}/day) - monitor option positions")
            
            if len(recommendations) == 0:
                recommendations.append("✅ Portfolio appears well-balanced within risk parameters")
                recommendations.append("💡 Consider opportunities in volatility arbitrage or curve trading")
            
            return "\n".join(f"- {rec}" for rec in recommendations)
            
        except Exception as e:
            return f"- ❌ Recommendation engine error: {e}"
