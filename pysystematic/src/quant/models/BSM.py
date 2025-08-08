
import torch
import numpy as np
import polars as pl
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional, Union, Literal
from pydantic import BaseModel, Field, validator, root_validator
from dataclasses import dataclass
import time
from typing_extensions import Annotated
from src.instruments.options import OptionMarketData
import math 
from torch import nn

class BlackScholesConfig(BaseModel):
    risk_free_rate: float = Field(default=0.049, ge=-0.05, le=1, description="Risk-free interest rate") # japan had a period of negative interest rate 
    initial_vol_guess: float = Field(default=0.3, gt=0, le=2, description="Initial volatility guess for implied vol calculation")
    max_iterations: int = Field(default=100, gt=0, description="Maximum iterations for implied volatility calculation")
    precision: float = Field(default=1e-6, gt=0, description="Convergence precision for implied volatility calculation")
    device: Literal["cpu", "cuda"] = Field(
        default="cuda" if torch.cuda.is_available() else "cpu",
        description="Device to use for tensor operations"
    )

class BlackScholesTensor:
    """
    PyTorch implementation of Black-Scholes model optimized for batch processing
    with Polars for data handling and Pydantic for type validation.
    """
    
    def __init__(self, config: Optional[BlackScholesConfig] = None):
        """
        Initialize the Black-Scholes model with the specified configuration.
        
        Args:
            config: BlackScholesConfig object with model parameters
        """
        self.config = config or BlackScholesConfig()
        self.device = self.config.device
        print(f"Using device: {self.device}")
    
    def _validate_options_batch(self, options_data: List[Dict]) -> List[OptionMarketData]:
        validated_options = []
        validation_errors = []
        
        for i, option in enumerate(options_data):
            try:
                validated_option = OptionMarketData(**option)
                validated_options.append(validated_option)
            except Exception as e:
                validation_errors.append(f"Error in option {i}: {str(e)}")
        
        if validation_errors:
            error_msg = "\n".join(validation_errors[:10])
            if len(validation_errors) > 10:
                error_msg += f"\n...and {len(validation_errors) - 10} more errors"
            raise ValueError(f"Validation errors in options data:\n{error_msg}")
        
        return validated_options
    
    def normal_cdf(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the cumulative distribution function of the standard normal distribution.
        Args:
            x: Input tensor
        Returns:
            CDF values for input tensor
        """
        return 0.5 * (1 + torch.erf(x / torch.sqrt(torch.tensor(2.0, device=self.device))))
    
    def black_scholes_price(
        self, 
        S: torch.Tensor, 
        K: torch.Tensor, 
        T: torch.Tensor, 
        r: torch.Tensor, 
        sigma: torch.Tensor,
        option_type: str = "call") -> torch.Tensor:
        """
        Calculate Black-Scholes price for options.
        Args:
            S: Stock/underlying prices tensor
            K: Strike prices tensor
            T: Time to maturity tensor (in years)
            r: Risk-free rate tensor
            sigma: Volatility tensor
            option_type: Type of option ("call" or "put")
        Returns:
            Tensor of option prices
        """
        # Calculate d1 and d2
        sqrt_T = torch.sqrt(T)
        d1 = (torch.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
        d2 = d1 - sigma * sqrt_T
        
        # Calculate option price based on type
        if option_type.lower() in ["call","c","ce"]:
            option_price = S * self.normal_cdf(d1) - K * torch.exp(-r * T) * self.normal_cdf(d2)
        else:  # put
            option_price = K * torch.exp(-r * T) * self.normal_cdf(-d2) - S * self.normal_cdf(-d1)
        
        return option_price
    
    def delta(
        self, 
        S: torch.Tensor, 
        K: torch.Tensor, 
        T: torch.Tensor, 
        r: torch.Tensor, 
        sigma: torch.Tensor,
        option_type: str) -> torch.Tensor:
        """
        Calculate the delta of options.
        
        Args:
            S: Stock/underlying prices tensor
            K: Strike prices tensor
            T: Time to maturity tensor (in years)
            r: Risk-free rate tensor
            sigma: Volatility tensor
            option_type: Type of option ("call" or "put")
            
        Returns:
            Tensor of option deltas
        """
        sqrt_T = torch.sqrt(T)
        d1 = (torch.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
        
        if (option_type).lower() in ["c","ce","call"]:
            return self.normal_cdf(d1)
        else:  # put
            return self.normal_cdf(d1) - 1.0
    
    def vega(self, 
                   S: torch.Tensor, 
                   K: torch.Tensor, 
                   T: torch.Tensor, 
                   r: torch.Tensor, 
                   sigma: torch.Tensor) -> torch.Tensor:
        """
        Calculate the vega of options (same for calls and puts).
        
        Args:
            S: Stock/underlying prices tensor
            K: Strike prices tensor
            T: Time to maturity tensor (in years)
            r: Risk-free rate tensor
            sigma: Volatility tensor
            
        Returns:
            Tensor of option vegas
        """
        sqrt_T = torch.sqrt(T)
        d1 = (torch.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
        
        return S * sqrt_T * torch.exp(-0.5 * d1 ** 2) / torch.sqrt(torch.tensor(2 * np.pi, device=self.device))
    
    def gamma(self, 
                    S: torch.Tensor, 
                    K: torch.Tensor, 
                    T: torch.Tensor, 
                    r: torch.Tensor, 
                    sigma: torch.Tensor) -> torch.Tensor:
        """
        Calculate the gamma of options (same for calls and puts).
        
        Args:
            S: Stock/underlying prices tensor
            K: Strike prices tensor
            T: Time to maturity tensor (in years)
            r: Risk-free rate tensor
            sigma: Volatility tensor
            
        Returns:
            Tensor of option gammas
        """
        sqrt_T = torch.sqrt(T)
        d1 = (torch.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
        
        return torch.exp(-0.5 * d1 ** 2) / (S * sigma * sqrt_T * torch.sqrt(torch.tensor(2 * np.pi, device=self.device)))
    
    def theta(self, 
                    S: torch.Tensor, 
                    K: torch.Tensor, 
                    T: torch.Tensor, 
                    r: torch.Tensor, 
                    sigma: torch.Tensor,
                    option_type: str = "call") -> torch.Tensor:
        """
        Calculate the theta of options.
        
        Args:
            S: Stock/underlying prices tensor
            K: Strike prices tensor
            T: Time to maturity tensor (in years)
            r: Risk-free rate tensor
            sigma: Volatility tensor
            option_type: Type of option ("call" or "put")
            
        Returns:
            Tensor of option thetas (per year)
        """
        sqrt_T = torch.sqrt(T)
        d1 = (torch.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
        d2 = d1 - sigma * sqrt_T
        
        # Calculate the common part of the theta formula
        common_term = -S * sigma * torch.exp(-0.5 * d1 ** 2) / (2 * sqrt_T * torch.sqrt(torch.tensor(2 * np.pi, device=self.device)))
        
        if option_type == "call":
            return common_term - r * K * torch.exp(-r * T) * self.normal_cdf(d2)
        else:  # put
            return common_term + r * K * torch.exp(-r * T) * self.normal_cdf(-d2)
    
    def implied_volatility(self, 
                          market_price: torch.Tensor, 
                          S: torch.Tensor, 
                          K: torch.Tensor, 
                          T: torch.Tensor, 
                          r: torch.Tensor,
                          option_type: str = "call") -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate implied volatility using the Newton-Raphson method.
        
        Args:
            market_price: Market prices of the options
            S: Stock/underlying prices tensor
            K: Strike prices tensor
            T: Time to maturity tensor (in years)
            r: Risk-free rate tensor
            option_type: Type of option ("call" or "put")
            
        Returns:
            Tuple of (implied volatility tensor, boolean tensor indicating convergence)
        """
        # Initialize volatility with the initial guess
        vol = torch.ones_like(market_price, device=self.device) * self.config.initial_vol_guess
        
        # Create a mask for valid options (avoid calculations for invalid data)
        valid_mask = ((market_price > 0) & (S > 0) & (K > 0) & (T > 0))
        
        # Initialize convergence tracking
        converged = torch.zeros_like(market_price, dtype=torch.bool, device=self.device)
        
        # Newton-Raphson iterations
        for _ in range(self.config.max_iterations):
            # Skip already converged options
            active_mask = valid_mask & ~converged
            
            if not torch.any(active_mask):
                break
                
            # Compute current prices and vegas for active options
            price = self.black_scholes_price(
                S[active_mask], 
                K[active_mask], 
                T[active_mask], 
                r[active_mask], 
                vol[active_mask],
                option_type
            )
            vega = self.option_vega(
                S[active_mask], 
                K[active_mask], 
                T[active_mask], 
                r[active_mask], 
                vol[active_mask]
            )
            
            # Compute the difference between computed price and market price
            diff = price - market_price[active_mask]
            
            # Update volatility using Newton-Raphson formula
            # Avoid division by zero in the update
            vega_mask = vega > 1e-10
            if torch.any(vega_mask):
                vol_update = diff[vega_mask] / vega[vega_mask]
                vol_active = vol[active_mask]
                vol_active[vega_mask] = vol_active[vega_mask] - vol_update
                # Ensure volatility remains positive and within reasonable bounds
                vol_active = torch.clamp(vol_active, min=0.001, max=5.0)
                vol[active_mask] = vol_active
            
            # Check for convergence
            new_converged = torch.abs(diff) < self.config.precision
            converged[active_mask] = new_converged
        
        return vol, converged
    
    def process_option_batch(self, options_data: Union[pl.DataFrame, List[Dict]]) -> pl.DataFrame:
        """
        Process a batch of options data with Polars and PyTorch.
        
        Args:
            options_data: Either a Polars DataFrame or a list of dictionaries containing option data
            
        Returns:
            Polars DataFrame with original data plus implied volatilities and greeks
        """
        start_time = time.time()
        
        # Convert to Polars DataFrame if input is a list of dictionaries
        if isinstance(options_data, list):
            # Validate with Pydantic
            validated_options = self._validate_options_batch(options_data)
            # Convert to Polars DataFrame
            options_df = pl.DataFrame([opt.dict() for opt in validated_options])
        else:
            # Already a Polars DataFrame
            options_df = options_data
        
        # Ensure required columns exist
        required_cols = ["underlying_price", "strike_price", "time_to_expiry", "option_premium"]
        missing_cols = [col for col in required_cols if col not in options_df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {', '.join(missing_cols)}")
        
        # Add option_type column if it doesn't exist (default to "call")
        if "option_type" not in options_df.columns:
            options_df = options_df.with_column(pl.lit("call").alias("option_type"))
        
        # Convert option types to a list for processing
        option_types = options_df.select("option_type").to_series().to_list()
        
        # Convert DataFrame to PyTorch tensors
        S = torch.tensor(options_df.select("underlying_price").to_numpy().flatten(), 
                         dtype=torch.float32, device=self.device)
        K = torch.tensor(options_df.select("strike_price").to_numpy().flatten(), 
                         dtype=torch.float32, device=self.device)
        T = torch.tensor(options_df.select("time_to_expiry").to_numpy().flatten(), 
                         dtype=torch.float32, device=self.device)
        market_price = torch.tensor(options_df.select("option_premium").to_numpy().flatten(), 
                                    dtype=torch.float32, device=self.device)
        
        # Create risk-free rate tensor with same shape as other inputs
        r = torch.ones_like(S, device=self.device) * self.config.risk_free_rate
        
        # Process options based on their type (call or put)
        unique_option_types = set(option_types)
        
        # Initialize result tensors
        all_impl_vol = torch.zeros_like(S)
        all_converged = torch.zeros_like(S, dtype=torch.bool)
        all_delta = torch.zeros_like(S)
        all_gamma = torch.zeros_like(S)
        all_vega = torch.zeros_like(S)
        all_theta = torch.zeros_like(S)
        all_theo_price = torch.zeros_like(S)
        
        # Process each option type separately
        for opt_type in unique_option_types:
            type_mask = torch.tensor([t == opt_type for t in option_types], 
                                     dtype=torch.bool, device=self.device)
            
            if not torch.any(type_mask):
                continue
                
            # Calculate implied volatility for this option type
            impl_vol, converged = self.implied_volatility(
                market_price[type_mask], 
                S[type_mask], 
                K[type_mask], 
                T[type_mask], 
                r[type_mask],
                opt_type
            )
            
            # Calculate greeks
            delta = self.option_delta(
                S[type_mask], K[type_mask], T[type_mask], r[type_mask], impl_vol, opt_type
            )
            gamma = self.option_gamma(
                S[type_mask], K[type_mask], T[type_mask], r[type_mask], impl_vol
            )
            vega = self.option_vega(
                S[type_mask], K[type_mask], T[type_mask], r[type_mask], impl_vol
            )
            theta = self.option_theta(
                S[type_mask], K[type_mask], T[type_mask], r[type_mask], impl_vol, opt_type
            )
            
            # Calculate theoretical price
            theo_price = self.black_scholes_price(
                S[type_mask], K[type_mask], T[type_mask], r[type_mask], impl_vol, opt_type
            )
            
            # Store results
            all_impl_vol[type_mask] = impl_vol
            all_converged[type_mask] = converged
            all_delta[type_mask] = delta
            all_gamma[type_mask] = gamma
            all_vega[type_mask] = vega
            all_theta[type_mask] = theta
            all_theo_price[type_mask] = theo_price
        
        # Convert results back to numpy arrays
        np_impl_vol = all_impl_vol.cpu().numpy()
        np_converged = all_converged.cpu().numpy()
        np_delta = all_delta.cpu().numpy()
        np_gamma = all_gamma.cpu().numpy()
        np_vega = all_vega.cpu().numpy()
        np_theta = all_theta.cpu().numpy()
        np_theo_price = all_theo_price.cpu().numpy()
        
        # Add results to DataFrame
        result_df = options_df.with_columns([
            pl.Series("implied_volatility", np_impl_vol),
            pl.Series("converged", np_converged),
            pl.Series("delta", np_delta),
            pl.Series("gamma", np_gamma),
            pl.Series("vega", np_vega),
            pl.Series("theta", np_theta),
            pl.Series("theoretical_price", np_theo_price),
            pl.Series("price_difference", np_theo_price - options_df["option_premium"].to_numpy())
        ])
        
        processing_time = time.time() - start_time
        print(f"Processed {len(result_df)} options in {processing_time:.2f} seconds")
        
        return result_df
    
    def analyze_volatility_surface(self, 
                                  options_df: pl.DataFrame, 
                                  plot: bool = True) -> Optional[plt.Figure]:
        """
        Analyze and optionally plot the volatility surface from processed options data.
        
        Args:
            options_df: Polars DataFrame with processed option data including implied volatilities
            plot: Whether to create and return a plot
            
        Returns:
            Matplotlib figure if plot is True, None otherwise
        """
        if "implied_volatility" not in options_df.columns:
            raise ValueError("Options DataFrame must contain 'implied_volatility' column")
            
        # Extract key metrics for moneyness calculation
        options_df = options_df.with_column(
            (pl.col("underlying_price") / pl.col("strike_price")).alias("moneyness")
        )
        
        # Convert to pandas for plotting (Polars doesn't have built-in 3D plotting)
        pd_df = options_df.to_pandas()
        
        # Group by moneyness and time to expiry
        moneyness_bins = pd.cut(pd_df['moneyness'], bins=10)
        time_bins = pd.cut(pd_df['time_to_expiry'], bins=10)
        
        # Calculate average volatility for each moneyness/time bucket
        vol_surface = pd_df.groupby([moneyness_bins, time_bins])['implied_volatility'].mean().unstack()
        
        if plot:
            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111, projection='3d')
            
            # Create mesh grid for 3D plot
            X, Y = np.meshgrid(
                np.array([x.mid for x in vol_surface.index]), 
                np.array([y.mid for y in vol_surface.columns])
            )
            Z = vol_surface.values.T
            
            # Create the surface plot
            surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.8)
            
            # Add labels and colorbar
            ax.set_xlabel('Moneyness (S/K)')
            ax.set_ylabel('Time to Expiry (Years)')
            ax.set_zlabel('Implied Volatility')
            ax.set_title('Implied Volatility Surface')
            fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
            
            return fig
        
        return None
    
class TorchBlackScholes(nn.Module):
    """PyTorch implementation of Black-Scholes for automatic differentiation"""
    
    def __init__(self):
        super(TorchBlackScholes, self).__init__()
        
    def forward(self, F, K, T, sigma, r=0.0, option_type='call'):
        """
        Black-Scholes formula implemented in PyTorch
        F: Forward price (tensor)
        K: Strike price (tensor)
        T: Time to maturity (tensor)
        sigma: Volatility (tensor)
        r: Risk-free rate (tensor)
        option_type: 'call' or 'put'
        """
        # Ensure all inputs are tensors
        F = torch.as_tensor(F, dtype=torch.float64)
        K = torch.as_tensor(K, dtype=torch.float64)
        T = torch.as_tensor(T, dtype=torch.float64)
        sigma = torch.as_tensor(sigma, dtype=torch.float64)
        r = torch.as_tensor(r, dtype=torch.float64)
        
        # Avoid division by zero
        T = torch.clamp(T, min=1e-8)
        sigma = torch.clamp(sigma, min=1e-6)
        
        # Calculate d1 and d2
        sqrt_T = torch.sqrt(T)
        d1 = (torch.log(F / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt_T)
        d2 = d1 - sigma * sqrt_T
        
        # Normal CDF approximation for PyTorch
        def torch_norm_cdf(x):
            return 0.5 * (1 + torch.erf(x / math.sqrt(2)))
        if option_type == 'call':
            price = torch.exp(-r * T) * (F * torch_norm_cdf(d1) - K * torch_norm_cdf(d2))
        else:  # put
            price = torch.exp(-r * T) * (K * torch_norm_cdf(-d2) - F * torch_norm_cdf(-d1))
        return price
