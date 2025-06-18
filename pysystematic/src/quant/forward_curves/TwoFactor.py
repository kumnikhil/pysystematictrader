import datetime as dt
from typing import Union, List, Optional, Tuple
from pydantic import BaseModel, validator
import yfinance as yf
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.integrate import odeint
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

class TwoFactorStochasticVolatilityModel:
    """
    Implementation of the Two-Factor Forward Curve Model with Stochastic Volatility
    from Higgins (2017) - arXiv:1708.01665
    """
    
    def __init__(self, device='cpu'):
        self.device = device
        # Model parameters
        self.sigma = None      # Base volatility
        self.beta1 = None      # Mean reversion speed factor 1
        self.beta2 = None      # Mean reversion speed factor 2
        self.R = None          # Relative weight of factor 2
        self.rho = None        # Correlation between factors 1 and 2
        self.beta_v = None     # Mean reversion speed of volatility
        self.alpha = None      # Volatility of volatility
        self.rho1 = None       # Correlation between factor 1 and volatility
        self.rho2 = None       # Correlation between factor 2 and volatility
        
        # Market data
        self.forward_prices = None
        self.maturities = None
        self.times = None
        
    def set_parameters(self, sigma=0.4, beta1=0.1, beta2=1.0, R=0.5, rho=-0.3, 
                      beta_v=0.5, alpha=1.0, rho1=0.3, rho2=0.3):
        """Set model parameters"""
        self.sigma = sigma
        self.beta1 = beta1
        self.beta2 = beta2
        self.R = R
        self.rho = rho
        self.beta_v = beta_v
        self.alpha = alpha
        self.rho1 = rho1
        self.rho2 = rho2
    
    def volatility_term_structure(self, T):
        """
        Calculate the deterministic volatility term structure
        σ_F(T) = σ * sqrt(e^(-2β₁T) + R²e^(-2β₂T) + 2ρRe^(-(β₁+β₂)T))
        """
        term1 = np.exp(-2 * self.beta1 * T)
        term2 = self.R**2 * np.exp(-2 * self.beta2 * T)
        term3 = 2 * self.rho * self.R * np.exp(-(self.beta1 + self.beta2) * T)
        
        return self.sigma * np.sqrt(term1 + term2 + term3)
    
    def characteristic_function_ode(self, y, tau, theta, T, te):
        """
        ODE system for characteristic function calculation
        Returns [dA/dτ, dB/dτ]
        """
        A, B = y
        
        # Calculate σ²_F at appropriate time
        T_eff = T - (te - tau)  # Effective time to maturity
        sigma_F_sq = self.sigma**2 * (
            np.exp(-2 * self.beta1 * T_eff) + 
            self.R**2 * np.exp(-2 * self.beta2 * T_eff) + 
            2 * self.rho * self.R * np.exp(-(self.beta1 + self.beta2) * T_eff)
        )
        
        # ODE equations from the paper
        dA_dtau = self.beta_v * B
        
        dB_dtau = (-0.5 * (theta**2 + 1j*theta) * sigma_F_sq - 
                   self.beta_v * B + 
                   0.5 * self.alpha**2 * B**2 + 
                   1j * theta * B * self.alpha * self.sigma * (
                       np.exp(-self.beta1 * tau) * np.exp(-self.beta1 * (T - te)) * self.rho1 + 
                       self.R * np.exp(-self.beta2 * tau) * np.exp(-self.beta2 * (T - te)) * self.rho2
                   ))
        
        return [dA_dtau, dB_dtau]
    
    def calculate_characteristic_function(self, theta, T, te, x0=0, v0=1):
        """
        Calculate characteristic function using ODE integration
        """
        tau_span = np.linspace(0, te, 100)
        y0 = [0, 0]  # Initial conditions A(0) = B(0) = 0
        
        try:
            solution = odeint(self.characteristic_function_ode, y0, tau_span, 
                            args=(theta, T, te), rtol=1e-8)
            A_final, B_final = solution[-1]
            
            # Calculate characteristic function
            char_func = np.exp(1j * theta * x0 + A_final + B_final * v0)
            return char_func
        except:
            return np.exp(1j * theta * x0)  # Fallback
    
    def european_option_price(self, F0, K, T, te, option_type='call', num_integration_points=100):
        """
        Price European option using characteristic function method
        """
        # Integration bounds and points
        theta_max = 50
        theta_points = np.linspace(0.01, theta_max, num_integration_points)
        dtheta = theta_points[1] - theta_points[0]
        
        # Calculate integral for option pricing
        integral_sum = 0
        for theta in theta_points:
            try:
                char_func = self.calculate_characteristic_function(theta, T, te)
                integrand = np.real(char_func * np.exp(-1j * theta * np.log(K/F0)) / (theta**2 + 1j * theta))
                integral_sum += integrand * dtheta
            except:
                continue
        
        # Option price calculation (simplified version)
        if option_type == 'call':
            option_price = max(0, F0 - K/2 - K/np.pi * integral_sum)
        else:  # put
            option_price = max(0, K - F0 + K/2 + K/np.pi * integral_sum)
        
        return option_price
    
    def simulate_forward_paths(self, T_max=2.0, n_steps=100, n_paths=1000, n_maturities=12, 
                              initial_curve=None, curve_maturities=None):
        """
        OPTIMIZED Monte Carlo simulation using matrix operations and actual forward prices
        
        Args:
            T_max: Maximum simulation time (years)
            n_steps: Number of time steps
            n_paths: Number of Monte Carlo paths
            n_maturities: Number of maturity points
            initial_curve: Initial forward prices F(0,T) [n_maturities]
            curve_maturities: Corresponding maturities [n_maturities]
        """
        dt = T_max / n_steps
        sqrt_dt = np.sqrt(dt)
        
        # Time grid
        times = np.linspace(0, T_max, n_steps + 1)
        
        # Maturity grid
        maturities = np.linspace(0.1, 3.0, n_maturities)
        
        # Set up initial forward curve (convert relative to actual prices)
        if initial_curve is not None and curve_maturities is not None:
            # Interpolate market data to simulation grid
            F0 = np.interp(maturities, curve_maturities, initial_curve)
        else:
            # Default: flat curve at $75/barrel
            F0 = 75.0 * np.ones(n_maturities)
        
        # Initialize arrays
        v_paths = np.ones((n_paths, n_steps + 1))  # Volatility factor paths
        u1_paths = np.zeros((n_paths, n_steps + 1))  # Factor 1 paths
        u2_paths = np.zeros((n_paths, n_steps + 1))  # Factor 2 paths
        
        # Correlation matrix for the three Brownian motions
        corr_matrix = np.array([[1.0, self.rho, self.rho1],
                               [self.rho, 1.0, self.rho2],
                               [self.rho1, self.rho2, 1.0]])
        L = np.linalg.cholesky(corr_matrix)
        
        # Simulate paths
        for i in range(n_steps):
            # Generate correlated random numbers
            random_numbers = np.random.normal(0, 1, (n_paths, 3))
            correlated_randoms = random_numbers @ L.T
            
            dz1 = correlated_randoms[:, 0] * sqrt_dt
            dz2 = correlated_randoms[:, 1] * sqrt_dt
            dz3 = correlated_randoms[:, 2] * sqrt_dt
            
            # Update volatility factor (Heston process)
            v_drift = self.beta_v * (1 - v_paths[:, i]) * dt
            v_diffusion = self.alpha * np.sqrt(np.maximum(v_paths[:, i], 0)) * dz3
            v_paths[:, i + 1] = np.maximum(v_paths[:, i] + v_drift + v_diffusion, 0.01)
            
            # Update factors with volatility scaling
            sqrt_v = np.sqrt(v_paths[:, i])
            u1_paths[:, i + 1] = u1_paths[:, i] + sqrt_v * np.exp(self.beta1 * times[i]) * dz1
            u2_paths[:, i + 1] = u2_paths[:, i] + sqrt_v * np.exp(self.beta2 * times[i]) * dz2
        
        # ===== OPTIMIZED MATRIX-BASED FORWARD PRICE CALCULATION =====
        
        # Pre-compute exponential weight vectors (compute once, use many times)
        weight1 = np.exp(-self.beta1 * maturities)        # Factor 1 weights [n_maturities]
        weight2 = self.R * np.exp(-self.beta2 * maturities)  # Factor 2 weights [n_maturities]
        
        # Create time-maturity mask for valid forward contracts (T > t)
        Times, Maturities = np.meshgrid(times, maturities, indexing='ij')
        valid_mask = Maturities > Times  # [n_steps+1, n_maturities]
        
        # Initialize forward paths array
        forward_paths = np.zeros((n_paths, n_steps + 1, n_maturities))
        
        # VECTORIZED CALCULATION using broadcasting
        # Shape: [n_paths, n_steps+1, 1] × [1, 1, n_maturities] = [n_paths, n_steps+1, n_maturities]
        u1_broadcast = u1_paths[:, :, None]  # [n_paths, n_steps+1, 1]
        u2_broadcast = u2_paths[:, :, None]  # [n_paths, n_steps+1, 1]
        
        # Broadcast weights to match factor dimensions
        weight1_broadcast = weight1[None, None, :]  # [1, 1, n_maturities]
        weight2_broadcast = weight2[None, None, :]  # [1, 1, n_maturities]
        
        # Calculate log-forward price changes for ALL paths, times, maturities at once
        x_all = self.sigma * (u1_broadcast * weight1_broadcast + 
                             u2_broadcast * weight2_broadcast)  # [n_paths, n_steps+1, n_maturities]
        
        # Convert to actual forward prices: F(t,T) = F(0,T) * exp(x(t,T))
        F0_broadcast = F0[None, None, :]  # [1, 1, n_maturities]
        forward_prices_raw = F0_broadcast * np.exp(x_all)  # [n_paths, n_steps+1, n_maturities]
        
        # Apply expiration logic: set expired contracts to spot price
        for i in range(n_steps + 1):
            for j in range(n_maturities):
                if valid_mask[i, j]:  # Valid forward contract (T > t)
                    forward_paths[:, i, j] = forward_prices_raw[:, i, j]
                else:  # Expired contract (T <= t)
                    # Use shortest available maturity as spot price proxy
                    available_contracts = valid_mask[i, :]
                    if np.any(available_contracts):
                        # Find first available contract
                        first_available = np.where(available_contracts)[0][0]
                        forward_paths[:, i, j] = forward_prices_raw[:, i, first_available]
                    else:
                        # All contracts expired, use last known price
                        forward_paths[:, i, j] = forward_paths[:, max(0, i-1), j]
        
        return times, maturities, forward_paths, v_paths
    
    def fit_to_market_data(self, forward_prices, maturities, learning_rate=0.01, n_epochs=1000):
        """
        Calibrate model parameters to market data using gradient descent
        """
        # Convert to tensors
        forward_tensor = torch.tensor(forward_prices, dtype=torch.float32, device=self.device)
        maturity_tensor = torch.tensor(maturities, dtype=torch.float32, device=self.device)
        
        # Initialize parameters as tensors requiring gradients
        params = {
            'sigma': torch.tensor(0.4, requires_grad=True, device=self.device),
            'beta1': torch.tensor(0.1, requires_grad=True, device=self.device),
            'beta2': torch.tensor(1.0, requires_grad=True, device=self.device),
            'R': torch.tensor(0.5, requires_grad=True, device=self.device),
            'rho': torch.tensor(-0.3, requires_grad=True, device=self.device),
        }
        
        optimizer = optim.Adam(params.values(), lr=learning_rate)
        
        losses = []
        
        for epoch in range(n_epochs):
            optimizer.zero_grad()
            
            # Calculate model volatilities
            model_vols = []
            for T in maturities:
                T_tensor = torch.tensor(T, device=self.device)
                vol = params['sigma'] * torch.sqrt(
                    torch.exp(-2 * params['beta1'] * T_tensor) + 
                    params['R']**2 * torch.exp(-2 * params['beta2'] * T_tensor) + 
                    2 * params['rho'] * params['R'] * torch.exp(-(params['beta1'] + params['beta2']) * T_tensor)
                )
                model_vols.append(vol)
            
            model_vols_tensor = torch.stack(model_vols)
            
            # Calculate market implied volatilities (simplified)
            market_vols = torch.std(torch.diff(torch.log(forward_tensor), dim=0), dim=0) * np.sqrt(252)
            
            # Loss function
            loss = torch.mean((model_vols_tensor - market_vols)**2)
            
            loss.backward()
            optimizer.step()
            
            # Apply constraints
            with torch.no_grad():
                params['sigma'].clamp_(0.01, 2.0)
                params['beta1'].clamp_(0.01, 5.0)
                params['beta2'].clamp_(0.01, 5.0)
                params['R'].clamp_(0.01, 2.0)
                params['rho'].clamp_(-0.99, 0.99)
            
            losses.append(loss.item())
            
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss.item():.6f}")
        
        # Update model parameters
        self.sigma = params['sigma'].item()
        self.beta1 = params['beta1'].item()
        self.beta2 = params['beta2'].item()
        self.R = params['R'].item()
        self.rho = params['rho'].item()
        
        return losses
