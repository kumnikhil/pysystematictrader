# src/quant/volatility/BreedenLitzenberger.py
import os 
import pandas as pd
import polars as pl
import numpy as np
import datetime as dt
from pydantic import BaseModel, Field
from scipy import integrate
from typing import Callable, Tuple
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from src.utils.utils_kite import Kite
from src.quant.models.vanilla import BlackScholesMerton
import src.utils.smoothen as smooth
from dotenv import load_dotenv, find_dotenv
from tqdm import tqdm 
import time
def _get_api_creds():
    """
    Load API credentials from environment variables.
    """
    try:
        load_dotenv(find_dotenv())
        os.environ['API_KEY'] = os.getenv('API_KEY')
        os.environ['API_SECRET'] = os.getenv('API_SECRET')
    except Exception as e:
        print(f"Error loading API credentials: {e}")
        raise

class RiskNetutralDensity(object):
    """
    Risk Neutral Density (RND) model for options pricing.
    """
    def __init__(self, exchange_code:str, instrument_name:str, expiry:dt.datetime,rf_rate:float=0.05,calc_date:dt.datetime=None, time_frame:tuple=None, strikes_flt_method:str='puts'):
        self.exchange_code = exchange_code
        self.instrument_name = instrument_name
        self.expiry = expiry
        self.rf_rate = rf_rate
        self.calc_date = calc_date if calc_date else dt.datetime.today().date()
        self.time_frame = time_frame
        self.strikes_flt_method = strikes_flt_method

    def generate_vol_smile(self, kite_obj: Kite) -> Tuple[float, pd.DataFrame, Callable]:
        """
        Initialize and return a KiteConnect object.
        
        Args:
            kite_obj (KiteConnect): An existing KiteConnect object, if available.
            exhange_code (str): The exchange code for the instrument.
            instrument_name (str): The name of the instrument.
            expiry (dt.datetime): The expiry date of the instrument.
            time_frame (tuple, optional): A tuple representing the time frame for the instrument. Defaults None, in which fit the lastest data using quotes.
            
        Returns:
            KiteConnect: A KiteConnect object initialized with API credentials.
        """
        if self.exchange_code.upper() not in kite_obj.instruments_df.exchange.unique().tolist():
            raise ValueError(f"Exchange code {self.exchange_code} not found in instruments list.")
        if self.instrument_name.upper() not in kite_obj.instruments_df[kite_obj.instruments_df.exchange == self.exchange_code].name.unique().tolist():
            raise ValueError(f"Instrument name {self.instrument_name} not found in instruments list for exchange code {self.exchange_code}.")
        fno_df = kite_obj.instruments_df[(kite_obj.instruments_df.exchange==self.exchange_code.upper()) & (kite_obj.instruments_df.name == self.instrument_name.upper())]
        fno_df['expiry'] = pd.to_datetime(fno_df['expiry'])
        fno_df.sort_values(by = ['expiry', 'strike','instrument_type'], inplace=True)
        if self.expiry not in fno_df.expiry.unique():
            print(f"Expiry date {self.expiry} not found in instruments list for exchange code {self.exchange_code} and instrument name {self.instrument_name}.")
            target_expiry = pd.to_datetime(self.expiry)
            closest_idx = np.argmin(np.abs(fno_df['expiry'] - target_expiry))
            closest_date = fno_df.loc[closest_idx, 'expiry']
            print(f"Using closest expiry date: {closest_date}")
            self.expiry = closest_date
        
        # find the closet date to the expiry in the kite_obj.instruments_df,expiry column and use that as the expiry date
        exp_instrument_ = fno_df[fno_df.expiry == self.expiry].drop_duplicates(subset=['expiry','strike','instrument_type'])
        if exp_instrument_.empty:
            raise ValueError(f"No instruments found for expiry date {self.expiry} in exchange code {self.exchange_code} and instrument name {self.instrument_name}.")   
        fut_tkn = str(exp_instrument_[exp_instrument_.instrument_type=='FUT'].iloc[0]['instrument_token'])
        spot_quote = kite_obj.kite.quote([fut_tkn])
        self.spot_price =spot_quote[fut_tkn]['last_price']
        exp_instrument_['spot'] = spot_quote[fut_tkn]['last_price']
        exp_instrument_['moneyness'] = exp_instrument_['strike']/spot_quote[fut_tkn]['last_price']

        exp_options = exp_instrument_[exp_instrument_.is_option]

        quotes_price = kite_obj.kite.quote(exp_options['instrument_token'].values.tolist())
        exp_options['close'] = exp_options['instrument_token'].apply(lambda r: quotes_price[str(r)]['last_price'])
        exp_options['volume'] = exp_options['instrument_token'].apply(lambda r: quotes_price[str(r)]['volume'])
        exp_options['oi'] = exp_options['instrument_token'].apply(lambda r: quotes_price[str(r)]['oi'])
        exp_options['intrinsic_val'] = exp_options.apply(lambda r: max(0.0, r['spot']-r['strike']),axis=1)
        exp_options['time_val'] = exp_options.apply(lambda r: max(0.0, r['close']-r['intrinsic_val']),axis=1)
            
        calls_df = exp_options[exp_options.instrument_type=='CE']
        puts_df = exp_options[exp_options.instrument_type=='PE']
        calls_df['fitted_price'],_ = smooth.fit_montonic_prices(calls_df['strike'].values, calls_df['close'].values,'CE')
        puts_df['fitted_price'],_ = smooth.fit_montonic_prices(puts_df['strike'].values, puts_df['close'].values,'PE')
        options_df = pd.concat([calls_df, puts_df]).sort_values(by = ['strike'])
        bsm =  BlackScholesMerton()
        if self.strikes_flt_method == 'liquidity':
            flt_opts = options_df.sort_values(by = ['strike', 'oi', 'volume'],ascending=[True, False, False]).drop_duplicates(subset=['strike'])
        elif self.strikes_flt_method == "otm":
            otm_puts = options_df[(options_df.strike<self.spot_price) & (options_df.instrument_type=='PE')]
            otm_calls = options_df[(options_df.strike>=self.spot_price) & (options_df.instrument_type=='CE')]
            flt_opts = pd.concat([otm_puts, otm_calls]).sort_values(by = ['strike'],ascending=[True]).drop_duplicates(subset=['strike'])
        elif self.strikes_flt_method == "itm":
            itm_calls = options_df[(options_df.strike<self.spot_price) & (options_df.instrument_type=='CE')]
            itm_puts = options_df[(options_df.strike>=self.spot_price) & (options_df.instrument_type=='PE')]
            flt_opts = pd.concat([itm_calls, itm_puts]).sort_values(by = ['strike'],ascending=[True]).drop_duplicates(subset=['strike'])
        elif self.strikes_flt_method == "calls":
            flt_opts = options_df[(options_df.instrument_type=='CE')].sort_values(by = ['strike'],ascending=[True]).drop_duplicates(subset=['strike'])
        elif self.strikes_flt_method == "puts":
            flt_opts = options_df[(options_df.instrument_type=='PE')].sort_values(by = ['strike'],ascending=[True]).drop_duplicates(subset=['strike'])
        else:
            NotImplementedError(f"strikes to fit implied vol allowed values- [liquidity, otm, itm, calls, puts]. Received- {self.strikes_flt_method}")
        flt_opts['implied_vol'] = flt_opts.apply(
        lambda r: bsm.implied_volatility(
            r['close'],
            r['spot'],
            r['strike'], 
            (r['expiry'] - self.calc_date).days/365.,
            self.rf_rate,
            r['instrument_type']), 
        axis=1)

        flt_opts['fitted_iv'],spline_IV = smooth.get_smoothened_vol_curve(strikes=flt_opts['strike'].values, implied_vols=flt_opts['implied_vol'].values,return_spline=True)
        return (flt_opts, spline_IV)
    
    def get_risk_neutral_density(self, flt_opts:pd.DataFrame, spline_IV:Callable) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        strike_grd = np.linspace(flt_opts.strike.min(), flt_opts.strike.max(), 500)
        IV  = spline_IV(strike_grd)
        calls_grd = BlackScholesMerton.black_scholes_call(
            self.spot_price*np.ones_like(IV), 
            strike_grd, 
            np.ones_like(IV)*(flt_opts['expiry'].unique().tolist()[0] - self.calc_date).days/365.,
            self.rf_rate*np.ones_like(IV), 
            IV)
        h = strike_grd[1] - strike_grd[0]
        density = np.zeros_like(strike_grd)
        for i in range(1, len(strike_grd) - 1):
            density[i] = (calls_grd[i+1] - 2*calls_grd[i] + calls_grd[i-1]) / (h**2)
            
            # Forward/backward differences for boundary points
        density[0] = density[1]
        density[-1] = density[-2]

        discount  = np.exp(-self.rf_rate * (flt_opts['expiry'].unique().tolist()[0] - self.calc_date).days/365.)
        density  = density* discount
        density = np.maximum(density, 0.0)
        total_prob = integrate.trapezoid(density, strike_grd)
        if total_prob > 0:
            density = density / total_prob
        else:
            print(total_prob)

        cdf = np.zeros_like(density)
            
        # Method 1: Using cumulative trapezoidal rule
        dx = strike_grd[1] - strike_grd[0]  # Assuming uniform spacing
        cdf[0] = 0

        for i in range(1, len(density)):
            cdf[i] = cdf[i-1] + 0.5 * (density[i] + density[i-1]) * dx
        return (strike_grd, density, cdf)
    
    def plot_cdf(self, strike_grd, cdf):
        plt.figure(figsize=(12, 8))
        # Main CDF plot
        plt.plot(strike_grd, cdf, 'b-', linewidth=2.5, label='Risk-Neutral CDF')
        plt.fill_between(strike_grd, cdf, alpha=0.3, color='lightblue')

        # Add reference lines if provided
        prob_at_spot = np.interp(self.spot_price, strike_grd, cdf)
        plt.axvline(self.spot_price, color='red', linestyle='--', linewidth=2, 
                alpha=0.8, label=f'Spot Price ({self.spot_price:,.0f})')
        plt.axhline(prob_at_spot, color='red', linestyle=':', alpha=0.6)
        plt.text(self.spot_price, prob_at_spot + 0.05, 
                f'P(S_T ≤ {self.spot_price:,.0f}) = {prob_at_spot:.1%}', 
                ha='center', va='bottom', fontsize=10, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
        # Add percentile lines
        percentiles = [0.05, 0.25, 0.5, 0.75, 0.95]
        colors = ['purple', 'orange', 'black', 'orange', 'purple']
        for p, color in zip(percentiles, colors):
            # Find strike corresponding to percentile
            strike_at_percentile = np.interp(p, cdf, strike_grd)
            plt.axhline(p, color=color, linestyle=':', alpha=0.5, linewidth=1)
            plt.axvline(strike_at_percentile, color=color, linestyle=':', alpha=0.5, linewidth=1)
            # Add label
            if p == 0.5:  # Median
                plt.text(strike_at_percentile, 0.02, f'Median\n₹{strike_at_percentile:,.0f}', 
                        ha='center', va='bottom', fontsize=9, color=color, weight='bold')
            # elif p in [0.05, 0.95]:  # Tails
            else:
                plt.text(strike_at_percentile, p + 0.02, f'{p:.0%}\n₹{strike_at_percentile:,.0f}', 
                        ha='center', va='bottom', fontsize=8, color=color)
        # Formatting
        plt.xlabel('Underlying Price at Expiry (₹)', fontsize=12)
        plt.ylabel('Cumulative Probability', fontsize=12)
        plt.title('Risk-Neutral Cumulative Distribution Function', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=10)
        # Set axis limits
        plt.xlim(strike_grd.min(), strike_grd.max())
        plt.ylim(0, 1)
        plt.show()
        return None
    def plot_cdf_with_bins(self,strikes_dense, density, num_bins=10, bin_method='equal_width'):
        """
        Plot cumulative density function with probability bins overlay
        
        Args:
            strikes_dense: Array of strike prices
            density: Risk-neutral probability density values
            spot_price: Current spot price (optional, for reference line)
            forward_price: Forward price (optional, for reference line)
            num_bins: Number of bins to create
            bin_method: 'equal_width', 'equal_prob', or 'custom'
        """
        
        # Calculate CDF
        dx = strikes_dense[1] - strikes_dense[0]
        cdf = np.zeros_like(density)
        cdf[0] = 0
        for i in range(1, len(density)):
            cdf[i] = cdf[i-1] + 0.5 * (density[i] + density[i-1]) * dx
        
        # Normalize CDF
        if cdf[-1] > 0:
            cdf = cdf / cdf[-1]
        
        # Create bins based on method
        if bin_method == 'equal_width':
            bin_edges = np.linspace(strikes_dense.min(), strikes_dense.max(), num_bins + 1)
        elif bin_method == 'equal_prob':
            # Create bins with equal probability mass
            prob_points = np.linspace(0, 1, num_bins + 1)
            bin_edges = np.interp(prob_points, cdf, strikes_dense)
        else:  # custom bins around key levels
            # Create bins around spot price
            range_factor = 0.3  # ±30% range
            bin_edges = np.linspace(self.spot_price * (1 - range_factor), self.spot_price * (1 + range_factor), num_bins + 1)
            
        # Calculate bin probabilities
        bin_probs = []
        bin_centers = []
        
        for i in range(len(bin_edges) - 1):
            left_edge = bin_edges[i]
            right_edge = bin_edges[i + 1]
            
            # Find probabilities at bin edges
            left_prob = np.interp(left_edge, strikes_dense, cdf)
            right_prob = np.interp(right_edge, strikes_dense, cdf)
            
            bin_prob = right_prob - left_prob
            bin_probs.append(bin_prob)
            bin_centers.append((left_edge + right_edge) / 2)
        
        # Create subplot layout
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12), 
                                    gridspec_kw={'height_ratios': [2, 1]})
        
        # ============ TOP PLOT: CDF with bin overlays ============
        
        # Plot main CDF
        ax1.plot(strikes_dense, cdf, 'b-', linewidth=3, label='Risk-Neutral CDF', zorder=5)
        ax1.fill_between(strikes_dense, cdf, alpha=0.2, color='lightblue', zorder=1)
        
        # Color map for bins
        colors = plt.cm.Set3(np.linspace(0, 1, len(bin_probs)))
        
        # Add bin overlays on CDF
        for i, (left, right, prob, color) in enumerate(zip(bin_edges[:-1], bin_edges[1:], bin_probs, colors)):
            # Find y-coordinates for the bin on CDF
            left_y = np.interp(left, strikes_dense, cdf)
            right_y = np.interp(right, strikes_dense, cdf)
            
            # Create mask for this bin region
            mask = (strikes_dense >= left) & (strikes_dense <= right)
            if mask.any():
                # Highlight the CDF segment for this bin
                ax1.fill_between(strikes_dense[mask], 0, cdf[mask], 
                            alpha=0.6, color=color, 
                            label=f'Bin {i+1}: {prob:.1%}', zorder=3)
                
                # Add bin boundary lines
                ax1.axvline(left, color='black', linestyle=':', alpha=0.7, linewidth=1)
                ax1.axvline(right, color='black', linestyle=':', alpha=0.7, linewidth=1)
                
                # Add probability labels
                mid_x = (left + right) / 2
                mid_y = np.interp(mid_x, strikes_dense, cdf)
                
                ax1.text(mid_x, mid_y + 0.03, f'{prob:.1%}', 
                        ha='center', va='bottom', fontweight='bold', 
                        fontsize=10, color='darkred',
                        bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))
        
        # Add reference lines
        
        prob_at_spot = np.interp(self.spot_price, strikes_dense, cdf)
        ax1.axvline(self.spot_price, color='red', linestyle='--', linewidth=2, 
                alpha=0.8, label=f'Spot ({self.spot_price:,.0f})', zorder=6)
        ax1.text(self.spot_price, prob_at_spot + 0.05, 
                f'P(S≤Spot) = {prob_at_spot:.1%}', 
                ha='center', va='bottom', fontsize=10, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.8))
        
        
        # Format top plot
        ax1.set_xlabel('Underlying Price at Expiry (₹)', fontsize=12)
        ax1.set_ylabel('Cumulative Probability', fontsize=12)
        ax1.set_title('Risk-Neutral CDF with Probability Bins', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
        ax1.set_xlim(strikes_dense.min(), strikes_dense.max())
        ax1.set_ylim(0, 1.05)
        
        # ============ BOTTOM PLOT: Bin probabilities bar chart ============
        
        bars = ax2.bar(range(len(bin_probs)), np.array(bin_probs) * 100, 
                    color=colors, alpha=0.7, edgecolor='black', linewidth=1)
        
        # Add value labels on bars
        for i, (bar, prob) in enumerate(zip(bars, bin_probs)):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{prob:.1%}', ha='center', va='bottom', fontweight='bold', fontsize=10)
        
        # Customize bar chart
        bin_labels = []
        for i, (left, right) in enumerate(zip(bin_edges[:-1], bin_edges[1:])):
            if left >= 1000:
                label = f'₹{left/1000:.0f}K-{right/1000:.0f}K'
            else:
                label = f'₹{left:.0f}-{right:.0f}'
            bin_labels.append(label)
        
        ax2.set_xlabel('Price Bins', fontsize=12)
        ax2.set_ylabel('Probability (%)', fontsize=12)
        ax2.set_title('Probability Distribution by Bins', fontsize=12, fontweight='bold')
        ax2.set_xticks(range(len(bin_labels)))
        ax2.set_xticklabels(bin_labels, rotation=45, ha='right', fontsize=9)
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Add statistics box
        mean = np.trapz(strikes_dense * density, strikes_dense)
        std = np.sqrt(np.trapz((strikes_dense - mean)**2 * density, strikes_dense))
        
        stats_text = f"""Statistics:
    Mean: ₹{mean:,.0f}
    Std: ₹{std:,.0f}
    Bins: {len(bin_probs)}
    Method: {bin_method}"""
        
        ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes,
                verticalalignment='top', fontsize=10, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        plt.show()
        
        # Return bin information
        bin_info = pd.DataFrame({
            'bin_number': range(1, len(bin_probs) + 1),
            'left_edge': bin_edges[:-1],
            'right_edge': bin_edges[1:],
            'bin_center': bin_centers,
            'probability': bin_probs,
            'probability_pct': np.array(bin_probs) * 100
        })
        return cdf, bin_info

# Alternative: Focused probability bins around key levels
    def plot_cdf_key_levels(self,strikes_dense, density, spot_price, moneyness_levels=None):
        """
        Plot CDF with bins focused on key moneyness levels
        """
        if moneyness_levels is None:
            # Default key levels (moneyness: K/S)
            moneyness_levels = [0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15]
        
        # Convert moneyness to strike levels
        strike_levels = [spot_price * m for m in moneyness_levels]
        
        # Calculate CDF
        dx = strikes_dense[1] - strikes_dense[0]
        cdf = np.cumsum(density) * dx
        cdf = cdf / cdf[-1]
        
        # Calculate probabilities between key levels
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        
        # Plot CDF
        ax1.plot(strikes_dense, cdf, 'b-', linewidth=3, label='Risk-Neutral CDF')
        
        # Color different regions
        colors = ['red', 'orange', 'yellow', 'lightgreen', 'green', 'cyan', 'blue', 'purple']
        
        level_probs = []
        level_labels = []
        
        for i in range(len(strike_levels) - 1):
            left_strike = strike_levels[i]
            right_strike = strike_levels[i + 1]
            
            # Calculate probability in this range
            left_prob = np.interp(left_strike, strikes_dense, cdf)
            right_prob = np.interp(right_strike, strikes_dense, cdf)
            range_prob = right_prob - left_prob
            
            level_probs.append(range_prob)
            
            # Create label with moneyness
            left_moneyness = moneyness_levels[i]
            right_moneyness = moneyness_levels[i + 1]
            level_labels.append(f'{left_moneyness:.2f}-{right_moneyness:.2f}')
            
            # Highlight region
            mask = (strikes_dense >= left_strike) & (strikes_dense <= right_strike)
            if mask.any():
                ax1.fill_between(strikes_dense[mask], 0, cdf[mask], 
                            alpha=0.6, color=colors[i % len(colors)], 
                            label=f'{left_moneyness:.2f}-{right_moneyness:.2f}: {range_prob:.1%}')
            
            # Add vertical lines
            ax1.axvline(left_strike, color='black', linestyle=':', alpha=0.7)
            ax1.axvline(right_strike, color='black', linestyle=':', alpha=0.7)
        
        # Add spot price line
        ax1.axvline(spot_price, color='red', linestyle='--', linewidth=3, 
                alpha=0.8, label=f'Spot (₹{spot_price:,.0f})')
        
        ax1.set_xlabel('Strike Price (₹)')
        ax1.set_ylabel('Cumulative Probability')
        ax1.set_title('CDF with Key Moneyness Level Probabilities')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
        ax1.grid(True, alpha=0.3)
        
        # Bar chart of probabilities by moneyness range
        bars = ax2.bar(range(len(level_probs)), np.array(level_probs) * 100, 
                    color=colors[:len(level_probs)], alpha=0.7, edgecolor='black')
        
        # Add labels
        for bar, prob in zip(bars, level_probs):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{prob:.1%}', ha='center', va='bottom', fontweight='bold')
        
        ax2.set_xlabel('Moneyness Range (K/S)')
        ax2.set_ylabel('Probability (%)')
        ax2.set_title('Probability by Moneyness Levels')
        ax2.set_xticks(range(len(level_labels)))
        ax2.set_xticklabels(level_labels, rotation=45)
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.show()
        
        return level_probs, level_labels

    # Example usage:
    """
    # Assuming you have density and strikes_dense from risk-neutral estimation
    cdf_values, bin_info = plot_cdf_with_bins(
        strikes_dense, density, 
        spot_price=18500, 
        forward_price=18580,
        num_bins=8, 
        bin_method='equal_width'
    )

    print("Bin Information:")
    print(bin_info)

    # Alternative with key moneyness levels
    level_probs, level_labels = plot_cdf_key_levels(
        strikes_dense, density, spot_price=18500
    )
    """
    
if __name__ == "__main__":
    start_time = time.time()
    _get_api_creds()
    kite_obj = Kite()
    RND = RiskNetutralDensity(
        exchange_code='NFO',    
        instrument_name='NIFTY',
        expiry=dt.datetime(2025, 8, 28),
        rf_rate=0.05,
        calc_date=dt.datetime(2024, 8, 8),
        strikes_flt_method='puts'
    )
    flt_opts, spline_IV = RND.generate_vol_smile(kite_obj)
    strike_grd, density, cdf = RND.get_risk_neutral_density(flt_opts, spline_IV)
    # RND.plot_cdf(strike_grd, cdf)
    RND.plot_cdf_with_bins(strike_grd, density,  num_bins=10, bin_method='equal_width')
    end_time = time.time()
    print(f"Execution time: {end_time - start_time:.2f} seconds")