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
    def __init__(self, exchange_code:str, instrument_name:str, expiry:dt.datetime,rf_rate:float=0.05,calc_date:dt.datetime=None, time_frame:tuple=None, strikes_flt_method:str='liquidity'):
        self.exchange_code = exchange_code
        self.instrument_name = instrument_name
        self.expiry = expiry
        self.rf_rate = rf_rate
        self.calc_date = calc_date if calc_date else dt.datetime.today().date()
        self.time_frame = time_frame
        self.strikes_flt_method = strikes_flt_method
        _get_api_creds()

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
        if self.exchange_code.upper() not in kite_obj.instruments.unique().tolist():
            raise ValueError(f"Exchange code {self.exchange_code} not found in instruments list.")
        if self.instrument_name.upper() not in kite_obj.instruments[kite_obj.instruments.exchange == self.exchange_code].instrument_name.unique().tolist():
            raise ValueError(f"Instrument name {self.instrument_name} not found in instruments list for exchange code {self.exchange_code}.")
        fno_df = kite_obj.instruments_df[(kite_obj.instruments_df.exchange==self.exchange_code.upper()) & (kite_obj.instruments_df.name == self.instrument_name.upper())]
        fno_df.sort_values(by = ['expiry', 'strike','instrument_type'], inplace=True)
        if self.expiry not in fno_df.expiry.unique():
            print(f"Expiry date {self.expiry} not found in instruments list for exchange code {self.exchange_code} and instrument name {self.instrument_name}.")
            closest_idx = np.argmin(np.abs(fno_df['expiry'] -self.expiry))
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
        flt_opts['implied_vol'] = flt_opts.progress_apply(
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
    
if __name__ == "__main__":
    start_time = time.time()
    kite_obj = Kite()
    RND = RiskNetutralDensity(
        exchange_code='NFO',    
        instrument_name='NIFTY',
        expiry=dt.datetime(2025, 8, 28),
        rf_rate=0.05,
        calc_date=dt.datetime(2024, 8, 8),
        strikes_flt_method='liquidity'
    )
    flt_opts, spline_IV = RND.generate_vol_smile(kite_obj)
    strike_grd, density, cdf = RND.get_risk_neutral_density(flt_opts, spline_IV)
    RND.plot_cdf(strike_grd, cdf)
    end_time = time.time()
    print(f"Execution time: {end_time - start_time:.2f} seconds")
