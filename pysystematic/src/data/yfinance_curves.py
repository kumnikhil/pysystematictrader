import datetime as dt
from typing import Union, List, Optional
from pydantic import BaseModel, validator
import yfinance as yf
from src.utils.utils_tickers import MonthRicMap, prodRIC
import pandas as pd

class yfticker(BaseModel):
    ticker: Union[List[str], str]
    @validator('ticker', pre=True)
    def convert_to_list(cls, val):
        if isinstance(val, str):
            return [val]
        return val
    
class WTI:
    def __init__(self):
        self.MonthRicMap = MonthRicMap
        self.baseRIC = yfticker(ticker='CL').ticker
        self.info = yf.Ticker(f"{self.baseRIC[0]}=F").info
        self.front_prompt_period =  self.contract_date_frm_RIC(self.info['underlyingExchangeSymbol'])

    def contract_date_2_RIC(self, contract_date:str) -> str:
        """convert contract date to Ric code compatible with yfinance"""
        if WTI._validate_date_string(contract_date):
            year, month, _ = map(int, contract_date.split('-'))
            year_int = int(str(year)[-2:])
            return f"{self.baseRIC[0]}{self.MonthRicMap[month]}{year_int}.NYM"
    def contract_date_frm_RIC(self, ric_code):
        period_code =  ric_code.replace('CL','').replace('.NYM','')
        yy =  int(period_code[-2:])
        month = [k for k,v in MonthRicMap.items() if v==period_code[0]][0]
        return (month, yy)
    @staticmethod
    def _validate_date_string(date_string:str)-> bool:
        try:
            dt.datetime.strptime(date_string, '%Y-%m-%d')
        except ValueError:
            print(f"Invalid date format or value. Expected format- YYYY-MM-DD, received value- {date_string}", flush=True)
            return False
        finally:
            return True
    @staticmethod
    def ric_expireDate(ticker:str)-> dt.datetime:
        return dt.datetime.fromtimestamp(yf.Ticker(ticker).info['expireDate']).date()
    
    def forward_curve_history(self, start_date:Optional[str] = None, end_date:Optional[str] = None,forward_maturity:int=24):
        ric_list = []
        front_month, front_yr = self.front_prompt_period
        counter = 0
        for i in range(forward_maturity):
            ric_month = front_month+i
            if ric_month>12:
                ric_yr = front_yr+ric_month//12
                ric_month = ric_month%12
                if ric_month==0:
                    counter+=1
                    ric_month=12
            else:
                ric_yr =  front_yr
            ric_list.append(f"{self.baseRIC[0]}{self.MonthRicMap[ric_month]}{ric_yr}.NYM")
        ticks =  yf.Tickers(ric_list)
        hist = ticks.history(start =  start_date, end =  end_date, interval='1d')
        return hist, ric_list
    
    def close_prices(self, hist_df:pd.Dataframe, ric_list:Optional[List[str]]=None):
        if hist_df.empty :
            print("Error: Not enough data or contracts fetched. Exiting.")
            return None
        if isinstance(ric_list, List) and len(ric_list)>0:
            hist_df = hist_df['Close'][ric_list]
        else:
            hist_df = hist_df['Close']
        hist_df = hist_df.fillna(method='ffill')
        return hist_df
