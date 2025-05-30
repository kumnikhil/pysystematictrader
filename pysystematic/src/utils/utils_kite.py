from kiteconnect import KiteConnect
import json 
import os 
import pandas as pd
import polars as pl
import datetime as dt
from pydantic import BaseModel, Field
from src.utils.validate_instrument import validate_all_intruments
from typing import List, Optional
import logging
logger = logging.getLogger(__name__)

class Kite(object):
    def __init__(self, session_file = "./appdata/api_creds/session_key.json"):
        self.kite = KiteConnect(api_key=os.getenv("API_KEY"))
        if not os.path.isfile(session_file):
            raise FileNotFoundError("Session token file for kiteconnect not found.")
        with open(session_file,'r') as f:
            self.sesion_token = json.load(f)['session_token']
        self.kite.generate_session(self.sesion_token, api_secret=os.getenv("API_SECRET"))
        self.instruments_df = validate_all_intruments(self.kite.instruments())

    def get_historical_data(self, instrument_code, start_ts, end_ts, frequency, minutes_ctr=None):
        if frequency.lower() not in ['day', 'minute']:
            raise AttributeError(f"Unknown value of the frequency- expected day / minute. Got {frequency}")
        hist_data_list = None
        if frequency=='minute' and isinstance(minutes_ctr, int):
            if minutes_ctr<5 and (pd.to_datetime(end_ts) -  pd.to_datetime(start_ts)).days>60:
                logger.info("For minute level historical data the max history allowed is 60days so changing the start_ts accordingly.")
                start_ts  = (pd.to_datetime(end_ts) - dt.timedelta(days=60)).strftime("%Y-%m-%d %H:%m:%S")
                
            try:
                hist_data_list = self.kite.historical_data(instrument_code, start_ts,end_ts,f"{minutes_ctr}minute",oi=True)
            except Exception as e:
                logger.error(e)
        elif frequency=='minute' and not isinstance(minutes_ctr, int):
            try:
                hist_data_list = self.kite.historical_data(instrument_code, start_ts,end_ts,"5minute",oi=True)
            except Exception as e:
                logger.error(e)
        elif frequency=='day':
            try:
                hist_data_list = self.kite.historical_data(instrument_code, start_ts,end_ts,"day",oi=True)
            except Exception as e:
                logger.error(e)
        else:
            raise RuntimeError("Runtime error encountered.")
        if isinstance(hist_data_list, list):
            hist_lz = pl.LazyFrame(hist_data_list).with_columns([
                pl.col("date").dt.convert_time_zone("Asia/Kolkata"), # convert to datetime
                pl.col("date").dt.date().alias("trade_date"), # extract date
                pl.col("date").dt.time().alias("time"), # extract time
                pl.lit(instrument_code).alias("instrument_token")
            ])
            return hist_lz.collect()
        else:
            return None
        
    def instruments_history(self, instrument_list:list,start_ts:str, end_ts:str, frequency:str, minutes_ctr:Optional[str]=None)->List:
        ret_list = []
        for inst_code in [code for code in instrument_list if code in self.instruments_df.instrument_token.unique().tolist()]:
            ret_list.append(self.get_historical_data(inst_code, start_ts, end_ts, frequency, minutes_ctr))
        return ret_list
    
    def get_instruments_by_type(self, instrument_type:str):
        if str(instrument_type).upper() not in self.instruments_df.instrument_type.unique().tolist():
            raise AttributeError(f"Unknown instrument_type = {instrument_type}")
        return self.instruments_df[self.instruments_df.instrument_type==str(instrument_type).upper()]

    def get_instruments_by_segment(self, segment:str):
        if str(segment).upper() not in self.instruments_df.segment.unique().tolist():
            raise AttributeError(f"Unknown segment = {segment}")
        return self.instruments_df[self.instruments_df.segment==str(segment).upper()]
    
    def __get_VIX__(self):
        return self.instruments_df[self.instruments_df.name == 'INDIA VIX']
  
        