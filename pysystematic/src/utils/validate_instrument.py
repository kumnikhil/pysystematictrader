from pydantic import BaseModel, Field
from typing import Optional, Union, List
import pandas as pd
import datetime as dt
from enum import Enum
import pytz

india_TZ = pytz.timezone('Asia/Kolkata')

class InstrumentType(str, Enum):
    FUTURE = 'FUT'
    CALL_OPTION = 'CE'
    PUT_OPTION = 'PE'
    EQUITY = 'EQ'

class Segement(str, Enum):
    BFO_FUT = "BFO-FUT"
    BFO_OPT = 'BFO-OPT'
    BSE = 'BSE'
    CDS_FUT ='CDS-FUT'
    CDS_OPT = 'CDS-OPT'
    INDICES = 'INDICES'
    MCX_FUT = 'MCX-FUT'
    MCX_OPT = 'MCX-OPT'
    NCO = 'NCO'
    NCO_FUT = 'NCO-FUT'
    NCO_OPT = 'NCO-OPT'
    NFO_FUT = 'NFO-FUT'
    NFO_OPT = 'NFO-OPT' 
    NSE = 'NSE'

class Exchange(str, Enum):
    BFO = 'BFO'
    BSE = 'BSE'
    CDS = 'CDS'
    NSE = 'NSE'
    MCX = 'MCX'
    NSEIX = 'NSEIX'
    GLOBAL = 'GLOBAL'
    NCO ='NCO'
    NFO ='NFO'


class InstrumentInfo(BaseModel):
    instrument_token: int =  Field(...)
    exchange_token: str = Field(...)
    tradingsymbol:str = Field(...)
    name:str = Field(...)
    last_price: float = Field(..., ge=0.0)
    expiry: Optional[Union[dt.date,str]] = ''
    strike: Optional[float] = None
    tick_size: float = Field(...)
    lot_size: int = Field(...)
    instrument_type:InstrumentType = Field(...)
    segment:Segement=  Field(...)
    exchange:Exchange = Field(...)
    @property
    def is_option(self) -> bool:
        """Check if instrument is an option"""
        return self.instrument_type in [InstrumentType.CALL_OPTION, InstrumentType.PUT_OPTION]
    
    @property
    def is_derivative(self) -> bool:
        """Check if instrument is a derivative"""
        return self.instrument_type in [
            InstrumentType.CALL_OPTION, 
            InstrumentType.PUT_OPTION, 
            InstrumentType.FUTURE
        ]
    
    @property
    def days_to_expiry(self) -> Optional[int]:
        """Calculate days until expiry"""
        if self.expiry:
            return (self.expiry - dt.date.today()).days
        return None
    

def create_pandas_dataframe(validated_instruments: List[InstrumentInfo]) -> pd.DataFrame:
    """
    Convert list of validated InstrumentInfo objects to Pandas DataFrame
    """
    if not validated_instruments:
        return pd.DataFrame()
    
    # Convert Pydantic objects to dictionaries
    instruments_data = [instrument.model_dump() for instrument in validated_instruments]
    
    # Create DataFrame
    df = pd.DataFrame(instruments_data)
    df['instrument_type'] = df['instrument_type'].apply(lambda x: x.value)
    df['segment'] = df['segment'].apply(lambda x: x.value)
    df['exchange'] = df['exchange'].apply(lambda x: x.value)
    df['is_derivative'] = df['instrument_type'].apply(lambda r: True if r in ['CE', 'PE', 'FUT'] else False)
    df['is_option'] = df['instrument_type'].apply(lambda r: True if r in ['CE', 'PE'] else False)
    df['days_to_expiry'] = df['expiry'].apply(lambda x: (dt.datetime.combine(x, dt.time(15,30,0), tzinfo=india_TZ) - dt.datetime.now(tz = india_TZ)).days if x !='' else None)
    return df

def validate_all_intruments(instlist: list):
    validated_instruments = []
    failed_instruments = []
    for inst in instlist:
        try:
            validated_instruments.append(InstrumentInfo(**inst))
        except:
            failed_instruments.append(inst)

    if len(failed_instruments)>0 :
        print(f"Failed the validate - {len(failed_instruments)} instruments")
    instrument_df = create_pandas_dataframe(validated_instruments)
    return instrument_df

