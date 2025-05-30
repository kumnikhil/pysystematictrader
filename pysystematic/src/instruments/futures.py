from pydantic import BaseModel, Field, field_validator, computed_field, constr # type: ignore
import datetime as dt
import pytz
from typing import Literal, Union, Optional

india_TZ = pytz.timezone('Asia/Kolkata')

class FutureObj(BaseModel):
    product_code: Optional[str]
    expiry_date: Union[str, dt.date]
    contract_date: Union[str, dt.date]
    calc_datetime: Union[str, dt.datetime]
    @field_validator('expiry_date', mode = 'before')
    @classmethod
    def parse_expiry_date(cls, v):
        if isinstance(v, str):
            try:
                return dt.date.fromisoformat(v)
            except ValueError:
                raise ValueError(f"expiry_date must if passed as string must be in YYYY-MM-DD format, received- {v}")
    @field_validator('contract_date', mode='before')
    @classmethod
    def parse_contract_date(cls, v):
        if isinstance(v,str):
            try:
                return dt.date.fromisoformat(v)
            except ValueError:
                raise ValueError(f"contract_date must if passed as string must be in YYYY-MM-DD format, received- {v}")
    @field_validator('calc_datetime', mode='before')
    @classmethod
    def parse_calc_date(cls, v):
        if isinstance(v,str):
            try:
                return dt.datetime.fromisoformat(v)
            except ValueError:
                raise ValueError(f"calc_date must if passed as string must be in YYYY-MM-DD format, received- {v}")

class FutureMarketData(BaseModel):
    instrument_def: FutureObj
    price: float =  Field(...,description="Price of the underlying asset")
    @computed_field
    @property
    def tte(self) -> float:
        return (dt.datetime.combine(self.instrument_def.expiry_date, dt.time(15,30,0), tzinfo=india_TZ) - self.instrument_def.calc_datetime).days / 365
    
if __name__ == "__main__":
    inst ={'product_code':'fdsfsdfsd','expiry_date':'2025-10-22', 'contract_date':'2025-10-01','calc_datetime':'2025-05-27T19:00:00'}
    obj_inst = FutureObj(**inst)
    data = {'instrument_def':obj_inst, 'price':103.4}
    future_data =  FutureMarketData(**data)
    print(f"Time to expiry: {future_data.tte}")
    print(future_data.instrument_def.expiry_date)
    print(future_data.instrument_def.calc_datetime)
    

    