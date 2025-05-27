from pydantic import BaseModel, Field, field_validator, computed_field, constr # type: ignore
from datetime import date
from typing import Literal, Union, Optional


class FutureObj(BaseModel):
    product_code: Optional[str]
    expiry_date: Union[str, date]
    contract_date: Union[str, date]
    calc_date: Union[str, date]
    @field_validator('expiry_date', mode = 'before')
    @classmethod
    def parse_expiry_date(cls, v):
        if isinstance(v, str):
            try:
                return date.fromisoformat(v)
            except ValueError:
                raise ValueError(f"expiry_date must if passed as string must be in YYYY-MM-DD format, received- {v}")
    @field_validator('contract_date', mode='before')
    @classmethod
    def parse_contract_date(cls, v):
        if isinstance(v,str):
            try:
                return date.fromisoformat(v)
            except ValueError:
                raise ValueError(f"contract_date must if passed as string must be in YYYY-MM-DD format, received- {v}")
    @field_validator('calc_date', mode='before')
    @classmethod
    def parse_calc_date(cls, v):
        if isinstance(v,str):
            try:
                return date.fromisoformat(v)
            except ValueError:
                raise ValueError(f"calc_date must if passed as string must be in YYYY-MM-DD format, received- {v}")

class FutureMarketData(BaseModel):
    instrument_def: FutureObj
    price: float =  Field(...,description="Price of the underlying asset")
    @computed_field
    @property
    def tte(self) -> float:
        return (self.instrument_def.expiry_date - self.instrument_def.calc_date).days / 365
    
if __name__ == "__main__":
    inst ={'product_code':'fdsfsdfsd','expiry_date':'2025-10-22', 'contract_date':'2025-10-01','calc_date':'2025-05-27'}
    obj_inst = FutureObj(**inst)
    data = {'instrument_def':obj_inst, 'price':103.4}
    future_data =  FutureMarketData(**data)
    print(f"Time to expiry: {future_data.tte}")
    print(future_data.instrument_def.expiry_date)
    print(future_data.instrument_def.calc_date)
    