from pydantic import BaseModel, Field, field_validator, computed_field, constr # type: ignore
from datetime import date
from typing import Literal, Union, Optional


class OptionObj(BaseModel):
    underlying_product_code: Optional[str]
    product_code: Optional[str]
    expiry_date: Union[str, date]
    contract_date: Union[str, date]
    calc_date: Union[str, date]
    exercise_style: constr(to_upper = True) = Literal['EUROPEAN', 'AMERICAN' ]
    option_type: constr(to_upper = True) = Literal['PE', 'CE', 'CALL', 'PUT', 'C', 'P']
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

class OptionMarketData(BaseModel):
    instrument_def: OptionObj
    underlying_price: float =  Field(...,description="Price of the underlying asset")
    strike: float = Field(...,description="Strike price of the option")
    premium: float = Field(...,description="Market price of the option")
    
    implied_volatility: float =  Field(gt=0.0, description ="Implied vol of the option")
    @computed_field
    @property
    def tte(self) -> float:
        return (self.instrument_def.expiry_date - self.instrument_def.calc_date).days / 365
    @computed_field
    @property
    def option_type(self) -> str:
        return self.instrument_def.option_type
    
if __name__ == "__main__":
    inst ={'underlying_product_code':'fdsfsdfsd','product_code':'werewrew', 'expiry_date':'2025-10-22', 'contract_date':'2025-10-01', 'calc_date':'2025-05-27', 'exercise_style':'EUROPEAN', 'option_type':'c'}
    obj_inst =  OptionObj(**inst)
    data = {'instrument_def':obj_inst, 'underlying_price':103.4,'strike':100., 'premium':3.6, 'option_type':'c', 'implied_volatility':0.37}
    option_data =  OptionMarketData(**data)
    print(f"Time to expiry: {option_data.tte}")
    print(option_data.instrument_def.expiry_date)
    print(option_data.instrument_def.calc_date)
    print(option_data.instrument_def.exercise_style)
    print(option_data.instrument_def.option_type)
