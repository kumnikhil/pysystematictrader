from pydantic import BaseModel
from typing import Dict
from dataclasses import dataclass

MonthRicMap = {
    1: 'F',
    2: 'G',
    3: 'H',
    4: 'J',
    5: 'K',
    6: 'M',
    7: 'N',
    8: 'Q',
    9: 'U',
    10: 'V',
    11: 'X',
    12: 'Z'}

prodRIC = {
    'WTI':"CL"
}