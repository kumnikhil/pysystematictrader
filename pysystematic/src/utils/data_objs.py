from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple
@dataclass
class Position:
    """Represents a trading position"""
    instrument: str
    quantity: float
    entry_price: float
    current_price: float = 0.0
    maturity: float = 0.0  # Time to maturity in years
    option_type: Optional[str] = None  # 'call', 'put', or None for futures
    strike: Optional[float] = None
    
    @property
    def market_value(self) -> float:
        return self.quantity * self.current_price
    
    @property
    def unrealized_pnl(self) -> float:
        return self.quantity * (self.current_price - self.entry_price)

@dataclass
class RiskMetrics:
    """Portfolio risk metrics"""
    var_1d: float = 0.0
    var_10d: float = 0.0
    expected_shortfall: float = 0.0
    delta: float = 0.0
    gamma: float = 0.0
    vega: float = 0.0
    theta: float = 0.0
    total_exposure: float = 0.0
    stress_scenarios: Dict[str, float] = field(default_factory=dict)