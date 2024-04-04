# models.py

from dataclasses import dataclass
from enums import OrderType, Side

@dataclass
class Order:
    order_type: OrderType
    order_id: int
    side: Side
    price: float
    quantity: int

    def __post_init__(self):
        # Additional initialization or validation can be added here
        if self.price < 0:
            raise ValueError("Price cannot be negative")
        if self.quantity <= 0:
            raise ValueError("Quantity must be positive")


@dataclass
class TradeInfo:
    order_id: int
    price: float
    quantity: int


@dataclass
class Trade:
    bid_trade: TradeInfo
    ask_trade: TradeInfo


@dataclass
class LevelInfo:
    price: float
    quantity: int
