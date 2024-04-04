# enums.py

from enum import Enum, auto

class OrderType(Enum):
    Limit = auto()  # Make sure this line is included
    # Translating C++ enum OrderType values to Python
    GoodTillCancel = auto()
    FillAndKill = auto()
    FillOrKill = auto()
    GoodForDay = auto()
    Market = auto()

class Side(Enum):
    # Translating C++ enum Side values to Python
    Buy = auto()
    Sell = auto()
