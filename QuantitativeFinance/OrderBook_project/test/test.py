# %%
# constants.py

# Importing necessary libraries for mathematical operations and limits
import math

# Defining constants
class Constants:
    # In C++, InvalidPrice is set using std::numeric_limits<Price>::quiet_NaN();
    # In Python, we can use math.nan to represent a floating-point "Not a Number"
    InvalidPrice = math.nan
    
    # If there were other constants defined across the C++ project related to price, quantity, or IDs, 
    # they could be similarly translated here. As an example, setting a default max order quantity:
    MaxOrderQuantity = 1_000_000

    # Assuming there might be constants for order book matching precision or minimum price movement (tick size):
    PriceTickSize = 0.01  # Minimum price movement
    MatchingPrecision = 2  # Number of decimal places for price matching

    # Any other relevant constants used throughout the project would also be declared here,
    # such as default values for timeouts, intervals, or specific flags used in the system.


# %%
# enums.py

from enum import Enum, auto

class OrderType(Enum):
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


# %%
# models.py

from dataclasses import dataclass
# from enums import OrderType, Side

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


# %%
# order_book.py

# from models import Order, Trade, TradeInfo
# from enums import Side
from typing import List, Dict, Optional
from collections import defaultdict

class OrderBook:
    def __init__(self):
        self.bids: Dict[float, List[Order]] = defaultdict(list)  # Price to list of buy orders
        self.asks: Dict[float, List[Order]] = defaultdict(list)  # Price to list of sell orders
        self.trades: List[Trade] = []

    def add_order(self, order: Order) -> Optional[Trade]:
        if order.side == Side.Buy:
            matched_order = self.match_order(self.asks, order)
            if matched_order:
                trade = self.execute_trade(order, matched_order)
                self.trades.append(trade)
                return trade
            self.bids[order.price].append(order)
        else:
            matched_order = self.match_order(self.bids, order)
            if matched_order:
                trade = self.execute_trade(matched_order, order)
                self.trades.append(trade)
                return trade
            self.asks[order.price].append(order)
        return None

    def match_order(self, order_list: Dict[float, List[Order]], new_order: Order) -> Optional[Order]:
        # Simplified matching logic: match if there is an opposite order with the same or better price
        for price, orders in sorted(order_list.items()):
            if (new_order.side == Side.Buy and price <= new_order.price) or \
               (new_order.side == Side.Sell and price >= new_order.price):
                return orders.pop(0)  # Removing the matched order from the list
        return None

    def execute_trade(self, bid_order: Order, ask_order: Order) -> Trade:
        trade_price = (bid_order.price + ask_order.price) / 2  # Simplified trade price calculation
        trade_quantity = min(bid_order.quantity, ask_order.quantity)  # Trade the smallest quantity
        bid_trade_info = TradeInfo(bid_order.order_id, trade_price, trade_quantity)
        ask_trade_info = TradeInfo(ask_order.order_id, trade_price, trade_quantity)
        return Trade(bid_trade_info, ask_trade_info)

    def cancel_order(self, order_id: int, side: Side):
        # Simplified cancellation logic: remove the order with the given ID
        target_dict = self.bids if side == Side.Buy else self.asks
        for price, orders in target_dict.items():
            for order in orders:
                if order.order_id == order_id:
                    orders.remove(order)
                    return

    # Additional methods for order modification, querying order book state, etc., can be added here.



# %%
# utils.py

import uuid
# from .models import Order, Trade

def generate_order_id() -> int:
    """
    Generates a unique order ID using UUID.
    """
    return uuid.uuid4().int & (1<<64)-1

def format_order(order: Order) -> str:
    """
    Returns a string representation of an Order object for display purposes.
    """
    return (f"OrderID: {order.order_id}, Type: {order.order_type.name}, "
            f"Side: {order.side.name}, Price: {order.price}, Quantity: {order.quantity}")

def format_trade(trade: Trade) -> str:
    """
    Returns a string representation of a Trade object for display purposes.
    """
    bid_info = trade.bid_trade
    ask_info = trade.ask_trade
    return (f"Trade executed between OrderID: {bid_info.order_id} (Bid) and "
            f"OrderID: {ask_info.order_id} (Ask) at Price: {bid_info.price} for Quantity: {bid_info.quantity}")

def validate_order_parameters(price: float, quantity: int):
    """
    Validates the price and quantity parameters of an order.
    Raises ValueError if validation fails.
    """
    if price <= 0:
        raise ValueError("Price must be greater than 0.")
    if quantity <= 0:
        raise ValueError("Quantity must be greater than 0.")



# %%
# main.py

# from orderbook.order_book import OrderBook
# from orderbook.models import Order
# from orderbook.enums import OrderType, Side
from datetime import datetime

def main():
    print("Initializing the Order Book System")
    order_book = OrderBook()

    # Example orders
    orders = [
        Order(order_type=OrderType.Market, order_id=1, side=Side.Buy, price=101.0, quantity=10),
        Order(order_type=OrderType.Limit, order_id=2, side=Side.Sell, price=102.0, quantity=15),
        Order(order_type=OrderType.Limit, order_id=3, side=Side.Buy, price=100.5, quantity=5),
        Order(order_type=OrderType.Market, order_id=4, side=Side.Sell, price=101.5, quantity=10)
    ]

    # Adding orders to the order book and checking for matches
    for order in orders:
        print(f"Adding order: {order}")
        trade = order_book.add_order(order)
        if trade:
            print(f"Trade executed: {trade}")

    # Displaying current state of the order book
    print("\nCurrent state of the order book:")
    bids, asks = order_book.get_order_infos()
    print("Bids:")
    for bid in bids:
        print(bid)
    print("Asks:")
    for ask in asks:
        print(ask)

    # Example of canceling an order
    print("\nCanceling order with ID 2")
    order_book.cancel_order(order_id=2, side=Side.Sell)
    
    # Displaying state of the order book after cancellation
    print("State of the order book after cancellation:")
    bids, asks = order_book.get_order_infos()
    print("Bids:")
    for bid in bids:
        print(bid)
    print("Asks:")
    for ask in asks:
        print(ask)

if __name__ == "__main__":
    main()


# %%



