# utils.py

import uuid
from .models import Order, Trade

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

