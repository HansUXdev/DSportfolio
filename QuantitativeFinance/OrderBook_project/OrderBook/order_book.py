# order_book.py

from models import Order, Trade, TradeInfo
from enums import Side
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
    def get_order_infos(self):
        # This example assumes bids and asks are stored in dictionaries keyed by price
        # and returns a list of orders for each.
        # Adjust this logic based on your actual data structure.
        bids_info = [(price, orders) for price, orders in self.bids.items()]
        asks_info = [(price, orders) for price, orders in self.asks.items()]
        return bids_info, asks_info
    

    
    # Additional methods for order modification, querying order book state, etc., can be added here.

