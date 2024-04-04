# main.py

from orderbook.order_book import OrderBook
from orderbook.models import Order
from orderbook.enums import OrderType, Side
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
