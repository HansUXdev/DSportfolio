



OrderBook-Python-Project
│
├── main.py                     - Main entry point for the application
│
├── orderbook                   - Package for order book components
│   ├── __init__.py             - Initializes the orderbook package and exposes components
│   ├── models.py               - Defines data models (Order, Trade, TradeInfo, LevelInfo)
│   ├── enums.py                - Enumerations (OrderType, Side)
│   ├── constants.py            - Defines system-wide constants
│   ├── order_book.py           - Contains the OrderBook class managing the order book logic
│   └── utils.py                - Utility functions and common definitions
│
├── tests                       - Directory for test cases
│   ├── __init__.py             - Initializes the tests package
│   ├── test_order_book.py      - Test cases for order book functionality
│   └── test_utils.py           - Test cases for utility functions
│
├── README.md                   - Project overview, setup instructions, and usage
└── requirements.txt            - Lists dependencies for the project
