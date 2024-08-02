# src/data_collection/fundamentals.py

import yfinance as yf
import pandas as pd

def get_fundamentals_data(ticker):
    data = yf.Ticker(ticker)
    fundamentals = {
        'balance_sheet': data.balance_sheet,
        'cashflow': data.cashflow,
        'earnings': data.earnings,
        'financials': data.financials
    }
    return fundamentals


def fetch_fundamentals(ticker):
    """
    Fetches comprehensive fundamental data for a given ticker, including balance sheet and cash flow,
    and returns it as a DataFrame.

    Args:
    - ticker (str): The ticker symbol of the stock.

    Returns:
    - DataFrame: DataFrame with merged market, technical, and fundamental data.
    """
    try:
        # Define start date and end date based on current date and one year ago
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')

        ticker_obj = yf.Ticker(ticker)
        
        # Fetch Beta from ticker's info
        beta_value = ticker_obj.info.get('beta', 0)
        
        balance_sheet = ticker_obj.balance_sheet
        cashflow = ticker_obj.cashflow

        balance_sheet_transposed = balance_sheet.T
        cashflow_transposed = cashflow.T

        fundamentals = pd.concat([balance_sheet_transposed, cashflow_transposed], axis=1)
        fundamentals.index.names = ['Date']
        
        # Insert Beta as the first column
        fundamentals.insert(0, 'Beta', beta_value)

        fundamentals.fillna(method='backfill', inplace=True)
        fundamentals.fillna(method='ffill', inplace=True)
        fundamentals.fillna(0, inplace=True)

        # Example of calculating growth rate of free cash flows (replace with your actual data)
        free_cash_flows = pd.Series([100, 120, 140, 160, 180])
        growth_rate = free_cash_flows.pct_change().mean()
        print("Free Cash Flow Growth Rate:", growth_rate)

        return fundamentals

    except Exception as e:
        print(f"Failed to fetch or process fundamental data for {ticker}: {e}")
        return pd.DataFrame()  # Return empty DataFrame in case of failure

# # Apply the function to fetch and merge fundamental data for MSFT
# fundamental_data = fetch_fundamentals('MSFT')

# # Displaying the first few rows of the fundamental dataset
# fundamental_data

def calculate_cashflow_growth_rate(free_cash_flows):
    return free_cash_flows.pct_change().mean()

def project_future_free_cash_flows(last_cash_flow, growth_rate, years):
    return [last_cash_flow * (1 + growth_rate) ** i for i in range(1, years + 1)]


def calculate_terminal_value(last_cash_flow, growth_rate, required_rate, years):
    return last_cash_flow * (1 + growth_rate) / (required_rate - growth_rate) / (1 + required_rate) ** years

def calculate_fair_value(discounted_cash_flows, terminal_value, outstanding_shares):
    total_present_value = sum(discounted_cash_flows) + terminal_value
    return total_present_value / outstanding_shares

def get_cost_of_equity(risk_free_rate, beta, market_return):
    return risk_free_rate + beta * (market_return - risk_free_rate)

def get_cost_of_debt(interest_rate, tax_rate):
    return interest_rate * (1 - tax_rate)

def get_proportions(market_value_equity, market_value_debt):
    total_value = market_value_equity + market_value_debt
    return market_value_equity / total_value, market_value_debt / total_value

def calculate_wacc(cost_of_equity, cost_of_debt, equity_proportion, debt_proportion, tax_rate):
    wacc = (cost_of_equity * equity_proportion) + ((1 - tax_rate) * cost_of_debt * debt_proportion)
    return wacc

def calculate_intrinsic_value(dividend_data, discount_rate):
    intrinsic_value = 0
    for year, dividend in enumerate(dividend_data, start=1):
        if year <= 5:
            growth_rate = 0.05
        elif 5 < year <= 10:
            growth_rate = 0.03
        else:
            growth_rate = 0.01
        intrinsic_value += dividend / ((1 + discount_rate) ** year)
    return intrinsic_value

def calculate_cost_of_equity(beta, risk_free_rate, market_return):
    """
    Calculate the cost of equity using the CAPM formula.
    
    :param beta: Beta of the stock
    :param risk_free_rate: Risk-free rate
    :param market_return: Expected market return
    :return: Cost of equity
    """
    return risk_free_rate + beta * (market_return - risk_free_rate)

def dcf_valuation(cash_flows, discount_rate):
    """
    Calculate the present value of cash flows using the discounted cash flow (DCF) method.
    
    Args:
    - cash_flows (list): List of projected cash flows.
    - discount_rate (float): Discount rate (required rate of return).
    
    Returns:
    - float: Present value of the cash flows.
    """
    dcf_value = sum(cf / (1 + discount_rate)**n for n, cf in enumerate(cash_flows, start=1))
    return dcf_value

def calculate_expected_return(risk_free_rate, beta, market_return, market_risk_premium):
    """
    Calculate the expected return of an asset using the Capital Asset Pricing Model (CAPM).
    
    Args:
    - risk_free_rate (float): Risk-free rate (e.g., yield on Treasury bills).
    - beta (float): Beta coefficient of the asset.
    - market_return (float): Expected return of the market portfolio.
    - market_risk_premium (float): Market risk premium.
    
    Returns:
    - float: Expected return of the asset.
    """
    expected_return = risk_free_rate + beta * market_risk_premium
    return expected_return

def three_stage_dividend_discount_model(symbol, discount_rate):
    dividend_data = fetch_dividend_data(symbol)
    intrinsic_value = calculate_intrinsic_value(dividend_data, discount_rate)
    return intrinsic_value


def residual_income_model(net_income, equity, required_return):
    """
    Calculate the value of equity using the Residual Income Model.
    
    Args:
    - net_income (float): Net income of the company.
    - equity (float): Book value of equity.
    - required_return (float): Required rate of return on equity.
    
    Returns:
    - float: Estimated value of equity using the Residual Income Model.
    """
    # Calculate the present value of expected future residual income
    residual_income = net_income - (required_return * equity)
    
    # Value of equity is the book value of equity plus the present value of expected future residual income
    equity_value = equity + residual_income
    
    return equity_value

