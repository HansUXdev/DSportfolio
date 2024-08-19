
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









def prepare_data(data, target_columns, shift_target_by=-1):
    """
    Prepares features and target by selecting the correct columns and shifting the target.

    Parameters:
        data (pd.DataFrame): The input DataFrame with all the data.
        target_columns (list): The list of columns to be used as target.
        shift_target_by (int): The number of rows to shift the target by for prediction.

    Returns:
        tuple: Tuple containing the features and target DataFrames.
    """
    # Check if target columns exist
    missing_targets = [col for col in target_columns if col not in data.columns]
    if missing_targets:
        raise ValueError(f"The following target columns are missing from the data: {missing_targets}")
    
    # Assume all other columns are features
    feature_columns = [col for col in data.columns if col not in target_columns]
    
    # Handle missing values
    data = data.dropna(subset=feature_columns + target_columns)

    # Select features and target
    features = data[feature_columns]
    target = data[target_columns].shift(shift_target_by)

    # Drop the rows with NaN values in the target
    valid_indices = target.dropna().index
    features = features.loc[valid_indices]
    target = target.dropna()

    return features, target

def split_data(features, target, test_size=0.2, random_state=42):
    """
    Splits features and target into training and testing sets.

    Parameters:
        features (pd.DataFrame): Features DataFrame.
        target (pd.DataFrame): Target DataFrame.
        test_size (float): The proportion of the dataset to include in the test split.
        random_state (int): Random state for reproducibility.

    Returns:
        tuple: Tuple containing the split data.
    """
    return train_test_split(features, target, test_size=test_size, random_state=random_state)

def train_model(X_train, y_train, n_estimators=100, random_state=42):
    """
    Trains a RandomForestRegressor model on the provided data.

    Parameters:
        X_train (pd.DataFrame): The training features.
        y_train (pd.DataFrame): The training target.
        n_estimators (int): The number of trees in the forest.
        random_state (int): Random state for reproducibility.

    Returns:
        RandomForestRegressor: The trained model.
    """
    model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
    # Flatten y_train to a 1D array if it's not already
    y_train = y_train.values.flatten() if isinstance(y_train, pd.DataFrame) else y_train
    model.fit(X_train, y_train)
    return model

def evaluate_feature_importances(model, feature_columns):
    """
    Evaluates and prints the feature importances of the model.

    Parameters:
        model (RandomForestRegressor): The trained model.
        feature_columns (list): The list of columns that were used as features.

    Returns:
        None
    """
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    print("Feature ranking:")
    for f in range(len(feature_columns)):
        print(f"{f + 1}. feature {feature_columns[indices[f]]} ({importances[indices[f]]})")


def check_data_errors(data):
    errors = []

    # Check for missing values
    if data.isnull().values.any():
        errors.append("Issue: Data contains missing values.")
    
    # Check for duplicate dates
    if data.index.duplicated().any():
        errors.append("Issue: Data contains duplicate dates.")
    
    # Outliers in price data
    z_scores = np.abs((data['Close'] - data['Close'].mean()) / data['Close'].std())
    if z_scores[z_scores > 3].any():
        errors.append("Issue: Data contains potential outliers in 'Close' prices.")
    
    # Volume checks
    if (data['Volume'] == 0).any():
        errors.append("Issue: Data contains days with zero volume.")
    if ((data['Volume'].diff() / data['Volume']).abs() > 5).any():
        errors.append("Issue: Data contains unexpected spikes in volume.")
    
    # Continuity of dates, excluding weekends and public holidays
    cal = USFederalHolidayCalendar()
    holidays = cal.holidays(start=data.index.min(), end=data.index.max())
    business_days = pd.date_range(start=data.index.min(), end=data.index.max(), freq='B')
    business_days = business_days[~business_days.isin(holidays)]  # Exclude holidays
    
    missing_dates = business_days.difference(data.index).tolist()
    if missing_dates:
        formatted_dates = ', '.join([d.strftime('%Y-%m-%d') for d in missing_dates])
        errors.append(f"Issue: Data might be missing trading days: {formatted_dates}")

    return errors