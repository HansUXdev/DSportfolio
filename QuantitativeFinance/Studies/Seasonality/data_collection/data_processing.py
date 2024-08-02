from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.linear_model import LinearRegression

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

""

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

from pandas.tseries.holiday import USFederalHolidayCalendar

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

# # Example usage
# spy_data = fetch_and_process_data("SPY")  # Assuming this function returns data with DateTimeIndex
# errors = check_data_errors(spy_data)
# if errors:
#     for error in errors:
#         print(error)
# else:
#     print("No issues detected in the data.")
