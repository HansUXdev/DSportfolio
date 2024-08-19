############################################################################################################
# Options Utilities
############################################################################################################
# Function to detrend time series data using a linear regression model
def detrend_data(data, column):
    # Linear regression model requires reshaped index as a feature
    X = np.arange(len(data)).reshape(-1, 1)
    y = data[column].values  # Original values to detrend
    
    # Create and fit the model
    model = LinearRegression()
    model.fit(X, y)
    
    # Predict the trend
    trend = model.predict(X)
    
    # Detrend by subtracting the trend from the original data
    detrended = y - trend
    data[f'{column}_detrended'] = detrended
    
    # Return the detrended data and the trend for further analysis
    return data, trend

def seasonal_decomposition(data, column, period):
    # Perform seasonal decomposition
    decomposition = seasonal_decompose(data[column], model='multiplicative', period=period)
    
    # Add components to DataFrame
    data['trend_component'] = decomposition.trend
    data['seasonal_component'] = decomposition.seasonal
    data['residual_component'] = decomposition.resid
    
    # Seasonally adjust the data
    data[column + '_seasonally_adjusted'] = data[column] / data['seasonal_component']
    
    return data

# Function to calculate price differences
def calculate_price_differences(data, column):
    data[f'{column}_diff'] = data[column].diff()
    return data

# Function to calculate log returns
def calculate_log_returns(data, column):
    data[f'{column}_log_return'] = np.log(data[column] / data[column].shift(1))
    return data

# Function to calculate volume changes
def calculate_volume_changes(data, volume_column):
    data[f'{volume_column}_changes'] = data[volume_column].diff()
    return data
def black_scholes(S, K, T, r, sigma, option_type):
    # S: spot price of the asset
    # K: strike price
    # T: time to maturity
    # r: risk-free rate
    # sigma: volatility of the asset
    # option_type: 'call' or 'put'
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == "call":
        price = (S * si.norm.cdf(d1, 0.0, 1.0) - K * np.exp(-r * T) * si.norm.cdf(d2, 0.0, 1.0))
    else:
        price = (K * np.exp(-r * T) * si.norm.cdf(-d2, 0.0, 1.0) - S * si.norm.cdf(-d1, 0.0, 1.0))
    return price

def binomial_tree(S, K, T, r, sigma, N, option_type):
    # Parameters as described in the Black-Scholes function
    # N: number of binomial steps
    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp(r * dt) - d) / (u - d)
    C = np.zeros((N + 1, N + 1))
    for i in range(N + 1):
        C[i, N] = max(0, S * d**i * u**(N - i) - K if option_type == "call" else K - S * d**i * u**(N - i))
    for j in range(N - 1, -1, -1):
        for i in range(j + 1):
            C[i, j] = (p * C[i, j + 1] + (1 - p) * C[i + 1, j + 1]) * np.exp(-r * dt)
    return C[0, 0]

def monte_carlo_simulation(S, K, T, r, sigma, n_simulations, option_type):
    dt = T / 365
    results = []
    for _ in range(n_simulations):
        path = S * np.cumprod(np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * np.random.normal(size=365)))
        if option_type == "call":
            results.append(max(0, path[-1] - K))
        else:
            results.append(max(0, K - path[-1]))
    mc_price = np.mean(results) * np.exp(-r * T)
    return mc_price


# # Section 4: Finite Difference Methods for American Options
# print("## Finite Difference Methods for American Options")
def finite_difference_american_option(S, K, T, r, sigma, option_type):
    # Grid parameters
    N = 1000  # time steps
    M = 200  # price steps
    dt = T / N
    dS = 2 * S / M
    grid = np.zeros((M+1, N+1))
    S_values = np.linspace(0, 2 * S, M+1)
    
    # Set up the final conditions
    if option_type == "call":
        grid[:, -1] = np.maximum(S_values - K, 0)
    else:
        grid[:, -1] = np.maximum(K - S_values, 0)
    
    # Coefficients for the matrix
    a = 0.5 * dt * (sigma**2 * np.arange(M+1)**2 - r * np.arange(M+1))
    b = -dt * (sigma**2 * np.arange(M+1)**2 + r)
    c = 0.5 * dt * (sigma**2 * np.arange(M+1)**2 + r * np.arange(M+1))
    
    # Solving the equation backwards in time
    for j in reversed(range(N)):
        rhs = grid[:, j+1]
        # Set up the matrix
        mat = np.zeros((3, M+1))
        mat[0, 1:] = -a[1:]
        mat[1, :] = 1 - b
        mat[2, :-1] = -c[:-1]
        grid[1:-1, j] = solve_banded((1, 1), mat[:, 1:-1], rhs[1:-1])
        # Apply early exercise condition
        if option_type == "call":
            grid[:, j] = np.maximum(grid[:, j], S_values - K)
        else:
            grid[:, j] = np.maximum(grid[:, j], K - S_values)

    return grid[M//2, 0]
def heston_model(S, K, T, r, kappa, theta, xi, rho, v0, n_simulations, option_type):
    dt = T / 365
    prices = np.zeros(n_simulations)
    v = np.maximum(v0 + np.zeros(n_simulations), 0)
    
    for t in range(1, 365):
        dw1 = np.random.normal(size=n_simulations)
        dw2 = rho * dw1 + np.sqrt(1 - rho**2) * np.random.normal(size=n_simulations)
        S += S * (r * dt + np.sqrt(v) * np.sqrt(dt) * dw1)
        v += kappa * (theta - v) * dt + xi * np.sqrt(v) * np.sqrt(dt) * dw2
        v = np.maximum(v, 0)
    
    if option_type == "call":
        prices = np.exp(-r * T) * np.maximum(S - K, 0)
    else:
        prices = np.exp(-r * T) * np.maximum(K - S, 0)
    
    return np.mean(prices)
def merton_jump_diffusion(S, K, T, r, sigma, lambda_, mu_j, sigma_j, option_type):
    """
    Merton's Jump Diffusion model for option pricing.
    lambda_: Jump frequency per year
    mu_j: Mean jump size
    sigma_j: Jump size volatility
    """
    def integrand(k):
        # Merton's characteristic function part for jump
        jump_part = np.exp(-lambda_ * T + k * np.log(1 + mu_j) + 0.5 * k**2 * sigma_j**2 * T)
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2 - k * mu_j) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        if option_type == "call":
            return np.exp(-r * T) * jump_part * (S * si.norm.cdf(d1) - K * si.norm.cdf(d2))
        else:
            return np.exp(-r * T) * jump_part * (K * si.norm.cdf(-d2) - S * si.norm.cdf(-d1))

    # Numerical integration for the jump diffusion part
    price, _ = quad(integrand, 0, np.inf)
    return price

def barrier_option(S, K, H, T, r, sigma, option_type, barrier_type):
    """
    Analytical price for European barrier options.
    H: Barrier level
    barrier_type: 'up-and-out' or 'down-and-out'
    """
    # Coefficients for barrier options
    mu = (r - 0.5 * sigma**2) / (sigma**2)
    lambda_ = np.sqrt(mu**2 + 2 * r / sigma**2)
    x1 = np.log(S / K) / (sigma * np.sqrt(T)) + (1 + mu) * sigma * np.sqrt(T)
    x2 = np.log(S / H) / (sigma * np.sqrt(T)) + (1 + mu) * sigma * np.sqrt(T)
    y1 = np.log(H**2 / (S * K)) / (sigma * np.sqrt(T)) + (1 + mu) * sigma * np.sqrt(T)
    y2 = np.log(H / S) / (sigma * np.sqrt(T)) + (1 + mu) * sigma * np.sqrt(T)
    
    if barrier_type == "up-and-out":
        if option_type == "call":
            if H > K:
                price = black_scholes(S, K, T, r, sigma, option_type) \
                        - black_scholes(S, H, T, r, sigma, option_type) \
                        - (H - K) * np.exp(-r * T) * si.norm.cdf(x2) \
                        + (H - K) * np.exp(-r * T) * si.norm.cdf(y2)
            else:
                price = 0  # Option knocked out
        else:
            raise ValueError("Barrier put options are not typically used with up-and-out barriers.")
    elif barrier_type == "down-and-out":
        if option_type == "call":
            price = black_scholes(S, K, T, r, sigma, option_type) \
                    - (H - K) * np.exp(-r * T) * si.norm.cdf(-y2)
        else:
            price = black_scholes(S, K, T, r, sigma, option_type) \
                    - black_scholes(S, H, T, r, sigma, option_type) \
                    - (H - K) * np.exp(-r * T) * si.norm.cdf(-y1)

    return price

def calculate_implied_volatility(options_df):
    options_df['mid'] = (options_df['bid'] + options_df['ask']) / 2
    atm_options = options_df.loc[(options_df['strike'] == options_df['strike'].median()) & (options_df['option_type'] == 'call')]
    atm_options['implied_volatility'] = atm_options['impliedVolatility']
    return atm_options['implied_volatility']
