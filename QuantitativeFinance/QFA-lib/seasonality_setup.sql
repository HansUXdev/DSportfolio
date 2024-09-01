-- Create a database (optional, depending on the system setup)
CREATE DATABASE IF NOT EXISTS seasonality_analysis;

-- Switch to the database (useful for environments like MySQL)
USE seasonality_analysis;

-- Create the assets table
CREATE TABLE IF NOT EXISTS assets (
    asset_id SERIAL PRIMARY KEY,  -- Auto-incrementing ID
    asset_name VARCHAR(50) NOT NULL,
    asset_type VARCHAR(20),
    description TEXT
);

-- Create the daily_returns table
CREATE TABLE IF NOT EXISTS daily_returns (
    date DATE NOT NULL,
    asset_id INTEGER NOT NULL,
    daily_return FLOAT,
    market_hours_return FLOAT,
    pre_market_return FLOAT,
    post_market_return FLOAT,
    PRIMARY KEY (date, asset_id),
    FOREIGN KEY (asset_id) REFERENCES assets(asset_id)
);

-- Create the weekly_returns table
CREATE TABLE IF NOT EXISTS weekly_returns (
    year INTEGER NOT NULL,
    week_number INTEGER NOT NULL,
    asset_id INTEGER NOT NULL,
    weekly_return FLOAT,
    PRIMARY KEY (year, week_number, asset_id),
    FOREIGN KEY (asset_id) REFERENCES assets(asset_id)
);

-- Create the monthly_returns table
CREATE TABLE IF NOT EXISTS monthly_returns (
    year INTEGER NOT NULL,
    month INTEGER NOT NULL,
    asset_id INTEGER NOT NULL,
    monthly_return FLOAT,
    PRIMARY KEY (year, month, asset_id),
    FOREIGN KEY (asset_id) REFERENCES assets(asset_id)
);

-- Create the intraday_returns table
CREATE TABLE IF NOT EXISTS intraday_returns (
    date DATE NOT NULL,
    time_period VARCHAR(20) NOT NULL,  -- e.g., 'pre-market', 'market', 'post-market'
    asset_id INTEGER NOT NULL,
    return FLOAT,
    PRIMARY KEY (date, time_period, asset_id),
    FOREIGN KEY (asset_id) REFERENCES assets(asset_id)
);

-- Indexes for daily_returns
CREATE INDEX IF NOT EXISTS idx_daily_returns_date ON daily_returns(date);
CREATE INDEX IF NOT EXISTS idx_daily_returns_asset ON daily_returns(asset_id);

-- Indexes for weekly_returns
CREATE INDEX IF NOT EXISTS idx_weekly_returns_year_week ON weekly_returns(year, week_number);
CREATE INDEX IF NOT EXISTS idx_weekly_returns_asset ON weekly_returns(asset_id);

-- Indexes for monthly_returns
CREATE INDEX IF NOT EXISTS idx_monthly_returns_year_month ON monthly_returns(year, month);
CREATE INDEX IF NOT EXISTS idx_monthly_returns_asset ON monthly_returns(asset_id);

-- Indexes for intraday_returns
CREATE INDEX IF NOT EXISTS idx_intraday_returns_date ON intraday_returns(date);
CREATE INDEX IF NOT EXISTS idx_intraday_returns_time_period ON intraday_returns(time_period);
CREATE INDEX IF NOT EXISTS idx_intraday_returns_asset ON intraday_returns(asset_id);

-- Commit any changes (for systems like MySQL)
COMMIT;
