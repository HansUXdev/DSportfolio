import pandas as pd
import os

class DataLoader:
    def __init__(self, directory):
        self.directory = directory

    def search_files(self, file_extension=".csv"):
        """Search for files with a specific extension in the directory."""
        files = [f for f in os.listdir(self.directory) if f.endswith(file_extension)]
        return files

    def load_data(self, file_name):
        """Load a CSV file into a pandas DataFrame."""
        file_path = os.path.join(self.directory, file_name)
        df = pd.read_csv(file_path)
        return df

    def preview_data(self, df, num_rows=5):
        """Preview the first few rows of the DataFrame."""
        return df.head(num_rows)

    def check_data_quality(self, df):
        """Check for missing values and data types."""
        missing_values = df.isnull().sum()
        data_types = df.dtypes
        return missing_values, data_types
