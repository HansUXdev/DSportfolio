class DataCleaningAgent:  
    def fill_missing_values(self, df, method='mean'):  
        """Fill missing values using a specified method."""  
        if method == 'mean':  
            df = df.fillna(df.mean())  
        elif method == 'median':  
            df = df.fillna(df.median())  
        elif method == 'mode':  
            df = df.fillna(df.mode().iloc[0])  
        return df  
  
    def remove_duplicates(self, df):  
        """Remove duplicate rows from the DataFrame."""  
        df = df.drop_duplicates()  
        return df  
  
    def export_cleaned_data(self, df, file_name, directory):  
        """Export the cleaned DataFrame to a new CSV file."""  
        file_path = os.path.join(directory, file_name)  
        df.to_csv(file_path, index=False)  