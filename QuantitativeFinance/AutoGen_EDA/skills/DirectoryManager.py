# filename: directory_manager.py

import os
import logging

class DirectoryManager:
    def __init__(self, directory_path='./data/csv_files/'):
        """
        Initialize the DirectoryManager with a specified directory path.
        :param directory_path: The path to the directory to manage.
        """
        self.directory_path = directory_path
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        logging.info(f"Initialized DirectoryManager for directory: {self.directory_path}")

    def create_directory(self):
        """
        Create the directory if it does not exist.
        """
        try:
            if not os.path.exists(self.directory_path):
                os.makedirs(self.directory_path)
                logging.info(f"Directory '{self.directory_path}' has been created.")
            else:
                logging.info(f"Directory '{self.directory_path}' already exists.")
        except Exception as e:
            logging.error(f"Failed to create directory '{self.directory_path}': {e}")
            raise

    def check_csv_files(self):
        """
        Check if the directory exists and list all CSV files in the directory.
        :return: List of CSV files in the directory.
        """
        try:
            if not os.path.exists(self.directory_path):
                logging.warning(f"Directory '{self.directory_path}' does not exist.")
                return []
            else:
                csv_files = [f for f in os.listdir(self.directory_path) if f.endswith('.csv')]
                if not csv_files:
                    logging.info(f"No CSV files found in the directory '{self.directory_path}'.")
                else:
                    logging.info(f"CSV files found in the directory '{self.directory_path}': {csv_files}")
                return csv_files
        except Exception as e:
            logging.error(f"Error checking CSV files in directory '{self.directory_path}': {e}")
            raise

# Usage example
if __name__ == "__main__":
    # Initialize the DirectoryManager for the desired directory
    dir_manager = DirectoryManager('./data/csv_files/')

    # Create the directory if it doesn't exist
    dir_manager.create_directory()

    # Check for CSV files in the directory
    csv_files = dir_manager.check_csv_files()
    print(f"CSV files in the directory: {csv_files}")
