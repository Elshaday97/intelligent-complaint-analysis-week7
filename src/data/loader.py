import pandas as pd
from scripts.constants import (
    RAW_COMPLAINTS_DATA_FILE_NAME,
    CLENAED_COMPLAINTS_DATA_FILE_NAME,
    PROCESSED_FILE_DIR,
    RAW_FILE_DIR,
    date_columns,
    Columns,
)
from pathlib import Path
from sklearn.model_selection import train_test_split
import math


class DataLoader:
    """Class to load and save complaint data from/to CSV files."""

    def __init__(self):
        self.raw_complaints_file_path = RAW_FILE_DIR + RAW_COMPLAINTS_DATA_FILE_NAME
        self.cleaned_complaints_file_path = (
            PROCESSED_FILE_DIR + CLENAED_COMPLAINTS_DATA_FILE_NAME
        )
        self.chunk_size = 10000
        self.chunks = []

    def load_from_csv(
        self,
        parse_dates: bool = False,
        load_clean: bool = False,
    ) -> pd.DataFrame:
        """
        Loads complaint data from a CSV file into a Pandas DataFrame.
        Args:
            parse_dates (bool): Whether to parse date columns.
            load_clean (bool): Whether to load cleaned data or raw data.
        Returns:
            pd.DataFrame: DataFrame containing the complaint data.
        Raises:
            FileNotFoundError: If the specified file does not exist.
            ValueError: If the loaded DataFrame is empty.
        """
        file_to_read = (
            self.cleaned_complaints_file_path
            if load_clean
            else self.raw_complaints_file_path
        )
        parse_dates = date_columns if parse_dates else None

        if not Path(file_to_read).exists:
            raise FileNotFoundError(f"File {file_to_read} not found")

        for chunk in pd.read_csv(
            file_to_read, chunksize=self.chunk_size, parse_dates=parse_dates
        ):
            self.chunks.append(chunk)

        df = pd.concat(self.chunks)
        print(f"Loaded {file_to_read} to Dataframe!")

        if df.empty:
            raise ValueError("Dataframe is empty. Please select another file")

        return df

    def save_to_csv(
        self, df: pd.DataFrame, default_cleaned: bool = True, path: str = ""
    ):
        """
        Saves the given DataFrame to a CSV file.
        Args:
            df (pd.DataFrame): DataFrame to save.
            default_cleaned (bool): Whether to save to the default cleaned file path.
            path (str): Custom file path to save the DataFrame if not using default.
        """
        file_path_to_save = (
            self.cleaned_complaints_file_path if default_cleaned else path
        )
        df.to_csv(file_path_to_save)
        print(f"Saved dataframe to {file_path_to_save}")

    def load_stratified_sample(self, n_samples=1000):
        df = pd.read_csv(
            self.cleaned_complaints_file_path,
        )

        stratified_sample, _ = train_test_split(
            df,
            test_size=0.2,
            stratify=df[Columns.PRODUCT.value],
            random_state=42,
        )

        return stratified_sample
