import pandas as pd
from scripts.constants import Columns, product_categories
from scripts.utils import clean_text


class DataPreprocessor:
    """
    Class to preprocess complaint data.
    Includes methods to subset product categories, drop missing values,
    remove empty feedbacks, and clean customer feedback."""

    def __init__(self, df: pd.DataFrame):

        self.df = df

    def subset_product_categories(self, df: pd.DataFrame):
        """
        Subsets the DataFrame to include only specified product categories.
        Args:
            df (pd.DataFrame): The input DataFrame to subset.
        Returns:
            pd.DataFrame: Subsetted DataFrame containing only specified product categories.
        """
        return df[self.df[Columns.PRODUCT.value].isin(product_categories)]

    def drop_missing_values(self):
        """
        Drops rows with missing values in the complaint narrative column.
        """
        initial_shape = self.df.shape

        self.df = self.df.dropna(subset=[Columns.COMPLAINT.value])

        dropped_count = initial_shape[0] - self.df.shape[0]
        if dropped_count > 0:
            print(f"Dropped {dropped_count} rows containing null narratives.")

        print(f"New shape: {self.df.shape}")

    def remove_empty_feedbacks(self):
        """
        Removes rows where the complaint narrative is empty or contains only whitespace.
        """
        self.df = self.df[self.df[Columns.COMPLAINT.value].str.strip().str.len() > 0]

    def clean_customer_feedback(self, df: pd.DataFrame, col: str):
        """
        Cleans the specified text column in the DataFrame using the clean_text function.
        Args:
            df (pd.DataFrame): The input DataFrame containing the text column to clean.
            col (str): The name of the column to clean.
        Returns:
            pd.DataFrame: DataFrame with the cleaned text column.
        """
        df[col] = df[col].apply(clean_text)
        return df

    def get_processed_data(self):
        """
        Executes the preprocessing steps and returns the processed DataFrame.
        Returns:
            pd.DataFrame: The processed DataFrame after all preprocessing steps.
        """
        self.drop_missing_values()
        self.remove_empty_feedbacks()
        return self.df
