import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
from src.data.loader import DataLoader


@pytest.fixture
def sample_df():
    return pd.DataFrame(
        {
            "complaint_id": [1, 2, 3, 4, 5, 6, 7, 8],
            "Product": ["A", "B", "A", "B", "B", "A", "A", "B"],
            "narrative": ["text"] * 8,
        }
    )


class TestDataLoader:
    @pytest.fixture
    def loader(self):
        return DataLoader()

    @patch("src.data.loader.pd.read_csv")
    @patch("src.data.loader.Path.exists")
    def test_load_from_csv_success(self, mock_exists, mock_read_csv, loader, sample_df):
        mock_exists.return_value = True
        mock_read_csv.return_value = [sample_df]

        result_df = loader.load_from_csv(load_clean=True)

        assert not result_df.empty
        assert len(result_df) == 8
        mock_read_csv.assert_called_once()

    @patch("src.data.loader.Path.exists")
    def test_load_from_csv_file_not_found(self, mock_exists, loader):
        mock_exists.return_value = False

        with pytest.raises(FileNotFoundError):
            loader.load_from_csv()

    @patch("src.data.loader.pd.read_csv")
    @patch("src.data.loader.Path.exists")
    def test_data_frame_empty(self, mock_exists, mock_read_csv, loader):
        mock_exists.return_value = True

        empty_df = pd.DataFrame()
        mock_read_csv.return_value = [empty_df]

        with pytest.raises(ValueError, match="Dataframe is empty"):
            loader.load_from_csv()

    @patch("src.data.loader.pd.read_csv")
    def test_load_stratified_sample(self, mock_read_csv, loader, sample_df):

        mock_read_csv.return_value = sample_df

        result = loader.load_stratified_sample(sample_size=0.2, stratify_col="Product")

        assert len(result) > 1
        assert "Product" in result.columns
