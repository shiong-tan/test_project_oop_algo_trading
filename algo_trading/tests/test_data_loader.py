"""Unit tests for data loader module."""

import pytest
import pandas as pd
import numpy as np
from algo_trading.data.loader import DataLoader, DataLoadError


class TestDataLoader:
    """Test suite for DataLoader class."""

    def test_initialization(self):
        """Test DataLoader initialization."""
        loader = DataLoader()
        assert loader.allowed_domains == ["hilpisch.com"]
        assert loader.timeout == 30
        assert loader.max_retries == 3

    def test_initialization_custom_params(self):
        """Test DataLoader with custom parameters."""
        loader = DataLoader(
            allowed_domains=["example.com"],
            timeout=60,
            max_retries=5,
        )
        assert loader.allowed_domains == ["example.com"]
        assert loader.timeout == 60
        assert loader.max_retries == 5

    def test_validate_url_valid(self):
        """Test URL validation with valid URL."""
        loader = DataLoader()
        # Should not raise exception
        loader._validate_url("http://hilpisch.com/data.csv")
        loader._validate_url("https://hilpisch.com/data.csv")

    def test_validate_url_invalid_scheme(self):
        """Test URL validation with invalid scheme."""
        loader = DataLoader()
        with pytest.raises(DataLoadError, match="Invalid URL scheme"):
            loader._validate_url("ftp://hilpisch.com/data.csv")

    def test_validate_url_untrusted_domain(self):
        """Test URL validation with untrusted domain."""
        loader = DataLoader(allowed_domains=["hilpisch.com"])
        with pytest.raises(DataLoadError, match="Untrusted domain"):
            loader._validate_url("http://malicious.com/data.csv")

    def test_load_from_csv_nonexistent(self):
        """Test loading from nonexistent CSV file."""
        loader = DataLoader()
        with pytest.raises(DataLoadError, match="File not found"):
            loader.load_from_csv("/nonexistent/file.csv")
