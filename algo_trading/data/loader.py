"""Data loading module with comprehensive error handling."""

import pandas as pd
import requests
from io import StringIO
from typing import Optional
from urllib.parse import urlparse
import logging

logger = logging.getLogger(__name__)


class DataLoadError(Exception):
    """Raised when data loading fails."""
    pass


class DataLoader:
    """Loads and validates financial data from various sources."""

    def __init__(
        self,
        allowed_domains: Optional[list] = None,
        timeout: int = 30,
        max_retries: int = 3,
    ):
        """Initialize DataLoader.

        Args:
            allowed_domains: Whitelist of allowed domains for security
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
        """
        self.allowed_domains = allowed_domains or ["hilpisch.com"]
        self.timeout = timeout
        self.max_retries = max_retries

    def _validate_url(self, url: str) -> None:
        """Validate URL is from trusted source.

        Args:
            url: URL to validate

        Raises:
            DataLoadError: If URL is invalid or untrusted
        """
        parsed = urlparse(url)

        if parsed.scheme not in ["http", "https"]:
            raise DataLoadError(f"Invalid URL scheme: {parsed.scheme}")

        if self.allowed_domains and parsed.netloc not in self.allowed_domains:
            raise DataLoadError(
                f"Untrusted domain: {parsed.netloc}. "
                f"Allowed: {self.allowed_domains}"
            )

    def load_from_url(
        self,
        url: str,
        symbol: Optional[str] = None,
        max_rows: Optional[int] = None,
    ) -> pd.DataFrame:
        """Load data from URL with error handling and validation.

        Args:
            url: Data source URL
            symbol: Symbol to extract (if None, returns all symbols)
            max_rows: Maximum number of rows to load

        Returns:
            DataFrame with loaded data

        Raises:
            DataLoadError: If loading or validation fails
        """
        # Validate URL
        self._validate_url(url)

        # Load data with retries
        for attempt in range(self.max_retries):
            try:
                logger.info(f"Loading data from {url} (attempt {attempt + 1})")

                response = requests.get(url, timeout=self.timeout)
                response.raise_for_status()

                # Parse CSV
                df = pd.read_csv(
                    StringIO(response.text),
                    index_col=0,
                    parse_dates=True,
                )

                break

            except requests.exceptions.Timeout:
                logger.warning(f"Request timeout (attempt {attempt + 1})")
                if attempt == self.max_retries - 1:
                    raise DataLoadError(f"Failed to load data after {self.max_retries} attempts")
                continue

            except requests.exceptions.RequestException as e:
                raise DataLoadError(f"Failed to fetch data from {url}: {e}")

            except pd.errors.ParserError as e:
                raise DataLoadError(f"Invalid CSV format from {url}: {e}")

        # Validate data
        if df.empty:
            raise DataLoadError(f"No data returned from {url}")

        logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")

        # Check for missing values
        initial_rows = len(df)
        df = df.dropna()
        dropped = initial_rows - len(df)

        if dropped > 0:
            logger.warning(
                f"Dropped {dropped} rows with NaN values "
                f"({dropped/initial_rows:.1%} of data)"
            )

        if dropped > initial_rows * 0.5:
            raise DataLoadError(
                f"Too many rows with NaN values: {dropped}/{initial_rows}"
            )

        # Extract symbol if specified
        if symbol:
            if symbol not in df.columns:
                raise DataLoadError(
                    f"Symbol {symbol} not found. "
                    f"Available: {df.columns.tolist()}"
                )

            df = pd.DataFrame(df[symbol])
            logger.info(f"Extracted symbol: {symbol}")

        # Limit rows if specified
        if max_rows and len(df) > max_rows:
            df = df.iloc[:max_rows]
            logger.info(f"Limited to {max_rows} rows")

        # Validate sufficient data
        if len(df) < 100:
            raise DataLoadError(
                f"Insufficient data after cleaning: {len(df)} rows (minimum 100)"
            )

        logger.info(f"Successfully loaded {len(df)} rows")

        return df

    def load_from_csv(self, file_path: str) -> pd.DataFrame:
        """Load data from local CSV file.

        Args:
            file_path: Path to CSV file

        Returns:
            DataFrame with loaded data

        Raises:
            DataLoadError: If loading fails
        """
        try:
            df = pd.read_csv(file_path, index_col=0, parse_dates=True)

            if df.empty:
                raise DataLoadError(f"Empty CSV file: {file_path}")

            logger.info(f"Loaded {len(df)} rows from {file_path}")

            return df

        except FileNotFoundError:
            raise DataLoadError(f"File not found: {file_path}")

        except pd.errors.ParserError as e:
            raise DataLoadError(f"Invalid CSV format: {e}")
