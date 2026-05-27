"""
Main data loading module for price and news data.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
from loguru import logger

from ..config import Config
from .preprocess import PricePreprocessor, NewsPreprocessor
from .aligner import NewsPriceAligner


class CommodityDataLoader:
    """
    Unified data loader for commodity price and news data.
    
    Handles loading, preprocessing, and temporal alignment of:
    - High-frequency price data (OHLCV)
    - Financial news headlines with timestamps
    """
    
    def __init__(self, config: Config, asset: str):
        """
        Initialize data loader.
        
        Args:
            config: Configuration object
            asset: Asset symbol ('gold' or 'oil')
        """
        self.config = config
        self.asset = asset
        self.data_config = config.data
        self.asset_config = self.data_config.assets_config.get(
            asset, self.data_config.default_asset_config
        )
        
        # Initialize preprocessors
        self.price_processor = PricePreprocessor(
            normalization=self.asset_config.price.normalization,
            columns=self.asset_config.price.columns
        )
        self.news_processor = NewsPreprocessor(
            max_length=self.config.model.finbert.max_length,
            language=self.data_config.news.language
        )
        self.aligner = NewsPriceAligner(
            news_lookback=self.data_config.alignment.news_lookback,
            forecast_horizon=self.data_config.alignment.forecast_horizon
        )
        
        # Data storage
        self.prices: Optional[pd.DataFrame] = None
        self.news: Optional[pd.DataFrame] = None
        self.aligned_data: Optional[pd.DataFrame] = None
        
    def load_price_data(self, path: Union[str, Path]) -> pd.DataFrame:
        """
        Load and preprocess price data.
        
        Args:
            path: Path to price data file (CSV, parquet, etc.)
            
        Returns:
            Preprocessed price DataFrame with datetime index
        """
        path = Path(path)
        logger.info(f"Loading price data for {self.asset} from {path}")
        
        # Load based on file extension
        if path.suffix == '.csv':
            df = pd.read_csv(path, parse_dates=['timestamp'])
        elif path.suffix in ['.parquet', '.pq']:
            df = pd.read_parquet(path)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")
        
        # Set timestamp as index
        df = df.set_index('timestamp').sort_index()
        
        # Preprocess
        df = self.price_processor.fit_transform(df, asset=self.asset)
        
        # Add derived features
        df = self._add_price_features(df)
        
        self.prices = df
        logger.info(f"Loaded {len(df)} price bars for {self.asset}")
        return df
    
    def load_news_data(self, path: Union[str, Path]) -> pd.DataFrame:
        """
        Load and preprocess news data.
        
        Args:
            path: Path to news data file or directory
            
        Returns:
            Preprocessed news DataFrame
        """
        path = Path(path)
        logger.info(f"Loading news data from {path}")
        
        # Handle directory vs file
        if path.is_dir():
            files = list(path.glob(f"*.{self.data_config.news.source_format}"))
            dfs = []
            for f in files:
                if self._asset_in_filename(f.name):
                    dfs.append(self._load_news_file(f))
            df = pd.concat(dfs, ignore_index=True)
        else:
            df = self._load_news_file(path)
        
        # Preprocess
        df = self.news_processor.fit_transform(df)
        
        # Filter by asset relevance (optional keyword filtering)
        if self.asset_config.news.keywords:
            df = df[df['headline'].str.contains(
                '|'.join(self.asset_config.news.keywords),
                case=False,
                na=False
            )]
        
        self.news = df
        logger.info(f"Loaded {len(df)} news items for {self.asset}")
        return df
    
    def align_data(self) -> pd.DataFrame:
        """
        Align news with price data based on timestamps.
        
        Returns:
            DataFrame with aligned price bars and associated news
        """
        if self.prices is None or self.news is None:
            raise ValueError("Must load price and news data before alignment")
        
        logger.info(f"Aligning news with {self.asset} price data")
        
        aligned = self.aligner.align(
            prices=self.prices,
            news=self.news,
            lookback=self.data_config.alignment.news_lookback,
            min_news=self.data_config.alignment.min_news_per_bar
        )
        
        # Add target variable (future return)
        aligned = self._add_target(aligned)
        
        self.aligned_data = aligned
        logger.info(f"Created {len(aligned)} aligned samples")
        return aligned
    
    def get_splits(self) -> Dict[str, pd.DataFrame]:
        """
        Split aligned data into train/val/test sets chronologically.
        
        Returns:
            Dictionary with 'train', 'val', 'test' DataFrames
        """
        if self.aligned_data is None:
            raise ValueError("Must align data before splitting")
        
        splits = {}
        for split_name, dates in self.data_config.splits.items():
            mask = (
                (self.aligned_data.index >= pd.Timestamp(dates['start'])) &
                (self.aligned_data.index <= pd.Timestamp(dates['end']))
            )
            splits[split_name] = self.aligned_data[mask].copy()
            logger.info(f"{split_name}: {len(splits[split_name])} samples")
        
        return splits
    
    def _load_news_file(self, path: Path) -> pd.DataFrame:
        """Load a single news file based on format."""
        fmt = self.data_config.news.source_format
        
        if fmt == 'jsonl':
            return pd.read_json(path, lines=True)
        elif fmt == 'csv':
            return pd.read_csv(path, parse_dates=[self.data_config.news.timestamp_column])
        elif fmt == 'parquet':
            df = pd.read_parquet(path)
            df[self.data_config.news.timestamp_column] = pd.to_datetime(
                df[self.data_config.news.timestamp_column]
            )
            return df
        else:
            raise ValueError(f"Unsupported news format: {fmt}")
    
    def _asset_in_filename(self, filename: str) -> bool:
        """Check if filename contains asset identifier."""
        asset_map = {'gold': ['gold', 'xau', 'xauusd'], 'oil': ['oil', 'wti', 'crude']}
        keywords = asset_map.get(self.asset, [self.asset])
        return any(kw in filename.lower() for kw in keywords)
    
    def _add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add derived price features."""
        df = df.copy()
        
        # Returns
        df['return'] = df['close'].pct_change()
        df['log_return'] = np.log(df['close'] / df['close'].shift(1))
        
        # Volatility (rolling std)
        window = self.asset_config.price.get('volatility_window', 12)
        df['volatility'] = df['return'].rolling(window).std()
        
        # Price range
        df['range'] = (df['high'] - df['low']) / df['close']
        
        # Time features
        df['hour'] = df.index.hour
        df['day_of_week'] = df.index.dayofweek
        df['is_news_day'] = df.index.dayofweek < 5  # Weekday flag
        
        return df
    
    def _add_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add prediction target (future return over forecast horizon)."""
        df = df.copy()
        
        # Calculate forward return
        horizon_minutes = int(
            self.data_config.alignment.forecast_horizon.replace('min', '')
        )
        steps_ahead = horizon_minutes // int(
            self.data_config.price.frequency.replace('min', '')
        )
        
        df['target_return'] = df['close'].shift(-steps_ahead) / df['close'] - 1
        df['target_direction'] = (df['target_return'] > 0).astype(int)
        
        # Remove rows with NaN target
        df = df.dropna(subset=['target_return'])
        
        return df
