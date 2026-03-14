"""
Data loader - CORRECTED version that reads RBI changes directly.
Your Excel column 28 already has discrete changes - no diff() needed!
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, List
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class DataLoader:
    """Load data with proper RBI event handling."""
    
    def __init__(self, excel_path: str, clean_data_sheet: str = "Clean Data", 
                 stock_data_sheet: str = "Stock Data"):
        """Initialize DataLoader."""
        self.excel_path = excel_path
        self.clean_data_sheet = clean_data_sheet
        self.stock_data_sheet = stock_data_sheet
        
    def load_external_indicators(self) -> pd.DataFrame:
        """Load external market indicators."""
        logger.info(f"Loading external indicators from {self.clean_data_sheet}")
        
        df = pd.read_excel(self.excel_path, sheet_name=self.clean_data_sheet)
        clean_df = self._build_indicator_dataframe(df)
        
        logger.info(f"Loaded {len(clean_df)} rows of external indicators")
        logger.info(f"Date range: {clean_df.index.min()} to {clean_df.index.max()}")
        
        return clean_df
    
    def _build_indicator_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build clean indicator DataFrame."""
        
        data = {}
        data['Date'] = pd.to_datetime(df.iloc[1:, 0], errors='coerce')
        
        # Extract all indicators
        data['NIFTY50_Price'] = pd.to_numeric(df.iloc[1:, 1], errors='coerce')
        data['NIFTY50_Returns'] = pd.to_numeric(df.iloc[1:, 2], errors='coerce')
        data['India_VIX'] = pd.to_numeric(df.iloc[1:, 3], errors='coerce')
        data['India_VIX_Returns'] = pd.to_numeric(df.iloc[1:, 5], errors='coerce')
        data['FII_Net_Investment'] = pd.to_numeric(df.iloc[1:, 6], errors='coerce')
        data['DII_Net_Investment'] = pd.to_numeric(df.iloc[1:, 8], errors='coerce')
        data['USDINR'] = pd.to_numeric(df.iloc[1:, 10], errors='coerce')
        data['USDINR_Returns'] = pd.to_numeric(df.iloc[1:, 12], errors='coerce')
        data['India_10Y_Yield'] = pd.to_numeric(df.iloc[1:, 13], errors='coerce')
        data['India_10Y_Yield_Returns'] = pd.to_numeric(df.iloc[1:, 15], errors='coerce')
        data['Brent_Crude'] = pd.to_numeric(df.iloc[1:, 16], errors='coerce')
        data['Brent_Crude_Returns'] = pd.to_numeric(df.iloc[1:, 18], errors='coerce')
        data['SP500_Price'] = pd.to_numeric(df.iloc[1:, 20], errors='coerce')
        data['SP500_Returns'] = pd.to_numeric(df.iloc[1:, 22], errors='coerce')
        data['US_Dollar_Index'] = pd.to_numeric(df.iloc[1:, 23], errors='coerce')
        data['US_Dollar_Index_Returns'] = pd.to_numeric(df.iloc[1:, 25], errors='coerce')
        
        # RBI Repo Rate (column 27) and Change (column 28)
        data['RBI_Repo_Rate'] = pd.to_numeric(df.iloc[1:, 27], errors='coerce')
        
        # CRITICAL: Column 28 already has discrete changes - read directly!
        # DO NOT apply diff() - the Excel already has the changes calculated
        rbi_change_raw = pd.to_numeric(df.iloc[1:, 28], errors='coerce')
        data['RBI_Repo_Change'] = rbi_change_raw.fillna(0)
        
        clean_df = pd.DataFrame(data)
        clean_df = clean_df[clean_df['Date'].notna()].copy()
        clean_df = clean_df.drop_duplicates(subset=['Date'], keep='first')
        clean_df.set_index('Date', inplace=True)
        clean_df.sort_index(inplace=True)
        clean_df.dropna(subset=['NIFTY50_Price'], inplace=True)
        
        # Log RBI event statistics
        rbi_events = clean_df['RBI_Repo_Change'][clean_df['RBI_Repo_Change'].abs() > 1e-8]
        logger.info(f"\nRBI Repo Rate Events:")
        logger.info(f"  Total events: {len(rbi_events)}")
        
        if len(rbi_events) > 0:
            logger.info(f"  Event magnitudes: {sorted(rbi_events.unique())}")
            logger.info(f"  Sample events (first 10):")
            for i, (date, change) in enumerate(list(rbi_events.head(10).items())):
                logger.info(f"    {date.date()}: {change:+.6f} ({change*100:+.2f}%)")
            if len(rbi_events) > 10:
                logger.info(f"    ... and {len(rbi_events) - 10} more events")
        else:
            logger.warning("  WARNING: No RBI events found!")
            logger.warning("  Check if column 28 contains the changes")
        
        return clean_df
    
    def load_stock_data(self, exclude_stocks: List[str] = None) -> Dict[str, pd.DataFrame]:
        """Load stock price data as separate DataFrames per security."""
        logger.info(f"Loading stock data from {self.stock_data_sheet}")
        
        if exclude_stocks is None:
            exclude_stocks = ['NIFTY50', 'NIFTY 50']
        
        df = pd.read_excel(self.excel_path, sheet_name=self.stock_data_sheet)
        stock_dfs = {}
        
        col_idx = 0
        while col_idx < len(df.columns) - 2:
            stock_name = str(df.columns[col_idx]).strip()
            
            if 'Unnamed' in stock_name or stock_name == 'Date':
                col_idx += 3
                continue
                
            if any(exclude.upper() in stock_name.upper() for exclude in exclude_stocks):
                logger.info(f"Excluding {stock_name} (present in external indicators)")
                col_idx += 3
                continue
            
            date_col = df.iloc[1:, col_idx]
            close_col = df.iloc[1:, col_idx + 1]
            
            dates = []
            for d in date_col:
                try:
                    if isinstance(d, (int, float)) and not pd.isna(d):
                        dates.append(pd.Timestamp('1899-12-30') + pd.Timedelta(days=int(d)))
                    else:
                        dates.append(pd.to_datetime(d, errors='coerce'))
                except:
                    dates.append(pd.NaT)
            
            valid_dates = [d for d in dates if pd.notna(d)]
            if len(valid_dates) >= 252:  # Minimum 1 year
                temp_df = pd.DataFrame({
                    'date': dates,
                    'close': pd.to_numeric(close_col, errors='coerce').values
                })
                temp_df = temp_df[temp_df['date'].notna()].copy()
                temp_df = temp_df.drop_duplicates(subset=['date'], keep='first')
                temp_df.set_index('date', inplace=True)
                temp_df.sort_index(inplace=True)
                temp_df = temp_df.ffill(limit=5)
                temp_df.dropna(inplace=True)
                
                stock_dfs[stock_name] = temp_df
                logger.info(f"  {stock_name}: {len(temp_df)} dates, "
                          f"from {temp_df.index.min().date()} to {temp_df.index.max().date()}")
            
            col_idx += 3
        
        logger.info(f"[OK] Loaded {len(stock_dfs)} stocks with sufficient data")
        return stock_dfs
    
    def align_per_security(self, stock_dfs: Dict[str, pd.DataFrame],
                          indicators: pd.DataFrame) -> Dict[str, Dict]:
        """
        Align each security with indicators from its OWN first trading date.
        Each stock gets its full historical range with matching indicators.
        """
        logger.info("="*60)
        logger.info("PER-SECURITY DATE ALIGNMENT")
        logger.info("Each stock uses its full available history!")
        logger.info("="*60)
        
        aligned_data = {}
        
        for stock_name, stock_df in stock_dfs.items():
            first_date = stock_df.index.min()
            last_date = stock_df.index.max()
            
            # Get indicators from THIS stock's first date onwards
            indicators_for_stock = indicators.loc[first_date:]
            
            # Find common dates
            common_dates = stock_df.index.intersection(indicators_for_stock.index)
            
            if len(common_dates) < 252:
                logger.warning(f"Skipping {stock_name}: only {len(common_dates)} common dates")
                continue
            
            aligned_data[stock_name] = {
                'prices': stock_df.loc[common_dates, 'close'],
                'indicators': indicators_for_stock.loc[common_dates],
                'first_date': common_dates.min(),
                'last_date': common_dates.max(),
                'n_days': len(common_dates)
            }
            
            logger.info(f"  {stock_name:20s}: {len(common_dates):4d} days, "
                       f"{common_dates.min().date()} to {common_dates.max().date()}")
        
        logger.info("="*60)
        logger.info(f"[OK] Aligned {len(aligned_data)} securities with per-security dates")
        
        return aligned_data
    
    def load_all_data(self) -> Dict:
        """
        Load all data with per-security alignment.
        Returns dict with per_security_data structure.
        """
        try:
            stock_dfs = self.load_stock_data(exclude_stocks=['NIFTY50', 'NIFTY 50'])
            indicators = self.load_external_indicators()
            aligned_data = self.align_per_security(stock_dfs, indicators)
            
            if len(aligned_data) == 0:
                raise ValueError("No securities with sufficient aligned data")
            
            return {
                'per_security_data': aligned_data,
                'full_indicators': indicators,
                'stock_names': list(aligned_data.keys())
            }
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise