import unittest
import pandas as pd
import numpy as np
from src.features import calculate_returns, add_technical_indicators

class TestFeatureEngineering(unittest.TestCase):
    """
    Unit tests for feature engineering functions in src/features.py.
    """
    
    def setUp(self):
        """Setup a consistent, simple DataFrame for all tests."""
        # We need a large number of days (> 200) for SMA_200 to be calculated
        dates = pd.to_datetime(pd.date_range('2024-01-01', periods=250, freq='B')) # 250 Business Days
        self.mock_prices = pd.DataFrame({
            'TEST_A': np.linspace(100, 200, 250) + np.random.randn(250) * 0.1, # Trend data
            'TEST_B': np.array([100] * 200 + [110] * 50) # Simple step for known returns
        }, index=dates)
        self.mock_prices.index.name = 'Date'

    def test_calculate_returns_alignment(self):
        """Test if target returns are calculated and aligned correctly (no data leakage)."""
        horizon = 5 # 5-day prediction
        
        # Calculate returns
        features, targets = calculate_returns(self.mock_prices, prediction_horizon=horizon)
        
        # Assertions
        self.assertIsInstance(targets, pd.DataFrame)
        self.assertIsInstance(features, pd.DataFrame)
        
        # Check that the number of dropped rows is equal to the horizon (250 initial - 5 horizon = 245 expected)
        self.assertEqual(targets.shape[0], self.mock_prices.shape[0] - horizon)
        
        # Check that the dates are aligned and not shifted wrongly
        self.assertEqual(features.index[0], self.mock_prices.index[0])
        self.assertNotIn(self.mock_prices.index[-1], targets.index)

    def test_calculate_returns_value(self):
        """Test if the calculated return value is mathematically correct for a known step."""
        horizon = 1
        
        features, targets = calculate_returns(self.mock_prices, prediction_horizon=horizon)
        
        # TEST_B goes from 100 to 100, then to 100... The 1-day return should be 0.0 before the jump (index 50)
        expected_return_value = 0.0
        
        actual_return = targets['TEST_B'].iloc[50]
        
        self.assertAlmostEqual(actual_return, expected_return_value, places=5)

    def test_add_technical_indicators_structure(self):
        """Test if the correct number of indicator columns are added per ticker."""
        
        features, _ = calculate_returns(self.mock_prices, prediction_horizon=5)
        
        # The function should drop the first 200 rows due to SMA_200.
        # 250 initial rows - 5 (horizon) = 245 rows. 
        # 245 rows - 200 (SMA window) = 45 rows expected in the final output.
        features_with_indicators = add_technical_indicators(features)
        
        # Total columns expected: Original (2) + (7 new features * 2 tickers) = 16 columns
        self.assertEqual(features_with_indicators.shape[1], 16) 
        
        # Assertion Corrigée: Nous attendons 45 lignes après le nettoyage
        self.assertEqual(features_with_indicators.shape[0], 45) 
        
        # Check if a specific column exists (RSI for TEST_A)
        self.assertIn('TEST_A_RSI', features_with_indicators.columns)
        self.assertFalse(features_with_indicators.isnull().values.any()) # Final check: no NaNs remaining
        

if __name__ == '__main__':
    unittest.main()