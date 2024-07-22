import pandas as pd
import numpy as np


class DataTransformation:
    """
    A class used to perform various data transformation techniques.

    Methods
    -------
    normalize_data(data, method)
        Normalize the data using the specified method.

    standardize_data(data)
        Standardize the data to have a mean of 0 and a standard deviation of 1.
    """

    def __init__(self):
        """
        Initialize the DataTransformation class.
        """
        pass  # Currently, there's no initialization needed

    def normalize_data(self, data: pd.DataFrame, method: str) -> pd.DataFrame:
        """
        Normalize the data using the specified method.

        Parameters:
        data (pd.DataFrame): The input data to be normalized.
        method (str): The normalization method to use ('Standard', 'MinMax', 'MaxAbsoluteScaling', 'Log', 'Power').

        Returns:
        pd.DataFrame: The normalized data.

        Raises:
        ValueError: If an invalid method is provided or if data contains non-numeric values.
        """
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Input data must be a pandas DataFrame.")

        if data.isnull().values.any():
            raise ValueError("Input data contains missing values. Please handle missing data before normalization.")

        if not all(data.dtypes.apply(lambda x: np.issubdtype(x, np.number))):
            raise ValueError("All columns in the input data must be numeric.")

        normalized_data = pd.DataFrame()

        if method == 'Standard':
            normalized_data = (data - data.mean()) / data.std()
        elif method == 'MinMax':
            normalized_data = (data - data.min()) / (data.max() - data.min())
        elif method == 'MaxAbsoluteScaling':
            normalized_data = data / data.abs().max()
        elif method == 'Log':
            if (data <= 0).any().any():
                raise ValueError("Log transformation requires all values to be positive.")
            normalized_data = np.log1p(data)
        elif method == 'Power':
            if (data < 0).any().any():
                raise ValueError("Power transformation requires all values to be non-negative.")
            normalized_data = np.power(data, 0.5)
        else:
            raise ValueError('Method must be "Standard", "MinMax", "MaxAbsoluteScaling", "Log", or "Power"')

        return normalized_data

    def standardize_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize the data to have a mean of 0 and a standard deviation of 1.

        Parameters:
        data (pd.DataFrame): The input data to be standardized.

        Returns:
        pd.DataFrame: The standardized data.

        Raises:
        ValueError: If data contains non-numeric values.
        """
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Input data must be a pandas DataFrame.")

        if data.isnull().values.any():
            raise ValueError("Input data contains missing values. Please handle missing data before standardization.")

        if not all(data.dtypes.apply(lambda x: np.issubdtype(x, np.number))):
            raise ValueError("All columns in the input data must be numeric.")

        standardized_data = (data - data.mean()) / data.std()
        return standardized_data