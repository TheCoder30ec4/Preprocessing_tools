import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class PolynomialFeaturesGenerator:
    def __init__(self, data: pd.DataFrame):
        """
        Initialize the PolynomialFeaturesGenerator class with a pandas DataFrame.

        Parameters:
        data (pd.DataFrame): The input DataFrame containing the data to generate polynomial features from.
        """
        self.data = data

    def create_polynomial_features(self, columns: list, degree: int) -> pd.DataFrame:
        """
        Create polynomial features for specified columns in the DataFrame.

        Parameters:
        columns (list): The columns to generate polynomial features for.
        degree (int): The degree of the polynomial features.

        Returns:
        pd.DataFrame: DataFrame with additional columns for the polynomial features.
        """
        try:
            for col in columns:
                for d in range(2, degree + 1):
                    self.data[f'{col}^{d}'] = self.data[col] ** d
            return self.data

        except KeyError as e:
            print(f"Key error occurred: {e}")
            return self.data
        except ValueError as e:
            print(f"Value error occurred: {e}")
            return self.data
        except Exception as e:
            print(f"An unexpected error occurred while creating polynomial features: {e}")
            return self.data

    def apply_polynomial_features(self, columns: list, degree: int) -> pd.DataFrame:
        """
        Apply polynomial feature generation to specified columns.

        Parameters:
        columns (list): The columns to generate polynomial features for.
        degree (int): The degree of the polynomial features.

        Returns:
        pd.DataFrame: DataFrame with additional columns for the polynomial features.
        """
        return self.create_polynomial_features(columns, degree)

    """
    Error Handling:
    - Ensure that the columns specified exist in the DataFrame.
    - Handle cases where the polynomial feature generation fails due to invalid input.
    """

"""
def test_polynomial_features():
    # Create a small demo dataframe
    data = {
        'x1': [1, 2, 3, 4, 5],
        'x2': [6, 7, 8, 9, 10]
    }
    df = pd.DataFrame(data)

    # Instantiate the polynomial features generator class
    poly_gen = PolynomialFeaturesGenerator(df)

    # Apply polynomial feature generation
    poly_df = poly_gen.apply_polynomial_features(columns=['x1', 'x2'], degree=3)

    # Check if the polynomial features are created
    assert 'x1^2' in poly_df.columns, "Polynomial feature x1^2 not found in the DataFrame."
    assert 'x2^2' in poly_df.columns, "Polynomial feature x2^2 not found in the DataFrame."

    print(poly_df.head())

    print("All tests passed.")

# Run the test
test_polynomial_features()
"""