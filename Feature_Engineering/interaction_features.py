import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class InteractionFeaturesGenerator:
    def __init__(self, data: pd.DataFrame):
        """
        Initialize the InteractionFeaturesGenerator class with a pandas DataFrame.

        Parameters:
        data (pd.DataFrame): The input DataFrame containing the data to generate interaction features from.
        """
        self.data = data

    def create_interaction_features(self, columns: list) -> pd.DataFrame:
        """
        Create interaction features for specified columns in the DataFrame.

        Parameters:
        columns (list): The columns to generate interaction features for.

        Returns:
        pd.DataFrame: DataFrame with additional columns for the interaction features.
        """
        try:
            for i in range(len(columns)):
                for j in range(i + 1, len(columns)):
                    col1, col2 = columns[i], columns[j]
                    self.data[f'{col1}_x_{col2}'] = self.data[col1] * self.data[col2]
            return self.data

        except KeyError as e:
            print(f"Key error occurred: {e}")
            return self.data
        except ValueError as e:
            print(f"Value error occurred: {e}")
            return self.data
        except Exception as e:
            print(f"An unexpected error occurred while creating interaction features: {e}")
            return self.data

    def apply_interaction_features(self, columns: list) -> pd.DataFrame:
        """
        Apply interaction feature generation to specified columns.

        Parameters:
        columns (list): The columns to generate interaction features for.

        Returns:
        pd.DataFrame: DataFrame with additional columns for the interaction features.
        """
        return self.create_interaction_features(columns)

    """
    Error Handling:
    - Ensure that the columns specified exist in the DataFrame.
    - Handle cases where the interaction feature generation fails due to invalid input.
    """

"""
def test_interaction_features():
    # Create a small demo dataframe
    data = {
        'x1': [1, 2, 3, 4, 5],
        'x2': [5, 4, 3, 2, 1]
    }
    df = pd.DataFrame(data)

    # Instantiate the interaction features generator class
    interaction_gen = InteractionFeaturesGenerator(df)

    # Apply interaction feature generation
    interaction_df = interaction_gen.apply_interaction_features(columns=['x1', 'x2'])

    # Check if the interaction features are created
    assert 'x1_x_x2' in interaction_df.columns, "Interaction feature x1_x_x2 not found in the DataFrame."

    print(interaction_df.head())

    print("All tests passed.")

# Run the test
test_interaction_features()
"""
