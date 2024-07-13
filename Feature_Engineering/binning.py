import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class Binning:
    def __init__(self, data: pd.DataFrame):
        """
        Initialize the Binning class with a pandas DataFrame.

        Parameters:
        data (pd.DataFrame): The input DataFrame containing the data to be binned.
        """
        self.data = data

    def create_bins(self, column: str, bins: int or list, labels: list = None) -> pd.DataFrame:
        """
        Create bins for a specified column in the DataFrame.

        Parameters:
        column (str): The column to bin.
        bins (int or list): The number of bins or the bin edges.
        labels (list, optional): The labels for the bins.

        Returns:
        pd.DataFrame: DataFrame with an additional column for the binned data.
        """
        try:
            self.data[f'{column}_binned'] = pd.cut(self.data[column], bins=bins, labels=labels)
            return self.data
        except ValueError as e:
            print(f"Value error occurred: {e}")
            return self.data
        except KeyError as e:
            print(f"Key error occurred: {e}")
            return self.data
        except Exception as e:
            print(f"An unexpected error occurred while creating bins: {e}")
            return self.data

    def plot_bins(self, column: str):
        """
        Plot the distribution of the binned data.

        Parameters:
        column (str): The binned column to plot.
        """
        try:
            plt.figure(figsize=(10, 6))
            sns.countplot(x=column, data=self.data)
            plt.title(f'Distribution of {column}')
            plt.xlabel(column)
            plt.ylabel('Count')
            plt.show()
        except Exception as e:
            print(f"An error occurred while plotting bins: {e}")

    def apply_binning(self, column: str, bins: int or list, labels: list = None, plot: bool = False) -> pd.DataFrame:
        """
        Apply binning to a specified column and optionally plot the distribution.

        Parameters:
        column (str): The column to bin.
        bins (int or list): The number of bins or the bin edges.
        labels (list, optional): The labels for the bins.
        plot (bool, optional): Whether to plot the distribution of the binned data.

        Returns:
        pd.DataFrame: DataFrame with an additional column for the binned data.
        """
        self.create_bins(column, bins, labels)
        if plot:
            self.plot_bins(f'{column}_binned')
        return self.data

    """
    Error Handling:
    - Ensure that the column specified exists in the DataFrame.
    - Handle cases where the binning operation fails due to invalid bin edges or labels.
    - Handle plotting errors, such as when the specified column does not exist or is not binned.
    """

"""
def test_binning():
    # Create a small demo dataframe
    data = {
        'value': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    }
    df = pd.DataFrame(data)

    # Instantiate the binning class
    binning = Binning(df)

    # Apply binning
    binned_df = binning.apply_binning(column='value', bins=3, labels=['Low', 'Medium', 'High'], plot=True)

    # Check if the binned column is created
    assert f'value_binned' in binned_df.columns, "Binned column not found in the DataFrame."

    # Check if the binned column has the correct number of bins
    assert binned_df['value_binned'].nunique() == 3, "The number of unique bins is not correct."

    print("All tests passed.")

# Run the test
test_binning()
"""
