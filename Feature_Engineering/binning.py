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
        except Exception as e:
            print(f"An error occurred while creating bins: {e}")
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


