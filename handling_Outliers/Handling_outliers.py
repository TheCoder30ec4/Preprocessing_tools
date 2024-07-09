import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


class HandlingOutliers:
    """
    A class to handle outliers in a pandas DataFrame using z-score and IQR methods.

    Attributes
    ----------
    dataFrame : pd.DataFrame
        The DataFrame containing the data.

    Methods
    -------
    z_score(column_name: str, threshold_value: tuple, plot: bool = False) -> pd.DataFrame:
        Removes outliers from the specified column using the z-score method.

    IQR(column: str, plot: bool = False) -> pd.DataFrame:
        Removes outliers from the specified column using the IQR method.
    """

    def __init__(self, dataFrame):
        """
        Constructs all the necessary attributes for the HandlingOutliers object.

        Parameters
        ----------
            dataFrame : pd.DataFrame
                The DataFrame containing the data.
        """
        if not isinstance(dataFrame, pd.DataFrame):
            raise TypeError("dataFrame should be a pandas DataFrame")
        self.dataFrame = dataFrame

    def z_score(self, column_name: str, threshold_value: tuple, plot: bool = False) -> pd.DataFrame:
        """
        Removes outliers from the specified column using the z-score method.

        Parameters
        ----------
        column_name : str
            The column name for which outliers need to be removed.
        threshold_value : tuple
            A tuple containing two numeric values for the lower and upper z-score thresholds.
        plot : bool, optional
            Whether to plot the data before and after outlier removal (default is False).

        Returns
        -------
        pd.DataFrame
            A DataFrame with outliers removed based on the z-score method.
        """
        try:
            if not isinstance(threshold_value, tuple) or len(threshold_value) != 2:
                raise ValueError("Threshold values should be a tuple containing exactly two values.")

            lower_threshold, upper_threshold = threshold_value

            if not all(isinstance(val, (int, float)) for val in threshold_value):
                raise TypeError("Both threshold values should be numeric (int or float).")

            if column_name not in self.dataFrame.columns:
                raise ValueError(f"Column {column_name} does not exist in the DataFrame.")

            # Calculate the mean and standard deviation of the specified column
            mean_val = self.dataFrame[column_name].mean()
            std_val = self.dataFrame[column_name].std()

            # Calculate the z-scores manually
            self.dataFrame['z_score'] = (self.dataFrame[column_name] - mean_val) / std_val

            # Filter the DataFrame to remove outliers
            cleaned_df = self.dataFrame[
                (self.dataFrame['z_score'] >= lower_threshold) & (self.dataFrame['z_score'] <= upper_threshold)]
            outliers_df = self.dataFrame[
                (self.dataFrame['z_score'] < lower_threshold) | (self.dataFrame['z_score'] > upper_threshold)]

            # Drop the z_score column as it's no longer needed
            cleaned_df = cleaned_df.drop(columns=['z_score'])
            outliers_df = outliers_df.drop(columns=['z_score'])

            # Plotting if requested
            if plot:
                fig, ax = plt.subplots(2, 1, figsize=(10, 12))

                # Plot the cleaned data
                ax[0].hist(cleaned_df[column_name], bins=30, edgecolor='k', alpha=0.7, label='Cleaned Data')
                ax[0].set_title(f'Histogram of {column_name} after Removing Outliers')
                ax[0].set_xlabel(column_name)
                ax[0].set_ylabel('Frequency')
                ax[0].legend()

                # Plot the outliers
                ax[1].hist(outliers_df[column_name], bins=30, edgecolor='r', alpha=0.7, label='Outliers')
                ax[1].set_title(f'Histogram of {column_name} Outliers')
                ax[1].set_xlabel(column_name)
                ax[1].set_ylabel('Frequency')
                ax[1].legend()

                plt.tight_layout()
                plt.show()

            return cleaned_df

        except Exception as e:
            print(f"An error occurred: {e}")

    def IQR(self, column: str, plot: bool = False) -> pd.DataFrame:
        """
        Removes outliers from the specified column using the IQR method.

        Parameters
        ----------
        column : str
            The column name for which outliers need to be removed.
        plot : bool, optional
            Whether to plot the data before and after outlier removal (default is False).

        Returns
        -------
        pd.DataFrame
            A DataFrame with outliers removed based on the IQR method.
        """
        try:
            if column not in self.dataFrame.columns:
                raise ValueError(f"Column {column} does not exist in the DataFrame.")

            Q1 = self.dataFrame[column].quantile(0.25)
            Q3 = self.dataFrame[column].quantile(0.75)

            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            cleaned_data = self.dataFrame[
                (self.dataFrame[column] >= lower_bound) & (self.dataFrame[column] <= upper_bound)]

            if plot:
                sns.boxplot(data=self.dataFrame[column])
                plt.title(f'Box plot of {column} (Original Data)')
                plt.show()
                sns.boxplot(data=cleaned_data[column])
                plt.title(f'Box plot of {column} (Cleaned Data)')
                plt.show()

            return cleaned_data

        except Exception as e:
            print(f"An error occurred: {e}")