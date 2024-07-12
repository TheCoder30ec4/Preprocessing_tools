import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer

class DataCleaner:
    def __init__(self, dataframe):
        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError("dataframe should be a pandas Dataframe")
        self.dataframe = dataframe

    def remove_col_or_row(self, axis=0, thresh=None):
        """
        Remove columns or rows with missing values.
        axis: 0 for rows, 1 for columns.
        thresh: int, require that many non-NA values.
        """
        if thresh is None:
            # Default threshold to remove rows or columns where more than 50% of values are missing
            thresh = int(self.dataframe.shape[1-axis] / 2)
        print(f"axis: {axis} thresh: {thresh}")
        self.dataframe = self.dataframe.dropna(axis=axis, thresh=thresh)
        return self.dataframe

    def fill_with_zeros(self):
        """
        Fill missing values with zeros.
        """
        self.dataframe = self.dataframe.fillna(0)
        return self.dataframe

    def fill_with_mean(self):
        """
        Fill missing values with the mean of the column.
        """
        mean = self.dataframe.mean()
        self.dataframe = self.dataframe.fillna(mean)
        return self.dataframe

    def fill_with_knn(self, n_neighbors=2):
        """
        Fill missing values using K-Nearest Neighbors (KNN) imputation
        """
        imputer = KNNImputer(missing_values=np.nan, n_neighbors=n_neighbors)
        self.dataframe = imputer.fit_transform(self.dataframe)
        return self.dataframe
