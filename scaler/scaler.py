import numpy as np
import pandas as pd

class MinMaxScaler:
    def __init__(self, feature_range=(0, 1)):
        if len(feature_range) != 2 or feature_range[0] >= feature_range[1]:
            raise ValueError("feature_range must be a tuple of two values where the first is less than the second")
        self.newmin = feature_range[0]
        self.newmax = feature_range[1]

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            numeric_columns = X.select_dtypes(include=[np.number]).columns
            if len(numeric_columns) == 0:
                raise ValueError("No numeric columns found in the DataFrame")
            arr = X[numeric_columns].to_numpy().astype('float64')
        elif isinstance(X, np.ndarray):
            arr = X.astype('float64')
        else:
            try:
                arr = np.array(X).astype('float64')
            except ValueError:
                raise ValueError("Input data must be a convertible to a numpy array of float64")

        if arr.size == 0:
            raise ValueError("Input data cannot be empty")

        for col in arr.T:
            maxim = np.max(col)
            minim = np.min(col)
            diff = maxim - minim
            if diff == 0:
                raise ValueError("One of the columns has all identical values, which would lead to division by zero")
            col[:] = (col - minim) / diff * (self.newmax - self.newmin) + self.newmin

        return arr

    def transform_csv(self, path):
        try:
            X = pd.read_csv(path)
            return self.transform(X)
        except FileNotFoundError:
            raise FileNotFoundError(f"The file at {path} was not found.")
        except pd.errors.EmptyDataError:
            raise ValueError("The CSV file is empty.")
        except Exception as e:
            raise ValueError(f"Error during CSV transformation: {str(e)}")