import numpy as np
import pandas as pd

class ScalerMixin:
    def _read_csv(self, path):
        try:
            X = pd.read_csv(path)
            if X.empty:
                raise ValueError("The CSV file is empty.")
            return X
        except FileNotFoundError:
            raise FileNotFoundError(f"The file at {path} was not found.")
        except pd.errors.EmptyDataError:
            raise ValueError("The CSV file is empty.")
        except Exception as e:
            raise ValueError(f"Error reading CSV file: {str(e)}")

    def fit_csv(self, path):
        X = self._read_csv(path)
        return self.fit(X)

    def transform_csv(self, path):
        X = self._read_csv(path)
        return self.transform(X)

    def fit_transform_csv(self, path):
        X = self._read_csv(path)
        return self.fit_transform(X)


class MinMaxScaler(ScalerMixin):
    def __init__(self, feature_range=(0, 1)):
        if len(feature_range) != 2 or feature_range[0] >= feature_range[1]:
            raise ValueError("feature_range must be a tuple of two values where the first is less than the second")
        self.feature_range = feature_range
        self.min_ = None
        self.scale_ = None

    def fit(self, X):
        X = self._validate_data(X)
        self.min_ = np.nanmin(X, axis=0)
        self.scale_ = np.nanmax(X, axis=0) - self.min_
        
        # Check for zero scale (division by zero)
        if np.any(self.scale_ == 0):
            raise ValueError("One or more features have zero variance, which would lead to division by zero")
        
        return self

    def transform(self, X):
        if self.min_ is None or self.scale_ is None:
            raise ValueError("Scaler has not been fitted. Call 'fit' before using 'transform'.")
        
        X = self._validate_data(X)
        X_scaled = (X - self.min_) / self.scale_
        X_scaled = X_scaled * (self.feature_range[1] - self.feature_range[0]) + self.feature_range[0]
        return X_scaled

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def _validate_data(self, X):
        if isinstance(X, pd.DataFrame):
            numeric_columns = X.select_dtypes(include=[np.number]).columns
            if len(numeric_columns) == 0:
                raise ValueError("No numeric columns found in the DataFrame")
            if len(numeric_columns) != len(X.columns):
                print("WARNING! Not all columns in the DataFrame are numeric. Non-numeric columns will be skipped.")
            arr = X[numeric_columns].to_numpy().astype('float64')
        elif isinstance(X, np.ndarray):
            arr = X.astype('float64')
        else:
            try:
                arr = np.array(X).astype('float64')
            except ValueError:
                raise ValueError("Input data must be convertible to a numpy array of float64")

        if arr.size == 0:
            raise ValueError("Input data cannot be empty")

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



class StandardScaler(ScalerMixin):
    def __init__(self):
        self.mean_ = None
        self.scale_ = None

    def fit(self, X):
        X = self._validate_data(X)
        self.mean_ = np.nanmean(X, axis=0)
        self.scale_ = np.nanstd(X, axis=0)
        
        if np.any(self.scale_ == 0):
            raise ValueError("One or more features have zero variance, which would lead to division by zero")
        
        return self

    def transform(self, X):
        if self.mean_ is None or self.scale_ is None:
            raise ValueError("Scaler has not been fitted. Call 'fit' before using 'transform'.")

        X = self._validate_data(X)
        X_scaled = (X - self.mean_) / self.scale_
        return X_scaled

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def _validate_data(self, X):
        if isinstance(X, pd.DataFrame):
            numeric_columns = X.select_dtypes(include=[np.number]).columns
            if len(numeric_columns) == 0:
                raise ValueError("No numeric columns found in the DataFrame")
            if len(numeric_columns) != len(X.columns):
                print("WARNING! Not all columns in the DataFrame are numeric. Non-numeric columns will be skipped.")
            arr = X[numeric_columns].to_numpy().astype('float64')
        elif isinstance(X, np.ndarray):
            arr = X.astype('float64')
        else:
            try:
                arr = np.array(X).astype('float64')
            except ValueError:
                raise ValueError("Input data must be convertible to a numpy array of float64")

        if arr.size == 0:
            raise ValueError("Input data cannot be empty")

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


class MaxAbsScaler(ScalerMixin):
    def __init__(self):
        self.max_abs_ = None

    def fit(self, X):
        X = self._validate_data(X)
        self.max_abs_ = np.nanmax(np.abs(X), axis=0)
        
        if np.any(self.max_abs_ == 0):
            raise ValueError("One or more features have all zero values, which would lead to division by zero")
        
        return self

    def transform(self, X):
        X = self._validate_data(X)
        
        if self.max_abs_ is None:
            raise ValueError("Scaler has not been fitted. Call 'fit' before using 'transform'.")
        
        X_scaled = X / self.max_abs_
        return X_scaled

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def _validate_data(self, X):
        if isinstance(X, pd.DataFrame):
            numeric_columns = X.select_dtypes(include=[np.number]).columns
            if len(numeric_columns) == 0:
                raise ValueError("No numeric columns found in the DataFrame")
            if len(numeric_columns) != len(X.columns):
                print("WARNING! Not all columns in the DataFrame are numeric. Non-numeric columns will be skipped.")
            arr = X[numeric_columns].to_numpy().astype('float64')
        elif isinstance(X, np.ndarray):
            arr = X.astype('float64')
        else:
            try:
                arr = np.array(X).astype('float64')
            except ValueError:
                raise ValueError("Input data must be convertible to a numpy array of float64")

        if arr.size == 0:
            raise ValueError("Input data cannot be empty")

        return arr

class RobustScaler(ScalerMixin):
    def __init__(self, quantile_range=(25.0, 75.0)):
        self.quantile_range = quantile_range
        self.center_ = None
        self.scale_ = None

    def fit(self, X):
        X = self._validate_data(X)
        q_min, q_max = self.quantile_range
        if not 0 <= q_min < q_max <= 100:
            raise ValueError("Invalid quantile range: {}".format(self.quantile_range))

        self.center_ = np.nanmedian(X, axis=0)
        q = np.nanpercentile(X, self.quantile_range, axis=0)
        self.scale_ = (q[1] - q[0])
        
        # Check for features with zero interquartile range
        zero_scale = self.scale_ == 0
        if np.any(zero_scale):
            warnings.warn("Features with zero interquartile range detected. These features will not be scaled.")
            self.scale_[zero_scale] = 1.0

        return self

    def transform(self, X):
        X = self._validate_data(X)
        
        if self.center_ is None or self.scale_ is None:
            raise ValueError("RobustScaler has not been fitted. Call 'fit' before using 'transform'.")
        
        X_scaled = (X - self.center_) / self.scale_
        return X_scaled

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def _validate_data(self, X):
        if isinstance(X, pd.DataFrame):
            numeric_columns = X.select_dtypes(include=[np.number]).columns
            if len(numeric_columns) == 0:
                raise ValueError("No numeric columns found in the DataFrame")
            if len(numeric_columns) != len(X.columns):
                print("WARNING! Not all columns in the DataFrame are numeric. Non-numeric columns will be skipped.")
            arr = X[numeric_columns].to_numpy().astype('float64')
        elif isinstance(X, np.ndarray):
            arr = X.astype('float64')
        else:
            try:
                arr = np.array(X).astype('float64')
            except ValueError:
                raise ValueError("Input data must be convertible to a numpy array of float64")

        if arr.size == 0:
            raise ValueError("Input data cannot be empty")

        return arr