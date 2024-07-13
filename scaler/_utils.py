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