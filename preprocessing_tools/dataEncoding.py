import pandas as pd

class DataEncoding:

    
    def label_encode(self, data: pd.DataFrame, columns: list) -> pd.DataFrame: 
        
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Input data must be a pandas DataFrame.")
        
        for column in columns:
            if column not in data.columns:
                raise ValueError(f"Column '{column}' is not in the DataFrame.")
            data[column] = data[column].astype('category').cat.codes
        
        return data
    

    def one_hot_encode(self, data: pd.DataFrame, columns: list) -> pd.DataFrame:
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Input data must be a pandas DataFrame.")
    
        data = pd.get_dummies(data, columns=columns)
        data = data.astype(int)
    
        return data


    def ordinal_encode(self, data: pd.DataFrame, columns: list, categories: dict) -> pd.DataFrame:
        
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Input data must be a pandas DataFrame.")
        
        for column in columns:
            if column not in data.columns:
                raise ValueError(f"Column '{column}' is not in the DataFrame.")
            if column not in categories:
                raise ValueError(f"Categories for column '{column}' must be provided.")
            
            category_mapping = {category: index for index, category in enumerate(categories[column])}
            data[column] = data[column].map(category_mapping)
        
        return data
