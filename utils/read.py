import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler

class TimeSeriesDataset:
    def __init__(self, lookback, features, target_feature):
        """
        Initializes the TimeSeriesDataset object.

        Args:
            lookback (int): The number of time steps to look back for features (x).
            features (list): A list of column names to include in the feature vectors (x).
            target_feature (str): The column name to use as the target variable (y).
        """
        self.lookback = lookback
        self.features = features
        self.target_feature = target_feature
        self.feature_scalers = {}  # Store scalers for each feature
        self.target_scaler = None  # Store scaler for the target feature

    def load_data_from_file(self, filename):
        """
        Searches for a file in specified folders, reads its data, and returns it as a DataFrame.

        Args:
            filename (str): The name of the file (without path) to search for.

        Returns:
            pandas.DataFrame: The data from the file as a DataFrame, or None if the file is not found.
        """
        base_dir = os.path.expanduser("~/Documents/STOCKS/ds0")
        folders = [
            os.path.join(base_dir, 'ETFs'),
            os.path.join(base_dir, 'Stocks'),
            os.path.join(base_dir, 'Data', 'ETFs'),
            os.path.join(base_dir, 'Data', 'Stocks')
        ]

        for folder in folders:
            filepath = os.path.join(folder, filename)
            if os.path.exists(filepath):
                try:
                    df = pd.read_csv(filepath, header=None, names=['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'OpenInt'])
                    return df
                except Exception as e:
                    print(f"Error reading {filepath}: {e}")
                    return None
        print(f"File '{filename}' not found in any of the specified folders.")
        return None

    def create_dataset(self, filename):
        """
        Creates the dataset with x and y from the given file.

        Args:
            filename (str): The name of the file to load data from.

        Returns:
            tuple: A tuple containing x (NumPy array of feature vectors) and y (NumPy array of target values), or None if data loading fails.
        """
        df = self.load_data_from_file(filename)
        if df is None:
            return None

        # Convert selected features to numeric, handling errors by coercing to NaN
        df[self.features] = df[self.features].apply(pd.to_numeric, errors='coerce')
        df[self.target_feature] = pd.to_numeric(df[self.target_feature], errors='coerce')
        df = df.dropna(subset=self.features + [self.target_feature])

        x_data = []
        y_data = []

        for i in range(len(df) - self.lookback):
            x_data.append(df[self.features].iloc[i:i + self.lookback].values)
            y_data.append(df[self.target_feature].iloc[i + self.lookback])

        return np.array(x_data), np.array(y_data).reshape((-1, 1)), df # Return the original dataframe as well

    def fit_scalers(self, df):
        """Fits scalers to the data and stores them."""
        for feature in self.features:
            scaler = MinMaxScaler(feature_range=(-1, 1))
            scaler.fit(df[feature].values.reshape(-1, 1))
            self.feature_scalers[feature] = scaler

        self.target_scaler = MinMaxScaler(feature_range=(-1, 1))
        self.target_scaler.fit(df[self.target_feature].values.reshape(-1, 1))

    def transform(self, x, y):
        """Transforms the x and y data using the fitted scalers."""
        x_transformed = []
        for sample in x:
            transformed_sample = []
            for i, feature in enumerate(self.features):
                scaler = self.feature_scalers[feature]
                transformed_sample.append(scaler.transform(sample[:, i].reshape(-1, 1)).flatten())
            x_transformed.append(np.array(transformed_sample).T)

        y_transformed = self.target_scaler.transform(y.reshape(-1, 1)).flatten()
        return np.array(x_transformed), y_transformed.reshape((-1, 1))

    def inverse_transform(self, x_transformed, y_transformed):
         """Inverse transforms the x and y data using the fitted scalers."""
         x_original = []
         for sample in x_transformed:
            original_sample = []
            for i, feature in enumerate(self.features):
                scaler = self.feature_scalers[feature]
                original_sample.append(scaler.inverse_transform(sample[:, i].reshape(-1, 1)).flatten())
            x_original.append(np.array(original_sample).T)

         y_original = self.target_scaler.inverse_transform(y_transformed.reshape(-1, 1)).flatten()
         return np.array(x_original), y_original.reshape((-1, 1))


# Example Usage:
if __name__ == "__main__":
    lookback = 10  # Number of time steps to look back
    features = ['Open', 'High', 'Low', 'Volume']  # Features to include in x
    target_feature = 'Close'  # Target feature for y
    filename = "aal.us.txt"  # File to load data from

    dataset_creator = TimeSeriesDataset(lookback, features, target_feature)
    x, y, df = dataset_creator.create_dataset(filename)

    if x is not None and y is not None:
        print("Original Shape of x:", x.shape)
        print("Original Shape of y:", y.shape)
        print("First 1 x samples:")
        print(x[0])
        print("First 1 y samples:")
        print(y[0])

        # Fit scalers
        dataset_creator.fit_scalers(df)

        # Transform data
        x_transformed, y_transformed = dataset_creator.transform(x, y)
        print("\nTransformed Shape of x:", x_transformed.shape)
        print("Transformed Shape of y:", y_transformed.shape)
        print("First 1 transformed x samples:")
        print(x_transformed[0])
        print("First 1 transformed y samples:")
        print(y_transformed[0])


        # Inverse transform data
        x_original, y_original = dataset_creator.inverse_transform(x_transformed, y_transformed)
        print("\nInverse Transformed Shape of x:", x_original.shape)
        print("Inverse Transformed Shape of y:", y_original.shape)
        print("First 1 inverse transformed x samples:")
        print(x_original[0])
        print("First 1 inverse transformed y samples:")
        print(y_original[0])


    else:
        print(f"Failed to create dataset for {filename}")