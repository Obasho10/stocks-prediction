import os
import sys

try:
    # This will work when run as a script
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
except NameError:
    # This will work when run in an interactive environment
    sys.path.append(os.path.abspath(os.path.join(os.getcwd(), 'utils','..')))


from utils.read import TimeSeriesDataset
from torch.utils.data import DataLoader

from torch.utils.data import Dataset
import torch
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
import pandas as pd
class TimeSeriesitDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]




def loadds(filename,lookback,features = ['Open', 'High', 'Low', 'Volume'],target_feature = 'Close',batch_size = 16):
    dataset_creator = TimeSeriesDataset(lookback, features, target_feature)
    x, y, df = dataset_creator.create_dataset(filename)

    if x is not None and y is not None:
        dataset_creator.fit_scalers(df)
        x_transformed, y_transformed = dataset_creator.transform(x, y)
    else:
        print(f"Failed to create dataset for {filename}")

    split_index = int(len(x_transformed) * 0.90)
    X_train = x_transformed[:split_index]
    X_test = x_transformed[split_index:]

    y_train = y_transformed[:split_index]
    y_test = y_transformed[split_index:]

    X_train.shape, X_test.shape, y_train.shape, y_test.shape
    train_dataset = TimeSeriesitDataset(X_train, y_train)
    test_dataset = TimeSeriesitDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader,test_loader,X_train, X_test, y_train, y_test,dataset_creator