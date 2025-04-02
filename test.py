from dataloader import loadds
from models import *
from train import tra 
import torch
import torch.nn as nn

def getm(id):
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    if(id==0):
        model = LSTM(4, 128, 3,device)
        model.to(device)
    elif(id==1):
        input_size = 4  # Number of features
        hidden_size = 128  # Hidden size for attention
        num_heads = 8 # Number of attention heads
        model = AttentionModel(input_size, hidden_size, num_heads)
        model.to(device)
    return model

def trainfordifflookback(model,filename,lookback,loss_function,features = ['Open', 'High', 'Low', 'Volume'],target_feature = 'Close',batch_size = 16):
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    train_loader,test_loader,X_train, X_test, y_train, y_test,dataset_creator=loadds(filename,lookback)
    model=tra(model,train_loader,test_loader,device,loss_function)
    with torch.no_grad():
        predicted = model(torch.tensor(X_train).float().to(device)).to('cpu').numpy()
    x_originaltr, y_originaltr = dataset_creator.inverse_transform(X_train, y_train)
    x_originaltr, y_predtr = dataset_creator.inverse_transform(X_train, predicted)
    with torch.no_grad():
        predicted = model(torch.tensor(X_test).float().to(device)).to('cpu').numpy()
    x_originalts, y_originalts = dataset_creator.inverse_transform(X_test, y_test)
    x_originalts, y_predts = dataset_creator.inverse_transform(X_test, predicted)
    return x_originaltr, y_originaltr,y_predtr,x_originalts, y_originalts,y_predts
import matplotlib.pyplot as plt
import os
def plot_and_save(x, y_true, y_pred, lookback, set_type, model_id, save_dir):
    """
    Plots and saves the predictions against the true values.

    Args:
        x (np.array): Input data.
        y_true (np.array): True target values.
        y_pred (np.array): Predicted target values.
        lookback (int): The lookback window.
        set_type (str): "train" or "test".
        model_id (int): Model identifier (0 or 1).
        save_dir (str): Directory to save the plots.
    """

    num_plots = len(y_true)
    for i in range(num_plots):
        plt.figure(figsize=(10, 6))
        # Plot the last true value
        plt.plot(range(len(y_pred[i])), y_pred[i], label='Predictions', color='red')
        plt.plot(range(len(y_pred[i])), y_true[i], marker='o', label='True Value', color='blue')
        plt.xlabel('Time Step')
        plt.ylabel('Close Price')
        plt.legend()
        plt.grid(True)

        # Create the save directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)

        # Save the plot
        filename = f'{i}.png'
        filepath = os.path.join(save_dir, filename)
        plt.savefig(filepath)
        plt.close()


def main():
    filename = "sbb.us.txt"
    loss_function = nn.MSELoss()
    save_dir = "/home/obasho/Documents/STOCKS/plots"  # Directory to save plots
    xtrainl, ytrainl, ytrainpredl, xtestl, ytestl, ytestpredl=[],[],[],[],[],[]
    for mid in range(2):
        model = getm(mid)
        for lookback in range(1, 30):
            xtrain, ytrain, ytrainpred, xtest, ytest, ytestpred = trainfordifflookback(model, filename, lookback, loss_function)
            xtrainl.append(xtrain)
            ytrainl.append(ytrain)
            ytrainpredl.append(ytrainpred)
            xtestl.append(xtest)
            ytestl.append(ytest)
            ytestpredl.append(ytestpred)

    # Plot and save training set plots
    plot_and_save(xtrainl, ytrainl, ytrainpredl, lookback, "train", mid, save_dir)
    # Plot and save testing set plots
    plot_and_save(xtestl, ytestl, ytestpredl, lookback, "test", mid, save_dir)


if __name__ == "__main__":
    main()