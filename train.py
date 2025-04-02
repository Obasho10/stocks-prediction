import torch
import torch.nn as nn

def train_one_epoch(model,train_loader,device,epoch,loss_function,optimizer):
    model.train(True)
    running_loss = 0.0

    for batch_index, batch in enumerate(train_loader):
        x_batch, y_batch = batch[0].to(device), batch[1].to(device)

        output = model(x_batch.float())
        loss = loss_function(output.float(), y_batch.float())
        running_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        '''if batch_index % 100 == 99:  # print every 100 batches
            avg_loss_across_batches = running_loss / 100
            print('Batch {0}, Loss: {1:.8f}'.format(batch_index+1,
                                                    avg_loss_across_batches))
            running_loss = 0.0'''
        return running_loss
    
def validate_one_epoch(model,test_loader,device,loss_function):
    model.train(False)
    running_loss = 0.0

    for batch_index, batch in enumerate(test_loader):
        x_batch, y_batch = batch[0].to(device), batch[1].to(device)

        with torch.no_grad():
            output = model(x_batch.float())
            loss = loss_function(output, y_batch)
            running_loss += loss.item()

    avg_loss_across_batches = running_loss / len(test_loader)

    '''print('Val Loss: {0:.8f}'.format(avg_loss_across_batches))
    print('***************************************************')
    print()'''
    return running_loss
    
    
def tra(model,train_loader,test_loader,device,loss_function):
    learning_rate = 0.0001
    num_epochs = 100
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        train_one_epoch(model,train_loader,device,epoch,loss_function,optimizer)
        validate_one_epoch(model,test_loader,device,loss_function)
    learning_rate = 0.0000001
    num_epochs = 200
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        train_one_epoch(model,train_loader,device,epoch,loss_function,optimizer)
        validate_one_epoch(model,test_loader,device,loss_function)
    return model