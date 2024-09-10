import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def train_step(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               accuracy_fn,
               device: torch.device = device):
    '''
    Performs a training with model trying to learn on data_loader

    Args:
        model (torch.nn.Module): Model to be trained.
        data_loader (torch.utils.data.DataLoader): Data loader containing data to train on.
        loss_fn (torch.nn.Module): Loss function to use for training.
        optimizer (torch.optim.Optimizer): Optimizer to use for training.
        accuracy_fn (callable): Function to calculate accuracy.
        device (torch.device, optional): Device to use for training. Defaults to torch.device('cuda:0' if torch.cuda.is_available() else 'cpu').

    Returns:
        tuple: A tuple of two values. The first is the average loss over all batches, and the second is the average accuracy over all batches.
    '''
    train_loss, train_acc = 0, 0
    
    # put model into training mode
    model.train()
    
    # add a loop to loop through the training batches
    for batch, (X, y) in enumerate(data_loader):    
        # put data on target device
        X, y = X.to(device), y.to(device)
        # 1. Forward pass
        y_pred = model(X)
        # 2. Calculate loss and acc (per batch)
        loss = loss_fn(y_pred, y)
        train_loss += loss # accumulate train loss
        train_acc += accuracy_fn(y_true=y, 
                                 y_pred=y_pred.argmax(dim=1))
        
        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step
        optimizer.step()
            
    # divide total train loss and accuracy by length of train dataloader
    train_loss /= len(data_loader)
    train_acc /= len(data_loader)
    
    print(f'Train Loss: {train_loss:.5f} | Train Acc: {train_acc:.2f}%')
    
def test_step(model: torch.nn.Module,
              data_loader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              accuracy_fn,
              device: torch.device):
    '''Performs a testing loop step on model going over data_loader

    Args:
        model (torch.nn.Module): The model to be tested.
        data_loader (torch.utils.data.DataLoader): Data loader containing data to test on.
        loss_fn (torch.nn.Module): Loss function to use for testing.
        accuracy_fn (callable): Function to calculate accuracy.
        device (torch.device, optional): Device to use for testing. Defaults to torch.device('cuda:0' if torch.cuda.is_available() else 'cpu').

    Returns:
        tuple: A tuple of two values. The first is the average loss over all batches, and the second is the average accuracy over all batches.
    '''
    test_loss, test_acc = 0, 0
    # put the model in eval mode
    model.eval()
    
    # turn on inference mode context manager
    with torch.inference_mode():
        for X, y in data_loader:
            # send the data to target device
            X, y = X.to(device), y.to(device)
            
            # 1. Forward pass
            test_pred = model(X)
            
            # 2. Calculate loss and acc
            test_loss += loss_fn(test_pred, y)
            test_acc += accuracy_fn(y_true=y, 
                                    y_pred=test_pred.argmax(dim=1))
            
        # adjust the metrics and print out
        test_loss /= len(data_loader)
        test_acc /= len(data_loader)
        
        # print out what's happening
        print(f'\nTest Loss: {test_loss:.5f} | Test Acc: {test_acc:.2f}%\n')    
