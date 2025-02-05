from tqdm.auto import tqdm
from train_test_functions import train_step, test_step

import torch
from torch import nn

device = "cuda" if torch.cuda.is_available() else "cpu"

# 1. Create a train function that takes in various model parameters + optimizer + dataloaders + loss function
def train(model: torch.nn.Module,
          train_dataloader,
          test_dataloader,
          optimizer,
          loss_fn: torch.nn.Module,
          epochs: int, 
          device=device):
    """
    Train a PyTorch model, and return the results of the training.

    Parameters
    ----------
    model : torch.nn.Module
        The PyTorch model to be trained.
    train_dataloader : torch.utils.data.DataLoader
        The dataloader containing the training data.
    test_dataloader : torch.utils.data.DataLoader
        The dataloader containing the testing data.
    optimizer : torch.optim.Optimizer
        The optimizer to use for training.
    loss_fn : torch.nn.Module, optional
        The loss function to use.
    epochs : int, optional
        The number of epochs to train for.
    device : torch.device, optional
        The device to use for training. Defaults to CUDA if available.

    Returns
    -------
    results : dict
        A dictionary containing the results of the training, with the following keys:
        - train_loss: a list of the training loss at each epoch
        - train_acc: a list of the training accuracy at each epoch
        - test_loss: a list of the testing loss at each epoch
        - test_acc: a list of the testing accuracy at each epoch
    """
  
    # 2. Create empty results dictionary
    results = {"train_loss": [],
               "train_acc": [],
               "test_loss": [],
               "test_acc": []}
  
    # 3. Loop through training and testing steps for a number of epochs
    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(model=model,
                                           dataloader=train_dataloader,
                                           loss_fn=loss_fn,
                                           optimizer=optimizer,
                                           device=device)
        test_loss, test_acc = test_step(model=model,
                                        dataloader=test_dataloader,
                                        loss_fn=loss_fn,
                                        device=device)
        
        # 4. Print out what's happening
        print(f"Epoch: {epoch} | Train loss: {train_loss:.4f} | Train acc: {train_acc:.4f} | Test loss: {test_loss:.4f} | Test acc: {test_acc:.4f}")

        # 5. Update results dictionary
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)
  
    # 6. Return the filled results at the end of the epochs
    return results
