import torch

from tqdm.auto import tqdm
from typing import Dict, List, Tuple

def train_step(model: torch.nn.Module, 
               dataloader: torch.utils.data.DataLoader, 
               loss_fn: torch.nn.Module, 
               optimizer: torch.optim.Optimizer,
               device:torch.device) -> Tuple[float, float]:
  # Put model in train mode
  model.train()

  train_loss, train_acc = 0, 0

  # Loop through dataloaders
  for X, y in dataloader:
      # Move the data to gpu if available
      X=X.to(device)
      y=y.to(device)
      # 1. Forward pass
      y_pred = model(X)

      # 2. Calculate  and accumulate loss across all batches
      loss = loss_fn(y_pred, y)
      train_loss += loss.item() 

      # 3. Optimizer zero grad
      optimizer.zero_grad()

      # 4. Loss backward
      loss.backward()

      # 5. Optimizer step
      optimizer.step()

      # Calculate and accumulate accuracy across all batches
      y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
      train_acc += (y_pred_class == y).sum().item()/len(y_pred)

  # Get average loss and accuracy per batch 
  train_loss = train_loss / len(dataloader)
  train_acc = train_acc / len(dataloader)
  return train_loss, train_acc




def test_step(model: torch.nn.Module, 
              dataloader: torch.utils.data.DataLoader, 
              loss_fn: torch.nn.Module,
              device:torch.device) -> Tuple[float, float]:
  # Put model in eval mode
  model.eval() 

  test_loss, test_acc = 0, 0

  # Turn on inference mode
  with torch.inference_mode():
      # Loop through dataLoaders
      for X,y in dataloader:
          # Move the data to gpu if available
          X=X.to(device)
          y=y.to(device)
          # 1. Forward pass
          test_pred = model(X)

          # 2. Calculate and accumulate loss across all batchess
          loss = loss_fn(test_pred, y)
          test_loss += loss.item()

          # Calculate and accumulate accuracy across all batches
          test_pred_labels = torch.argmax(torch.softmax(test_pred, dim=1),dim=1)           
          test_acc += ((test_pred_labels == y).sum().item()/len(test_pred_labels))

  # Get average loss and accuracy per batch 
  test_loss = test_loss / len(dataloader)
  test_acc = test_acc / len(dataloader)
  return test_loss, test_acc

def train(model: torch.nn.Module, 
          train_dataloader: torch.utils.data.DataLoader, 
          test_dataloader: torch.utils.data.DataLoader, 
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module,
          epochs: int,
          device:torch.device):

  # Loop through training and testing steps for a number of epochs
  for epoch in tqdm(range(epochs)):
      train_loss, train_acc = train_step(model=model,
                                          dataloader=train_dataloader,
                                          loss_fn=loss_fn,
                                          optimizer=optimizer,
                                          device=device)
      test_loss, test_acc = test_step(model=model,
          dataloader=test_dataloader,
          loss_fn=loss_fn,
          device=device
          )

      # Print out what's happening
      print(f"Epoch: {epoch+1} | train_loss: {train_loss:.4f} | train_acc: {train_acc:.4f} | test_loss: {test_loss:.4f} | test_acc: {test_acc:.4f}")

