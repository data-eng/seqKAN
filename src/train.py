import torch
from torch.utils.data import DataLoader
import time
import matplotlib.pyplot as plt


def train(model,
          train_dataset: torch.utils.data.Dataset,
          val_dataset: torch.utils.data.Dataset,
          dir_path: str,
          train_params: dict,
          device: torch.device,
          use_reg: bool):
    """
    Trains a DL model using early stopping
    :param model: Instance of DL model
    :param train_dataset: Training Dataset in PyTorch standard format
    :param val_dataset: Validation Dataset in PyTorch standard format
    :param dir_path: Path to store the model and the training/validation loss plot
    :param train_params: Training parameters (batch size, #epochs, eta, reg. eta, patience, loss function, optimizer)
    :param device: CPU or GPU
    :param use_reg: Train with regularization à la KAN
    :return:
    """

    batch_size = train_params['batch_size']
    epochs = train_params['epochs']
    lr = train_params['lr']
    lamb = train_params['lamb']
    patience = train_params['patience']

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    criterion = train_params['loss_fun']()
    optimizer = train_params['optimizer'](model.parameters(), lr=lr)

    best_val_loss = float('inf')
    patience_counter = 0

    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        model.train()

        running_train_loss = 0.0

        start_time = time.time()

        for sequences, targets in train_loader:
            sequences, targets = sequences.to(device), targets.to(device)

            outputs = model(sequences)

            loss = criterion(outputs.squeeze(), targets.squeeze())

            if use_reg:
                reg_ = model.KANlayer.get_reg('edge_forward_spline_n', 1.0, 2.0, 1.0, 1.0) + \
                       model.KANoutput.get_reg('edge_forward_spline_n', 1.0, 2.0, 1.0, 1.0)
            else:
                reg_ = 0.0

            loss_ = loss + lamb * reg_

            optimizer.zero_grad()
            loss_.backward()
            optimizer.step()
            running_train_loss += loss.item() * sequences.size(0)

        end_time = time.time()
        epoch_train_loss = running_train_loss / len(train_loader.dataset)
        train_losses.append(epoch_train_loss)

        model.eval()
        running_val_loss = 0.0

        with torch.no_grad():
            for sequences, targets in val_loader:
                sequences, targets = sequences.to(device), targets.to(device)
                outputs = model(sequences)
                loss = criterion(outputs.squeeze(), targets.squeeze())
                running_val_loss += loss.item() * sequences.size(0)

        epoch_val_loss = running_val_loss / len(val_loader.dataset)
        val_losses.append(epoch_val_loss)

        print(f"Epoch [{epoch + 1}/{epochs}], Train Loss: {epoch_train_loss:.4f}, "
              f"Validation Loss: {epoch_val_loss:.4f}, "
              f"Time: {(end_time - start_time) / 60:.2f} mins")

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), f"{dir_path}/model.pth")
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print("Early stopping triggered")
            break

    print('Plotting training and validation losses.')
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(f"{dir_path}/losses.png", dpi=600)
    plt.close()
