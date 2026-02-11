# model.py
import numpy as np
import torch
import torch.nn as nn
import torch_geometric.nn as pyg_nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from tqdm import tqdm


# ---------------- GAT model module ----------------
class AdvancedGAT(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels, final_out_dim, heads, num_layers, dropout):
        super(AdvancedGAT, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.final_out_dim = final_out_dim
        self.num_layers = num_layers
        self.heads = heads
        self.dropout = dropout

        # Define GAT layers
        self.gat_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        # Input layer
        self.gat_layers.append(GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout))
        self.batch_norms.append(nn.BatchNorm1d(hidden_channels * heads))

        # Hidden layers
        for _ in range(num_layers - 1):
            self.gat_layers.append(GATConv(hidden_channels * heads, hidden_channels, heads=heads, dropout=dropout))
            self.batch_norms.append(nn.BatchNorm1d(hidden_channels * heads))

        # Output layer
        self.gat_layers.append(GATConv(hidden_channels * heads, final_out_dim, heads=1, dropout=dropout))

        # Linear layer to adjust residual for the final layer to match input and output dimensions
        self.residual_layer = nn.Linear(in_channels, final_out_dim)

    def forward(self, data):
        # Get data
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        residual = x  # save input features for residual connection

        # Iterate through GAT layers
        for i, layer in enumerate(self.gat_layers):
            x = layer(x, edge_index, edge_attr)
            if i < self.num_layers - 1:  # apply activation and BatchNorm before the final layer
                x = F.leaky_relu(x, negative_slope=0.1)  # use LeakyReLU
                x = self.batch_norms[i](x)
            x = F.dropout(x, p=self.dropout, training=self.training)  # Dropout

        # Residual connection for the final layer
        # Adjust residual input dimension to match output
        residual = self.residual_layer(residual)

        # Final layer output
        x += residual  # residual connection

        return x


# ---------------- Training module ----------------
# Early stopping class
class EarlyStopping:
    def __init__(self, patience=10, delta=0, path="best_model.pth"):
        self.patience = patience  # number of epochs to tolerate without improvement
        self.delta = delta  # only consider improvement if loss decreases by more than delta
        self.best_loss = np.inf  # best training loss
        self.counter = 0  # counter for epochs without improvement
        self.path = path  # path to save the best model

    def __call__(self, train_loss, model):
        if train_loss < self.best_loss - self.delta:
            self.best_loss = train_loss
            self.counter = 0
            torch.save(model.state_dict(), self.path)  # save best model
        else:
            self.counter += 1

        if self.counter >= self.patience:
            return True
        return False


def train_gat(model, data, optimizer, scheduler=None, epochs=100, device='cuda', patience=5):
    model.to(device)
    model.train()
    data = data.to(device)  # move data to specified device (GPU or CPU)

    early_stopping = EarlyStopping(patience=patience, path="best_model.pth")

    print(f"Start training GAT model, data: {data.num_nodes} nodes, {data.edge_index.shape[1]} edges")

    for epoch in range(epochs):
        optimizer.zero_grad()  # zero gradients

        # forward pass
        out = model(data)  # forward using modified forward method with PyG Data

        # compute training loss
        loss = F.mse_loss(out, data.x)  # data.x is node features

        # backward pass
        loss.backward()

        # gradient clipping (if needed)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # prevent gradient explosion

        optimizer.step()  # update parameters

        # adjust learning rate
        if scheduler is not None:
            scheduler.step()

        # get current learning rate
        current_lr = optimizer.param_groups[0]['lr']

        # output current training info
        tqdm.write(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}, Learning Rate: {current_lr:.6f}")

        # early stopping based on training loss
        if early_stopping(loss.item(), model):
            print(f"Early stopping triggered at epoch {epoch + 1}.")
            model.load_state_dict(torch.load("best_model.pth"))  # load best model
            break  # stop training