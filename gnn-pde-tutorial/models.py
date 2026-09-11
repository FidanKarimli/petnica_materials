"""
This file contains the models for the GNN-based models.
"""
import torch
import copy
#from torch_scatter import scatter
from typing import Optional
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as gnn
from torch_geometric.nn import GCNConv, GATConv


##########Utility Functions##########

class Activation(nn.Module):
    """
        Parameters:
        -----------
            x: torch.FloatTensor
                input tensor
        Returns:
        --------
            y: torch.FloatTensor
                output tensor, same shape as the input tensor, since it's element-wise operation
    """
    def __init__(self, activation:str):
        super().__init__()
        activation = activation.lower() # prevent potential typo
        if activation in ['sigmoid', 'tanh']:
            # prevent potential warning message
            self.activation_fn = getattr(torch, activation)
        elif activation == "swish":
            self.beta = nn.Parameter(torch.ones(1), requires_grad=True)
    
        elif activation == "identity":
            self.activation_fn = lambda x: x
        else:
            self.activation_fn = getattr(F, activation)
        self.activation = activation
    def forward(self, x):
        if self.activation == "swish":
            return x * torch.sigmoid(self.beta * x)
        elif self.activation == "gelu":
            return x * torch.sigmoid(1.702 * x)
        elif self.activation == "mish":
            return x * torch.tanh(F.softplus(x))
        else:
            return self.activation_fn(x)

#---------------------------------#

def init_gnn_model(model, num_features, num_classes, kwargs):
    if model == "gcn":
        return GCN(num_features, num_classes, num_hidden=kwargs.n_hidden, num_layers=kwargs.n_layers, activation=kwargs.activation)
    elif model == "gat":
        return GAT(num_features, num_classes, num_hidden=kwargs.n_hidden, num_layers=kwargs.n_layers, num_heads=kwargs.num_heads, activation=kwargs.activation)
    else:
        raise NotImplementedError(f"Unknown model {model}")

class GCN(nn.Module):
    def __init__(self, num_features, num_classes, 
        num_hidden  = 64, num_layers  = 3, activation  = "relu"):
        super().__init__()
        self.layers     = nn.ModuleList([GCNConv(num_features, num_hidden)])
        for i in range(num_layers-2):
            self.layers.append(GCNConv(num_hidden, num_hidden))
        self.layers.append(GCNConv(num_hidden, num_classes))
        self.activation = Activation(activation)
        self.num_features = num_features
        self.num_classes  = num_classes
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, x, edge_index):
        for layer in self.layers[:-1]:
            x = self.activation(layer(x, edge_index))
        x = self.layers[-1](x, edge_index)
        return x


class GAT(nn.Module):
    def __init__(self, num_features, num_classes,
        num_hidden  = 64, num_layers  = 3, num_heads=4, activation  = "relu"):
        super().__init__()
        self.layers     = nn.ModuleList([GATConv(num_features, num_hidden//num_heads, heads=num_heads)])
        for i in range(num_layers-2):
            self.layers.append(GATConv(num_hidden, num_hidden//num_heads, heads=num_heads))
        # self.layers.append(GATConv(num_hidden, num_classes, num_heads=1))
        self.layers.append(GATConv(num_hidden, num_classes, heads=1))
        self.activation = Activation(activation)
        self.num_features = num_features
        self.num_classes  = num_classes
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, x, edge_index):
        xs = []
        for i in range(x.shape[0]):
            x_i = x[i]
            for layer in self.layers[:-1]:
                x_i = self.activation(layer(x_i, edge_index))
            x_i = self.layers[-1](x_i, edge_index)
            xs.append(x_i)
        x = torch.stack(xs, dim=0)
        
        return x