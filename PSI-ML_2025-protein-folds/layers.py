import torch
from torch import nn
from torch.nn import Parameter


class GraphConv(nn.Module):
    """
    Graph Convolution Layer according to (T. Kipf and M. Welling, ICLR 2017)
    """
    def __init__(self,
                 in_features,
                 out_features,
                 activation=None,
                 bias=True):

        super().__init__()
        self.weight = Parameter(torch.Tensor(in_features, out_features).float(), requires_grad=True)
        if bias:
            self.bias = Parameter(torch.Tensor(out_features).float(), requires_grad=True)
        else:
            self.register_parameter('bias', None)

        self.activation = activation
        self.reset_parameters()
        self.out_features = out_features

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, inp):
        A, X = inp
        batch, N = A.shape[:2]

        A_hat = A
        D_hat = (torch.sum(A_hat, 1) + 1e-6) ** (-0.5)
        A_hat = D_hat.view(batch, N, 1) * A_hat * D_hat.view(batch, 1, N)
        x = torch.bmm(A_hat, X)
        x = torch.matmul(x, self.weight)

        if self.bias is not None:
            x = x + self.bias

        if self.activation is not None:
            x = self.activation(x)

        return x