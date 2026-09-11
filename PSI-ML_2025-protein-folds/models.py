from layers import GraphConv
import torch
import torch.nn as nn

import pytorch_lightning as pl
from torcheval.metrics.functional import multiclass_accuracy

class GraphNN(nn.Module):
    def __init__(self, in_features=24, out_features=6, hidden_features=64):
        super(GraphNN, self).__init__()
        c_hid1 = hidden_features//2
        c_hid2 = hidden_features
        c_hid3 = hidden_features*2
        self.gcn_1 = GraphConv(in_features=in_features, out_features=c_hid1, activation=nn.ReLU())
        self.gcn_2 = GraphConv(in_features=c_hid1, out_features=c_hid2, activation=nn.ReLU())
        self.gcn_3 = GraphConv(in_features=c_hid2, out_features=c_hid3, activation=nn.ReLU())
        self.proj = nn.Linear(c_hid1+c_hid2+c_hid3, out_features)

    def forward(self, inp):
        out_1 = self.gcn_1(inp)
        out_2 = self.gcn_2((inp[0], out_1))
        out_3 = self.gcn_3((inp[0], out_2))
        out = torch.cat([out_1, out_2, out_3], dim=-1)
        out = torch.sum(out, dim=1)
        out = self.proj(out)
        return out

class SeqCNN(nn.Module):

    def __init__(self, in_features=24, out_features=6, hidden_features=64, ksizes=[15, 9, 5], gru=False):
        super(SeqCNN, self).__init__()
        # We increase the hidden dimension over layers. Here pre-calculated for simplicity.
        self.gru = gru
        c_hid1 = hidden_features//2
        c_hid2 = hidden_features
        c_hid3 = hidden_features*2

        self.conv_layers = nn.Sequential(
                nn.Conv1d(in_features, c_hid1, kernel_size=ksizes[0], stride=1, padding=0),
                nn.ReLU(),
                nn.Conv1d(c_hid1, c_hid2, kernel_size=ksizes[1], stride=1, padding=0),
                nn.ReLU(),
                nn.Conv1d(c_hid2, c_hid3, kernel_size=ksizes[2], stride=1, padding=0),
                nn.ReLU()
                )

        if self.gru:
            self.gru_layer = nn.GRU(c_hid3, c_hid3, 1, bidirectional=False, batch_first=True)

        self.ffd = nn.Sequential(
                nn.Linear(c_hid3, c_hid2),
                nn.ReLU(),
                nn.Linear(c_hid2, out_features)
        )

    def forward(self, x):
        # permute before applying conv
        h = self.conv_layers(x.permute(0, 2, 1))  # requires (batch_size, n_channels, L)

        if self.gru:
            h, _ = self.gru_layer(h.permute(0, 2, 1)) # requires (batch_size, L, n_channels)

            # max pooling
            # h = torch.max(h, dim=1)[0]
            h = torch.mean(h, dim=1)
        else:
            # max pooling
            # h = torch.max(h, dim=-1)[0]
            h = torch.mean(h, dim=-1)

        # projection to a single node
        h = self.ffd(h).squeeze(dim=-1)

        return h


class ProtFoldClassifier(pl.LightningModule):
    def __init__(self, vocab_size=24, num_classes=6, hidden_dim=64, lr=2e-4, weight_decay=0.0, gnn=False):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.weight_decay = weight_decay
        self.gnn = gnn
        self.num_classes = num_classes

        if self.gnn:
            self.classifier = GraphNN(in_features=vocab_size, out_features=num_classes, hidden_features=hidden_dim)
        else:
            self.classifier = SeqCNN(in_features=vocab_size, out_features=num_classes, hidden_features=hidden_dim)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x):
        y_hat = self.classifier(x)
        return y_hat

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        acc = multiclass_accuracy(torch.argmax(y_hat, dim=-1), y, average='macro', num_classes=self.num_classes)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_acc', acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        acc = multiclass_accuracy(torch.argmax(y_hat, dim=-1), y, average='macro', num_classes=self.num_classes)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_acc', acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
