import lightning as L

from torch_geometric.nn import GATConv
from dataset import GraphDataset

import torch
from torch import nn

class GAT(nn.Module):
    def __init__(self, in_dim, hidden, num_heads):
        super(GAT, self).__init__()
        self.conv1 = GATConv(in_dim, hidden//num_heads, num_heads)
        self.act = nn.ReLU()
        self.conv2 = GATConv(hidden, 1, 1)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.act(x)
        x = self.conv2(x, edge_index)
        return self.act(x)

class Model(L.LightningModule):
    def __init__(self, in_dim, out_dim, num_heads):
        super(Model, self).__init__()
        self.model = GAT(in_dim, out_dim, num_heads)

    def forward(self, x, edge_index):
        return self.model(x, edge_index)

    def training_step(self, batch, batch_idx):
        x, edge_index = batch.x, batch.edge_index
        y_hat = self(x, edge_index)
        y = batch.y
        loss = nn.L1Loss()(y_hat[batch.mask], y[batch.mask])
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, edge_index = batch.x, batch.edge_index
        y_hat = self(x, edge_index)
        y = batch.y
        loss = nn.L1Loss()(y_hat, y)
        self.log('val_loss', loss, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=1e-3)

if __name__ == "__main__":
    from torch_geometric.loader import DataLoader
    dataset = GraphDataset()
    hidden = 256
    model = Model(dataset.dim, hidden, 4)
    train_loader = DataLoader(dataset, batch_size=1, shuffle=True)

    trainer = L.Trainer(max_epochs=10, accelerator="auto")
    trainer.fit(model=model,
                train_dataloaders=train_loader,
                val_dataloaders=DataLoader(GraphDataset(), batch_size=1, shuffle=False),
                )

    trainer.test(cfg_path="best", test_dataloaders=DataLoader(GraphDataset(), batch_size=1, shuffle=False))
