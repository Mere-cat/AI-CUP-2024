import lightning as L

from torch_geometric.nn import GATConv, SuperGATConv
from dataset_old import GraphDataset

import torch
from torch import nn
try:
    torch.set_float32_matmul_precision('medium')
except:
    pass

eps = 1e-8

class GAT(nn.Module):
    def __init__(self, in_dim, hidden, num_heads, p=0.5):
        super(GAT, self).__init__()
        self.conv1 = SuperGATConv(in_dim, hidden//num_heads, num_heads)
        self.conv2 = SuperGATConv(hidden, hidden//num_heads, num_heads)
        self.fc = nn.Linear(hidden, 1)

        self.act = nn.GELU()
        self.outact = nn.ReLU()
        self.dropout = torch.nn.Dropout(p)

    def forward(self, x, edge_index):
        x = self.dropout(self.act(self.conv1(x, edge_index)))
        x = self.dropout(self.act(self.conv2(x, edge_index)))
        return self.fc(x)

class Model(L.LightningModule):
    def __init__(self, in_dim, out_dim, num_heads):
        super(Model, self).__init__()
        self.model = GAT(in_dim, out_dim, num_heads)

    def forward(self, x, edge_index):
        return self.model(x, edge_index)

    def training_step(self, batch, batch_idx):
        x, edge_index = batch.x, batch.edge_index
        y_hat = self(x, edge_index).reshape(-1)
        y = batch.y

        loss = nn.MSELoss()(y_hat[batch.mask], y[batch.mask])
        self.log('train_loss', torch.round(loss, decimals=5), prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, edge_index = batch.x, batch.edge_index
        y_hat = self(x, edge_index).reshape(-1)
        y = batch.y
        loss = nn.MSELoss()(y_hat, y)

        self.log('val_loss', torch.round(loss, decimals=5))

    def test_step(self, batch, batch_idx):
        x, edge_index = batch.x, batch.edge_index
        y_hat = self(x, edge_index).reshape(-1)
        y = batch.y

        loss = nn.MSELoss()(y_hat, y)
        self.log('test_loss', torch.round(loss, decimals=5))

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=0.0001)

if __name__ == "__main__":
    from torch_geometric.loader import DataLoader
    from lightning.pytorch.callbacks import RichProgressBar, ModelCheckpoint
    dataset = GraphDataset()
    hidden = 256
    model = Model(dataset.dim, hidden, 4)
    train_loader = DataLoader(dataset, batch_size=1, num_workers=0, shuffle=True)


    trainer = L.Trainer(max_epochs=1,
                        precision="bf16-mixed",
                        accumulate_grad_batches=10000,
                        callbacks=[ModelCheckpoint(save_top_k=1, monitor="val_loss"),
                                   RichProgressBar(leave=False)
                                   ],
                        accelerator="auto")

    trainer.fit(model=model,
                train_dataloaders=train_loader,
                val_dataloaders=DataLoader(GraphDataset(), batch_size=1, num_workers=0, shuffle=False),
                )

    trainer.test(ckpt_path="best", dataloaders=DataLoader(GraphDataset(), batch_size=1, num_workers=0, shuffle=False))
