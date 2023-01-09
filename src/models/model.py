from pytorch_lightning import LightningModule
import torch
from torch import nn, optim

class MyAwesomeModel(LightningModule):
    def __init__(self, hidden_channels):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(1, hidden_channels[0], 3),  # [N, 64, 26]
            nn.LeakyReLU(),
            nn.Conv2d(hidden_channels[0], hidden_channels[1], 3),  # [N, 32, 24]
            nn.LeakyReLU(),
            nn.Conv2d(hidden_channels[1], hidden_channels[2], 3),  # [N, 16, 22]
            nn.LeakyReLU(),
            nn.Conv2d(hidden_channels[2], hidden_channels[3], 3),  # [N, 8, 20]
            nn.LeakyReLU()
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(8 * 20 * 20, 128),
            nn.Dropout(),
            nn.Linear(128, 10)
        )
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        if x.ndim != 4:
            raise ValueError('Expected input to a 4D tensor')
        if x.shape[1] != 1 or x.shape[2] != 28 or x.shape[3] != 28:
            raise ValueError('Expected each sample to have shape [1, 28, 28]')

        return self.classifier(self.backbone(x))

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> float:
        data, target = batch
        preds = self(data)
        loss = self.criterion(preds, target)
        acc = (target == preds.argmax(dim=-1)).float().mean()
        self.log('train_loss', loss)
        self.log('train_acc', acc)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer