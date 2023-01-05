from torch import nn


class MyAwesomeModel(nn.Module):
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

    def forward(self, x):
        return self.classifier(self.backbone(x))
