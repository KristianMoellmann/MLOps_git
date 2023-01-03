import torch
import click
import pickle
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader
from model import MyAwesomeModel


class dataset(Dataset):
    def __init__(self, images, labels):
        self.data = images.view(-1, 1, 28, 28)
        self.labels = labels

    def __getitem__(self, item):
        return self.data[item].float(), self.labels[item]

    def __len__(self):
        return len(self.data)


@click.group()
def cli():
    pass


@click.command()
@click.option("--lr", default=1e-3, help='learning rate to use for training')
def train(lr):
    print("Training day and night")
    print(lr)
    model = MyAwesomeModel()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    with open('data/processed/train.pickle', 'rb') as handle:
        raw_data = pickle.load(handle)

    data = dataset(raw_data['images'], raw_data['labels'])
    dataloader = DataLoader(data, batch_size=128)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    n_epoch = 100
    loss_tracker = []

    for epoch in range(n_epoch):
        running_loss = 0
        for batch in dataloader:
            optimizer.zero_grad()
            x, y = batch
            preds = model(x.to(device))
            loss = criterion(preds, y.to(device))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        loss_tracker.append(running_loss/len(dataloader))
        print(f"Epoch {epoch + 1}/{n_epoch}. Loss: {running_loss/len(dataloader)}")
    torch.save(model.state_dict(), 'models/trained_model.pt')

    plt.plot(loss_tracker, '-')
    plt.xlabel('Training step')
    plt.ylabel('Training loss')
    plt.savefig("reports/figures/training_curve.png")


cli.add_command(train)

if __name__ == "__main__":
    cli()
