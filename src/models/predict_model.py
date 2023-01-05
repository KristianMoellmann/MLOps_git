import pickle

import click
import torch
from model import MyAwesomeModel
from torch.utils.data import DataLoader, Dataset


class dataset(Dataset):
    def __init__(self, images, labels):
        self.data = images.view(-1, 1, 28, 28)
        self.labels = labels

    def __getitem__(self, item):
        return self.data[item].float(), self.labels[item]

    def __len__(self):
        return len(self.data)


@click.command()
@click.argument('model_filepath', type=click.Path(exists=True))
@click.argument('test_filepath', type=click.Path(exists=True))
def evaluate(model_filepath, test_filepath):
    print("Evaluating model")

    model = MyAwesomeModel()
    model.load_state_dict(torch.load(model_filepath))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    with open(test_filepath, 'rb') as handle:
        raw_data = pickle.load(handle)

    data = dataset(raw_data['images'], raw_data['labels'])
    dataloader = DataLoader(data, batch_size=64)

    correct, total = 0, 0
    for batch in dataloader:
        x, y = batch

        preds = model(x.to(device))
        preds = preds.argmax(dim=-1)

        correct += (preds == y.to(device)).sum().item()
        total += y.numel()

    print(f"Test set accuracy {correct / total}")


if __name__ == "__main__":
    evaluate()
