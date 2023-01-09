import logging
import os
import pickle
import hydra
import torch
from model import MyAwesomeModel
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader, Dataset

log = logging.getLogger(__name__)


class dataset(Dataset):
    def __init__(self, images, labels):
        self.data = images.view(-1, 1, 28, 28)
        self.labels = labels

    def __getitem__(self, item):
        return self.data[item].float(), self.labels[item]

    def __len__(self):
        return len(self.data)


@hydra.main(config_path="conf", config_name="config.yaml")
def train(cfg):
    log.info("Training day and night")
    model_hparams = cfg.model
    training_hparams = cfg.training

    log.info(training_hparams.hyperparameters.lr)
    torch.manual_seed(training_hparams.hyperparameters.seed)

    model = MyAwesomeModel(model_hparams.hyperparameters.bb_hidden_channels)

    checkpoint_callback = ModelCheckpoint(
        dirpath="./models", monitor="train_loss", mode="min"
    )

    early_stopping_callback = EarlyStopping(
        monitor="train_loss", patience=10, verbose=True, mode="min"
    )

    trainer = Trainer(
        devices=1,
        accelerator="gpu",
        max_epochs=training_hparams.hyperparameters.epochs,
        limit_train_batches=1.0,
        callbacks=[checkpoint_callback, early_stopping_callback],
        logger=WandbLogger(project="kristianrm"),
        precision=16,
    )

    device = "cuda" if training_hparams.hyperparameters.cuda else "cpu"
    log.info(f"device: {device}")

    with open(training_hparams.hyperparameters.train_data_path, "rb") as handle:
        raw_data = pickle.load(handle)

    data = dataset(raw_data["images"], raw_data["labels"])
    train_loader = DataLoader(
        data, batch_size=training_hparams.hyperparameters.batch_size
    )
    trainer.fit(model, train_dataloaders=train_loader)
    torch.save(model, f"{os.getcwd()}/trained_model.pt")


if __name__ == "__main__":
    train()
