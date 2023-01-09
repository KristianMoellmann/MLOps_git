import os
import pickle
import pytest
import torch
from torch.utils.data import Dataset

from tests import _PATH_DATA


class dataset(Dataset):
    def __init__(self, images, labels):
        self.data = images.view(-1, 1, 28, 28)
        self.labels = labels

    def __getitem__(self, item):
        return self.data[item].float(), self.labels[item]

    def __len__(self):
        return len(self.data)


def load_data(path):
    with open(path, "rb") as handle:
        raw_data = pickle.load(handle)
    data = dataset(raw_data["images"], raw_data["labels"])
    return data


class TestData:
    @pytest.mark.skipif(
        not os.path.exists(os.path.join(_PATH_DATA, "processed/train.pickle")),
        reason="Train data files not found",
    )
    def test_train(self):
        train_data = load_data(os.path.join(_PATH_DATA, "processed/train.pickle"))
        assert (
            len(train_data.data) == 40000
        ), "Number of training data points should be 40,000"
        assert train_data.data.shape == (
            40000,
            1,
            28,
            28,
        ), "Shape of training data should be (40000, 1, 28, 28)"
        assert (
            len(train_data.labels) == 40000
        ), "Number of training labels should be 40,000"
        assert torch.sum(torch.unique(train_data.labels)) == torch.tensor(
            45
        ), "Sum of unique labels should be 45"

    @pytest.mark.skipif(
        not os.path.exists(os.path.join(_PATH_DATA, "processed/train.pickle")),
        reason="Test data files not found",
    )
    def test_test(self):
        test_data = load_data(os.path.join(_PATH_DATA, "processed/test.pickle"))
        assert (
            len(test_data.data) == 5000
        ), "Number of training data points should be 5,000"
        assert test_data.data.shape == (
            5000,
            1,
            28,
            28,
        ), "Shape of training data should be (5000, 1, 28, 28)"
        assert (
            len(test_data.labels) == 5000
        ), "Number of training labels should be 5,000"
        assert torch.sum(torch.unique(test_data.labels)) == torch.tensor(
            45
        ), "Sum of unique labels should be 45"
