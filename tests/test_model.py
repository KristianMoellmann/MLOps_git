import pytest
import torch

from src.models.model import MyAwesomeModel


class TestModel:
    def test_model(self):
        model = MyAwesomeModel([64, 32, 16, 8])
        assert model(torch.rand(1, 1, 28, 28)).shape == (
            1,
            10,
        ), "Output of model should me shape (1, 10)"

    def test_error_on_wrong_shape(self):
        model = MyAwesomeModel([64, 32, 16, 8])
        with pytest.raises(ValueError, match="Expected input to a 4D tensor"):
            model(torch.randn(1, 2, 3))
