import pytest

pytest.importorskip("torch")
import torch

from angelnet_core import AngelNet


def test_forward_output_shape(tmp_path):
    model = AngelNet(input_dim=16, hidden_dim=8, output_dim=4, base_lr=0.01, storage_dir=tmp_path)
    batch = torch.randn(5, 1, 4, 4)
    labels = torch.zeros(5, dtype=torch.long)

    output = model(batch, labels=labels, data_type="image")
    assert output.shape == (5, 4)


def test_autonomous_classify_runs(tmp_path):
    model = AngelNet(input_dim=16, hidden_dim=8, output_dim=4, base_lr=0.01, storage_dir=tmp_path)
    batch = torch.randn(2, 1, 4, 4)

    predictions = model.autonomous_classify(batch, data_type="image")
    assert predictions.shape == (2,)
    assert predictions.dtype == torch.long
