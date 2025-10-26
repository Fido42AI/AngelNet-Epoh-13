import pytest

pytest.importorskip("torch")
import torch

from universal_transformer import UniversalTransformer


def test_transform_and_decode_roundtrip(tmp_path):
    transformer = UniversalTransformer(output_dim=6, archive_dir=tmp_path)
    transformer.add_data_type("image", input_dim=4)

    data = torch.randn(3, 4)
    vector = transformer.transform(data, data_type="image")

    assert vector.shape == (3, 6)

    decoded = transformer.decode(vector, data_type="image")
    assert decoded.shape == (3, 4)
