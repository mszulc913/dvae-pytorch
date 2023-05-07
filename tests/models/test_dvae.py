import pytest
import torch.nn

from open_layout_lm.models.dvae import DVAE, Conv2DDecoder, Conv2DEncoder


class MockModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return 2 * x


def test_dvae_encode() -> None:
    dvae = DVAE(encoder=MockModel(), decoder=MockModel(), codebook_size=8, codebook_vector_dim=16)
    x = torch.ones((5, 8, 4, 4))

    x_encoded = dvae.encode(x)

    assert x_encoded.shape == (5, 4, 4, 16)
    assert x_encoded.isnan().any().tolist() is False


def test_dvae_decode() -> None:
    dvae = DVAE(encoder=MockModel(), decoder=MockModel(), codebook_size=8, codebook_vector_dim=16)
    z = torch.ones((5, 4, 4, 16))

    z_decoded = dvae.decode(z)

    assert z_decoded.shape == (5, 16, 4, 4)
    assert z_decoded.unique() == torch.tensor([2.])


def test_dvae_decode_from_indices() -> None:
    dvae = DVAE(encoder=MockModel(), decoder=MockModel(), codebook_size=8, codebook_vector_dim=16)
    z = torch.ones((5, 4, 4), dtype=torch.long)

    z_decoded = dvae.decode(z, from_indices=True)

    assert z_decoded.shape == (5, 16, 4, 4)
    assert z_decoded.isnan().any().tolist() is False


@pytest.mark.parametrize(
    ("num_layers", "num_resnet_blocks", "expected_shape"),
    [
        (0, 0, (5, 256, 128, 128)),
        (0, 1, (5, 256, 128, 128)),
        (2, 0, (5, 256, 32, 32)),
        (2, 1, (5, 256, 32, 32))
    ]
)
def test_encoder(num_layers: int, num_resnet_blocks: int, expected_shape: tuple[int, ...]) -> None:
    encoder = Conv2DEncoder(
        num_layers=num_layers,
        num_resnet_blocks=num_resnet_blocks,
        input_channels=3,
        output_channels=256
    )
    x = torch.ones((5, 3, 128, 128))

    x_encoded = encoder(x)

    assert x_encoded.shape == expected_shape
    assert x_encoded.isnan().any().tolist() is False


@pytest.mark.parametrize(
    ("num_layers", "num_resnet_blocks", "expected_shape"),
    [
        (0, 0, (5, 3, 32, 32)),
        (0, 1, (5, 3, 32, 32)),
        (2, 0, (5, 3, 128, 128)),
        (2, 1, (5, 3, 128, 128))
    ]
)
def test_decoder(num_layers: int, num_resnet_blocks: int, expected_shape: tuple[int, ...]) -> None:
    decoder = Conv2DDecoder(
        num_layers=num_layers,
        num_resnet_blocks=num_resnet_blocks,
        input_channels=256,
        output_channels=3
    )
    x = torch.ones((5, 256, 32, 32))

    encoded_img = decoder(x)

    assert encoded_img.shape == expected_shape
