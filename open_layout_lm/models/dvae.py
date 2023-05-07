import torch
from torch.nn.functional import gumbel_softmax


class DVAE(torch.nn.Module):
    """The discrete variational autoencoder.

    TODO: input image normalization
    TODO: bottleneck style encoder (like in the original DALL-E)
    Based on: https://github.com/lucidrains/DALLE-pytorch/blob/main/dalle_pytorch/dalle_pytorch.py

    References:
    (1) Zero-Shot Text-to-Image Generation
    (2) Neural Discrete Representation Learning
    """
    def __init__(
        self,
        *,
        encoder: torch.nn.Module,
        decoder: torch.nn.Module,
        temperature: float = 0.9,
        codebook_size: int = 512,
        codebook_vector_dim: int = 512,
    ) -> None:
        """Init the DVAE.

        Args:
            encoder: The encoder used to encode the image. The dimensionality of its output
                should match the dimensionality of the decoder's input. The shape of the encoder's
                output should be: `(batch_size, codebook_size, width, height)`.
            decoder: The decoder used to decode the image given a matrix of latent variables.
                The dimensionality of its input should match the dimensionality
                of the encoder's input.
            temperature: The temperature parameter for Gumbel Softmax sampling.
            codebook_size: Number of vectors in the codebook.
            codebook_vector_dim: Dimensionality of the codebook's vector.
        """
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.temperature = temperature
        self.codebook = torch.nn.Embedding(codebook_size, codebook_vector_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode the input tensor.

        This function is an alias of `self.encode`.

        Args:
            x: The tensor to be encoded. It should be of shape `(batch, ..., self.num_tokens)`.
            `"..."` could mean any number of dimensions the encoder and decoder support
            (for example width and height dimensions for 2D images).

        Returns:
            The input encoded with the encoder. The resulting encoding is a "soft encoding",
            meaning that it is an output of a Gumbel-Softmax function, not a set of
            exact codebook vectors.
        """
        return self.encode(x)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode the input tensor.

        This function is an alias of `self.encode`.

        Args:
            x: The tensor to be encoded. It should be of shape
            `(batch, width, height, self.codebook_size)`.

        Returns:
            The input encoded with the encoder. The resulting encoding is a "soft encoding",
            meaning that it is an output of a Gumbel-Softmax function, not a set of
            exact codebook vectors. The shape of the output is
            `(batch, width, height, self.codebook_vector_dim)`
        """
        logits = self.encoder(x).permute((0, 2, 3, 1))
        # (batch, width, height, self.codebook_size)
        soft_one_hot = gumbel_softmax(logits, dim=-1, tau=self.temperature, hard=False)

        return torch.matmul(
            soft_one_hot.unsqueeze(-2),
            self.codebook.weight
        ).squeeze(-2)

    def decode(self, z: torch.Tensor, *, from_indices: bool = False) -> torch.Tensor:
        """Reconstruct the input from latent variables.

        Args:
            z: The latent variables. Should be of shape: `(batch, width, height,
                codebook_vector_dim)`.
            from_indices: If `True`, `z` consists of codebook indices. Otherwise, `z`
                consists of codebook vectors (either actual exact vectors from the codebook
                or vectors sampled "softly" using Gumbel-Softmax relaxation).

        Returns:
            The reconstructed input (for example a batch of images).
        """
        if from_indices:
            z = self.codebook(z)
        return self.decoder(z.permute((0, 3, 1, 2)))


class ResidualBlock(torch.nn.Module):
    """A Conv2D block with skip connections.

    A single ResidualBlock module computes the following:
        `y = relu(x + norm(conv(relu(norm(conv(x))))))`
    where `x` is the input, `y` is the output, `norm` is a 2D batch norm and `conv` is
    a 2D convolution with kernel of size 3, stride 1 and padding 1.
    """
    def __init__(self, in_channels: int, out_channels: int):
        """Init the ResidualBlock.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
        """
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm1 = torch.nn.BatchNorm2d(num_features=out_channels)
        self.norm2 = torch.nn.BatchNorm2d(num_features=out_channels)
        self.relu = torch.nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the forward pass of the ResidualBlock.

        Args:
            x: The input.

        Returns:
            The output after applying the residual block. See the class description
            for more details.
        """
        y = self.conv1(x)
        y = self.norm1(y)
        y = self.relu(y)
        y = self.conv2(y)
        y = self.norm2(y)
        return self.relu(y + x)


class Conv2DEncoder(torch.nn.Module):
    """An image encoder based on 2D convolutions.

    Based on: https://github.com/lucidrains/DALLE-pytorch/blob/main/dalle_pytorch/dalle_pytorch.py
    """
    def __init__(
        self,
        *,
        input_channels: int = 3,
        output_channels: int = 512,
        hidden_size: int = 64,
        num_layers: int = 3,
        num_resnet_blocks: int = 1,
    ):
        """Init the encoder.

        Args:
            input_channels: Number of input channels.
            output_channels: Number of the output channels.
            hidden_size: Number of channels in the intermediate hidden layers.
            num_layers: Number of hidden layers.
            num_resnet_blocks: Number of resnet blocks added after each layer.
        """
        super().__init__()
        layers_list: list[torch.nn.Module] = [
            torch.nn.Conv2d(input_channels, hidden_size, kernel_size=1)
        ]
        for _ in range(num_layers):
            layers_list.append(
                torch.nn.Conv2d(hidden_size, hidden_size, kernel_size=4, stride=2, padding=1)
            )
            layers_list.append(torch.nn.ReLU())
            layers_list.extend(
                [ResidualBlock(hidden_size, hidden_size) for _ in range(num_resnet_blocks)]
            )

        layers_list.append(torch.nn.Conv2d(hidden_size, output_channels, kernel_size=1))
        self.layers = torch.nn.Sequential(*layers_list)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode an image.

        Args:
            x: The input image of shape `(batch, input_channels, in_width, in_height)`

        Returns:
            The encoder image of shape `(batch, output_channels, out_width, out_height)`
        """
        return self.layers(x)


class Conv2DDecoder(torch.nn.Module):
    """An image decoder based on 2D transposed convolutions.

    Based on: https://github.com/lucidrains/DALLE-pytorch/blob/main/dalle_pytorch/dalle_pytorch.py
    """
    def __init__(
        self,
        *,
        input_channels: int = 512,
        output_channels: int = 3,
        hidden_size: int = 64,
        num_layers: int = 3,
        num_resnet_blocks: int = 1,
    ):
        """Init the encoder.

        Args:
            input_channels: Number of input channels (dimensionality of the codebook).
            output_channels: Number of output channels.
            hidden_size: Number of channels in the intermediate hidden layers.
            num_layers: Number of hidden layers.
            num_resnet_blocks: Number of resnet blocks added after each layer.
        """
        super().__init__()
        layers_list: list[torch.nn.Module] = [
            torch.nn.ConvTranspose2d(input_channels, hidden_size, kernel_size=1)
        ]
        for _ in range(num_layers):
            layers_list.append(
                torch.nn.ConvTranspose2d(
                    hidden_size, hidden_size, kernel_size=4, stride=2, padding=1
                )
            )
            layers_list.append(torch.nn.ReLU())
            layers_list.extend(
                [ResidualBlock(hidden_size, hidden_size) for _ in range(num_resnet_blocks)]
            )

        layers_list.append(torch.nn.ConvTranspose2d(hidden_size, output_channels, kernel_size=1))
        self.layers = torch.nn.Sequential(*layers_list)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Encode an image given latent variables.

        Args:
            z: The input image of shape `(batch, input_channels, in_width, in_height)`.

        Returns:
            The encoder image of shape `(batch, output_channels, out_width, out_height)`.
        """
        return self.layers(z)
