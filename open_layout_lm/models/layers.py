import torch


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
