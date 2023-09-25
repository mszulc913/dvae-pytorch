import numpy as np
import pytorch_lightning as pl
import torch
from matplotlib import pyplot as plt
from pytorch_lightning import Callback
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import Dataset
from torchvision.utils import make_grid

from dvae_pytorch.training.lightning import DVAETrainModule


class SaveValVisualizationCallback(Callback):
    """Callback to save visualizations of a dataset throughout training.

    The callback will save
    - the first `n_images` images of the dataset auto-encoded by the model
    - the codebook vectors distribution.
    """

    def __init__(
        self, n_images: int, log_every_n_epochs: int, dataset: Dataset, logger: TensorBoardLogger
    ) -> None:
        """Init the callback.

        Args:
            n_images: The number of images to save.
            log_every_n_epochs: Log the visualization every `log_every_n_epochs` epochs.
            dataset: The dataset to visualize.
            logger: The logger to be used to save the visualizations.
        """
        self._n_images = n_images
        self._log_every_n_epochs = log_every_n_epochs
        self._dataset = dataset
        self._logger = logger

    def on_validation_epoch_end(  # noqa: D102
        self, _: pl.Trainer, pl_module: DVAETrainModule
    ) -> None:
        if pl_module.logger is None:
            raise ValueError("Logger is not set.")
        if pl_module.current_epoch % self._log_every_n_epochs == 0:
            self._plot_predictions(pl_module)
            self._plot_codebook_distribution(pl_module)

    def _plot_codebook_distribution(self, pl_module: DVAETrainModule) -> None:
        fig, ax = plt.subplots()
        y = pl_module.current_distribution.detach().cpu().numpy()
        x = range(y.shape[-1])
        ax.bar(x, y)
        ax.set_ylim(bottom=0.0, top=1.0)
        ax.set_title("Codebook distribution")
        fig.canvas.draw()
        plot_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(
            *reversed(fig.canvas.get_width_height()), 3
        )
        self._logger.experiment.add_image(
            "val/codebook_distribution",
            plot_array,
            dataformats="HWC",
            global_step=pl_module.current_epoch,
        )

    def _plot_predictions(self, pl_module: DVAETrainModule) -> None:
        img = torch.stack([self._dataset[i] for i in range(self._n_images)], dim=0).to(
            pl_module.device  # type: ignore[arg-type]
        )
        img_encoded = pl_module.dvae.encode(img, hard=True)[0]
        img_decoded = pl_module.dvae.decode(img_encoded).detach()
        grid = make_grid(torch.concat([img, img_decoded], dim=0), nrow=self._n_images)
        self._logger.experiment.add_image("val/predictions", grid, pl_module.current_epoch)
