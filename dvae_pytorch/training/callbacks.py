import pytorch_lightning as pl
import torch
from pytorch_lightning import Callback
from torch.utils.data import Dataset
from torchvision.utils import make_grid

from dvae_pytorch.training.lightning import DVAETrainModule


class SaveVisualizationCallback(Callback):
    """Callback to save visualizations of a dataset throughout training.

    The callback will save the first `n_images` images of the dataset.
    """

    def __init__(self, n_images: int, log_every_n_epochs: int, dataset: Dataset) -> None:
        """Init the callback.

        Args:
            n_images: The number of images to save.
            log_every_n_epochs: Log the visualization every `log_every_n_epochs` epochs.
            dataset: The dataset to visualize.
        """
        self._n_images = n_images
        self._log_every_n_epochs = log_every_n_epochs
        self._dataset = dataset

    def on_validation_epoch_end(  # noqa: D102
        self, _: pl.Trainer, pl_module: DVAETrainModule
    ) -> None:
        if pl_module.logger is None:
            raise ValueError("Logger is not set.")
        if pl_module.current_epoch % self._log_every_n_epochs == 0:
            img = torch.stack([self._dataset[i] for i in range(self._n_images)], dim=0).to(
                pl_module.device  # type: ignore[arg-type]
            )
            img_encoded = pl_module.dvae.encode(img, hard=True)[0]
            img_decoded = pl_module.dvae.decode(img_encoded).detach()
            grid = make_grid(torch.concat([img, img_decoded], dim=0), nrow=self._n_images)

            pl_module.logger.experiment.add_image("val", grid, pl_module.current_epoch)
