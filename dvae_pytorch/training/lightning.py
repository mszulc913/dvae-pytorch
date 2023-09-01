import pytorch_lightning as pl
import torch.optim.optimizer
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR

from dvae_pytorch.models.dvae import DVAE
from dvae_pytorch.training.config import DVAETrainingConfig
from dvae_pytorch.training.params_schedule import LinearScheduler


class DVAETrainModule(pl.LightningModule):
    """A PyTorch Lightning training module for the `DVAE`."""

    def __init__(self, *, dvae: DVAE, config: DVAETrainingConfig | None = None):
        """Init the DVAE training module.

        Args:
            dvae: An instance of the Discrete Variational Auto-Encoder to be used.
            config: The DVAE's training configuration. If `None`, default configuration
                will be used.
        """
        super().__init__()
        self.dvae = dvae
        self.config = DVAETrainingConfig() if config is None else config
        self._temperature_scheduler = LinearScheduler(
            start=self.config.temperature_scheduler.start,
            end=self.config.temperature_scheduler.end,
            warmup=self.config.temperature_scheduler.warmup,
            cooldown=self.config.temperature_scheduler.cooldown,
            steps=self.config.max_epochs,
        )
        self._kl_div_weight_scheduler = LinearScheduler(
            start=self.config.kl_div_weight_scheduler.start,
            end=self.config.kl_div_weight_scheduler.end,
            warmup=self.config.kl_div_weight_scheduler.warmup,
            cooldown=self.config.kl_div_weight_scheduler.cooldown,
            steps=self.config.max_epochs,
        )
        self.save_hyperparameters(self.config.dict(), ignore=["dvae", "config"])

    def configure_optimizers(self) -> tuple[list[AdamW], list[LinearLR]]:  # noqa: D102
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.config.learning_rate)
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=self.config.learning_rate_scheduler_min,
            total_iters=self.config.max_epochs,
        )
        return [optimizer], [scheduler]

    def _step(self, batch: torch.Tensor, mode: str) -> torch.Tensor:
        temperature = self._temperature_scheduler.get_value()
        kl_div_weight = self._kl_div_weight_scheduler.get_value()

        encoded, logits = self.dvae.encode(batch, temperature=temperature)
        decoded = self.dvae.decode(encoded)

        reconstruction_loss = torch.nn.functional.mse_loss(decoded, batch)

        log_softmax = torch.log_softmax(logits, dim=-1)
        log_uniform = torch.log(torch.ones_like(log_softmax) / self.dvae.codebook_size)
        elbo_loss = torch.nn.functional.kl_div(
            log_softmax, log_uniform, log_target=True, reduction="batchmean"
        )
        loss = reconstruction_loss + kl_div_weight * elbo_loss

        self.log(f"{mode}/loss", loss, prog_bar=mode == "train")
        self.log(f"{mode}/elbo_loss", elbo_loss)
        self.log(f"{mode}/reconstruction_loss", reconstruction_loss)

        return loss

    def training_step(self, batch: torch.Tensor, _: int) -> torch.Tensor:  # noqa: D102
        return self._step(batch, "train")

    def on_train_epoch_end(self) -> None:  # noqa: D102
        self._temperature_scheduler.step()
        self._kl_div_weight_scheduler.step()

        self.log("temperature", self._temperature_scheduler.get_value())
        self.log("kl_div_weight", self._kl_div_weight_scheduler.get_value())

    def validation_step(self, batch: torch.Tensor, _: int) -> torch.Tensor:  # noqa: D102
        return self._step(batch, "val")

    def test_step(self, batch: torch.Tensor, _: int) -> None:  # noqa: D102
        self._step(batch, "test")
