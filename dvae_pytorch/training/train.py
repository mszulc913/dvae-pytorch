import argparse
import logging
from pathlib import Path

import pytorch_lightning as pl
import torch
import yaml
from pydantic import ValidationError
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10

from dvae_pytorch.datasets.wrappers import ClassificationDatasetWrapper
from dvae_pytorch.models.dvae import DVAE, Conv2DDecoder, Conv2DEncoder
from dvae_pytorch.training.callbacks import SaveVisualizationCallback
from dvae_pytorch.training.config import TrainingRunConfig
from dvae_pytorch.training.lightning import DVAETrainModule


def _main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a Discrete Variational Auto-Encoder.",
    )
    parser.add_argument("--config", type=Path, required=True, help="Config file path.")
    args = parser.parse_args()
    config_path: Path = args.config

    try:
        with args.config.open() as f:
            config_dict = yaml.safe_load(f)
            config: TrainingRunConfig = TrainingRunConfig.parse_obj(config_dict)
    except FileNotFoundError:
        logging.error(f"Config file {config_path} does not exist. Exiting.")
    except yaml.YAMLError:
        logging.error(f"Config file {config_path} is not valid YAML. Exiting.")
    except ValidationError as e:
        logging.error(f"Config file {config_path} is not valid. Exiting.\n{e}")
    else:
        pl.seed_everything(config.seed)
        logger = TensorBoardLogger("./tb_logs", name="my_model")
        model = DVAE(
            encoder=Conv2DEncoder(
                input_channels=config.model.channels,
                output_channels=config.model.codebook_size,
                num_layers=config.model.encoder.num_layers,
                num_resnet_blocks=config.model.encoder.num_resnet_blocks,
                hidden_size=config.model.codebook_vector_dim,
            ),
            decoder=Conv2DDecoder(
                input_channels=config.model.codebook_vector_dim,
                output_channels=config.model.channels,
                num_layers=config.model.decoder.num_layers,
                num_resnet_blocks=config.model.decoder.num_resnet_blocks,
                hidden_size=config.model.codebook_size,
            ),
            codebook_size=config.model.codebook_size,
            codebook_vector_dim=config.model.codebook_vector_dim,
            temperature=config.training.temperature_scheduler.end,
        )
        model_pl = DVAETrainModule(dvae=model, config=config.training)

        to_tensor = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )

        train_ds, val_ds = torch.utils.data.random_split(
            ClassificationDatasetWrapper(
                CIFAR10("./", train=True, download=True, transform=to_tensor)
            ),
            [0.8, 0.2],
        )
        test_ds = ClassificationDatasetWrapper(
            CIFAR10("./", train=False, download=True, transform=to_tensor)
        )
        train_loader = DataLoader(
            train_ds, batch_size=config.training.batch_size, num_workers=12, shuffle=True
        )
        test_loader = DataLoader(test_ds, batch_size=config.training.batch_size, num_workers=12)
        val_loader = DataLoader(val_ds, batch_size=config.training.batch_size, num_workers=12)
        trainer = pl.Trainer(
            accelerator="auto",
            max_epochs=config.training.max_epochs,
            logger=logger,
            gradient_clip_val=config.training.gradient_clip_val,
            callbacks=[
                LearningRateMonitor(logging_interval="step"),
                SaveVisualizationCallback(
                    n_images=config.training.num_vis,
                    log_every_n_epochs=config.training.save_vis_every_n_epochs,
                    dataset=train_ds,
                ),
            ],
        )

        trainer.fit(model_pl, train_dataloaders=train_loader, val_dataloaders=val_loader)
        trainer.test(model_pl, dataloaders=test_loader)


if __name__ == "__main__":
    _main()
