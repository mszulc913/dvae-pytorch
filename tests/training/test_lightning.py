import pytorch_lightning as pl
import torch.utils.data
from torch.nn.functional import mse_loss
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import MNIST

from dvae_pytorch.datasets.wrappers import ClassificationDatasetWrapper
from dvae_pytorch.models.dvae import DVAE, Conv2DDecoder, Conv2DEncoder
from dvae_pytorch.training.config import DVAETrainingConfig, LinearSchedulerConfig
from dvae_pytorch.training.lightning import DVAETrainModule


def test_overfit_mnist() -> None:
    train_ds, val_ds = torch.utils.data.random_split(
        ClassificationDatasetWrapper(
            MNIST("../MNIST/", train=True, download=True, transform=transforms.ToTensor())
        ),
        [0.8, 0.2],
    )
    test_ds = ClassificationDatasetWrapper(
        MNIST("./", train=False, download=True, transform=transforms.ToTensor())
    )
    train_ds = Subset(train_ds, range(32))
    val_ds = Subset(val_ds, range(32))
    test_ds = Subset(test_ds, range(32))
    train_loader = DataLoader(train_ds, batch_size=16, num_workers=1, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=4, num_workers=1)
    val_loader = DataLoader(val_ds, batch_size=4, num_workers=1)
    model = DVAE(
        encoder=Conv2DEncoder(
            input_channels=1,
            output_channels=32,
            num_layers=1,
            num_resnet_blocks=1,
            hidden_size=16,
        ),
        decoder=Conv2DDecoder(
            input_channels=16,
            output_channels=1,
            num_layers=1,
            num_resnet_blocks=1,
            hidden_size=32,
        ),
        codebook_size=32,
        codebook_vector_dim=16,
        temperature=1 / 16,
    )
    config = DVAETrainingConfig(
        batch_size=4,
        max_epochs=30,
        learning_rate=1e-3,
        learning_rate_scheduler_min=1e-1,
        kl_div_weight_scheduler=LinearSchedulerConfig(start=0, end=1e-5, warmup=0.3, cooldown=0.2),
    )
    model_pl = DVAETrainModule(dvae=model, config=config)
    trainer = pl.Trainer(accelerator="cpu", max_epochs=30, logger=None)
    loss_start = trainer.test(model_pl, dataloaders=test_loader)[0]

    trainer.fit(model_pl, train_dataloaders=train_loader, val_dataloaders=val_loader)

    loss_end = trainer.test(model_pl, dataloaders=test_loader)[0]
    img = train_ds[0]
    img_decoded = model.decode(model.encode(img.unsqueeze(dim=0))[0])[0]
    mse_decoded = mse_loss(img_decoded, img).detach().item()
    assert mse_decoded < 0.1
    assert loss_end["test/reconstruction_loss"] < loss_start["test/reconstruction_loss"]
