import warnings

import pytorch_lightning as pl
from pytorch_lightning import Callback
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
import torch
import torch.multiprocessing

from plunet.dataloading.pl_datamodule import (
    PLDataModule,
)
from plunet.networks.unet import PLUnet

warnings.filterwarnings("ignore", category=UserWarning, module="torch._tensor")
warnings.filterwarnings("ignore", category=UserWarning, module="monai.data")


def train(
    data_dir: str,
    batch_size: int = 2,
    num_workers: int = 8,
    max_epochs: int = 1000,
    resume_ckpt_path: str = None,
    project_name: str = "unet_v0",
    sub_name: str = "1",
    log_dir: str = "logs/",
    aug_prob_to_one: bool = False,
    use_deep_supervision: bool = False,
):
    """
    Train the model on the specified data.

    The function sets up a data module and a model, configures logging,
    model checkpointing and learning rate monitoring,
    and starts the training process.

    Parameters
    ----------
    data_dir : str, optional
        Path to the directory containing training data.
    batch_size : int, optional
        Number of samples per batch of input data.
    num_workers : int, optional
        Number of subprocesses to use for data loading.
    max_epochs : int, optional
        Maximum number of epochs to train for.
    resume_ckpt_path : str, optional
        Path to the checkpoint file to resume training from.
    project_name : str, optional
        Name of the project for logging purposes.
    sub_name : str, optional
        Sub-name of the project for logging purposes.
    log_dir : str, optional
        Path to the directory where logs should be stored.
    aug_prob_to_one : bool, optional
        If True, all augmentation probabilities are set to 1.
    use_deep_supervision : bool, optional
        If True, enables deep supervision in the U-Net model.

    Returns
    -------
    None
    """

    # Set up the device
    torch.set_float32_matmul_precision('high')
    torch.multiprocessing.set_sharing_strategy('file_system')

    project_name = project_name
    checkpointing_name = project_name + "_" + sub_name

    # Set up logging
    csv_logger = pl_loggers.CSVLogger(log_dir, "csv_logs")
    tensorboard_logger = pl_loggers.TensorBoardLogger(log_dir, "tb_logs")

    # Set up model checkpointing
    checkpoint_callback_val_loss = ModelCheckpoint(
        dirpath="checkpoints/",
        filename=checkpointing_name + "-{epoch:02d}-{val_loss:.2f}",
        monitor="val_loss",
        mode="min",
        save_top_k=3,
    )

    checkpoint_callback_regular = ModelCheckpoint(
        save_top_k=-1,  # Save all checkpoints
        every_n_epochs=100,
        dirpath="checkpoints/",
        filename=checkpointing_name + "-{epoch}-{val_loss:.2f}",
        verbose=True,  # Print a message when a checkpoint is saved
    )

    lr_monitor = LearningRateMonitor(
        logging_interval="epoch", log_momentum=True)

    class PrintLearningRate(Callback):
        def on_epoch_start(self, trainer, pl_module):
            current_lr = trainer.optimizers[0].param_groups[0]["lr"]
            print(
                f"Epoch {trainer.current_epoch}: Learning Rate = {current_lr}")

    print_lr_cb = PrintLearningRate()
    # Set up the trainer
    trainer = pl.Trainer(
        precision="16-mixed",
        logger=[csv_logger, tensorboard_logger],
        callbacks=[
            checkpoint_callback_val_loss,
            checkpoint_callback_regular,
            lr_monitor,
            print_lr_cb,
        ],
        max_epochs=max_epochs,
    )

    # Set up the model
    if resume_ckpt_path is not None:
        model = PLUnet.load_from_checkpoint(
            resume_ckpt_path, strict=False
        )
    else:
        model = PLUnet(
            max_epochs=max_epochs, use_deep_supervision=use_deep_supervision
        )

    # Set up the data module
    data_module = PLDataModule(
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        aug_prob_to_one=aug_prob_to_one,
    )

    # Start the training process
    trainer.fit(model, data_module)
