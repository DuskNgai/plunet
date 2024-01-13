from pathlib import Path
from typing import Optional

import pytorch_lightning as pl
from monai.data import DataLoader

from plunet.dataloading.pl_dataset import PLDataset


class PLDataModule(pl.LightningDataModule):
    """
    Pytorch Lightning datamodule.

    Parameters
    ----------
    data_dir : str
        The directory where the data resides. The directory should have a
        specific structure.
    batch_size : int
        The batch size for the data loaders.
    num_workers : int
        The number of workers to use in the data loaders.
    aug_prob_to_one : bool, default False
        Whether to apply data augmentation.

    Attributes
    ----------
    train_img_dir : Path
        The path to the directory containing training images.
    train_lab_dir : Path
        The path to the directory containing training labels.
    val_img_dir : Path
        The path to the directory containing validation images.
    val_lab_dir : Path
        The path to the directory containing validation labels.
    train_dataset : PLDataset
        The training dataset.
    val_dataset : PLDataset
        The validation dataset.
    test_dataset : PLDataset
        The test dataset.
    """

    def __init__(self, data_dir, batch_size, num_workers, aug_prob_to_one=False):
        """Initialization of data paths and data loaders.

        The data_dir should have the following structure:
        data_dir/
        ├── imagesTr/       # Directory containing training images
        │   ├── img1.glp.png    # Image file (currently requires .png format)
        │   ├── img2.glp.png    # Image file
        │   └── ...
        ├── imagesVal/      # Directory containing validation images
        │   ├── img1.glp.png    # Image file
        │   ├── img2.glp.png    # Image file
        │   └── ...
        ├── labelsTr/       # Directory containing training labels
        │   ├── label1.glpOPC.png  # Label file (currently requires .png format)
        │   ├── label2.glpOPC.png  # Label file
        │   └── ...
        └── labelsVal/      # Directory containing validation labels
            ├── label1.glpOPC.png  # Label file
            ├── label2.glpOPC.png  # Label file
            └── ...
        """
        super().__init__()
        self.data_dir = Path(data_dir)
        self.train_img_dir = self.data_dir.joinpath("imagesTr")
        self.train_lab_dir = self.data_dir.joinpath("labelsTr")
        self.val_img_dir = self.data_dir.joinpath("imagesVal")
        self.val_lab_dir = self.data_dir.joinpath("labelsVal")
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.aug_prob_to_one = aug_prob_to_one

    def setup(self, stage: Optional[str] = None):
        """
        Setups the datasets for different stages of the training process.

        If stage is None, the datasets for both the fit and test stages are setup.

        Parameters
        ----------
        stage : str, optional
            The stage of the training process.
            One of None, "fit" or "test".
        """
        if stage in (None, "fit"):
            self.train_dataset = PLDataset(
                img_folder=self.train_img_dir,
                label_folder=self.train_lab_dir,
                train=True,
                aug_prob_to_one=self.aug_prob_to_one,
            )
            self.val_dataset = PLDataset(
                img_folder=self.val_img_dir, label_folder=self.val_lab_dir, train=False
            )

        if stage in (None, "test"):
            self.test_dataset = PLDataset(
                self.data_dir, test=True, transform=self.transform
            )  # TODO: How to do prediction?

    def train_dataloader(self) -> DataLoader:
        """
        Returns the training dataloader.

        Returns
        -------
        DataLoader
            The training dataloader.
        """
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self) -> DataLoader:
        """
        Returns the validation dataloader.

        Returns
        -------
        DataLoader
            The validation dataloader.
        """
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self) -> DataLoader:
        """
        Returns the test dataloader.

        Returns
        -------
        DataLoader
            The test dataloader.
        """
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
