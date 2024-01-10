import os

import torch

from plunet.networks.unet import PLUnet

from plunet.dataloading.data_utils import (
    load_data_for_inference,
    store_corrected_image,
)
from plunet.dataloading.pl_augmentation import get_mirrored_data, get_prediction_transforms


def test(
    data_folder,
    ckpt_path,
    out_folder,
    store_probabilities=False,
    test_time_augmentation=True,
    mask_threshold=0.0,
):
    """
    Optical proximity correction using a trained model.

    This function takes a path to a tomogram file, a path to a trained
    model checkpoint file, and a path to an output folder. It loads the
    trained model, and performs sliding window inference with 4-fold test-time
    augmentation on the new data, and then stores the resulting mask
    in the output folder.

    Parameters
    ----------
    data_folder : str
        Path to the folder of tested data.
    ckpt_path : str
        Path to the trained model checkpoint file.
    out_folder : str
        Path to the folder where the output mask should be stored.
    store_probabilities : bool, optional
        If True, store the predicted probabilities along with the mask
        (default is False).
    test_time_augmentation: bool, optional
        If True, test-time augmentation is performed, i.e. data is rotated
        into eight different orientations and predictions are averaged.
    mask_threshold: float, optional
        Threshold for the mask. Only voxels with a
        score higher than this threshold will be corrected. (default: 0.0)

    Returns
    -------
    mask_file: str
        Path to the corrected image.

    Raises
    ------
    FileNotFoundError
        If `image_path` or `ckpt_path` does not point to a file.
    """
    # Load the trained PyTorch Lightning model
    model_checkpoint = ckpt_path
    # TODO: Probably better to not keep this with custom checkpoint names
    ckpt_token = os.path.basename(model_checkpoint).split("-val_loss")[0]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize the model and load trained weights from checkpoint
    pl_model = PLUnet.load_from_checkpoint(
        model_checkpoint, map_location=device, strict=False
    )
    pl_model.to(device)

    # Put the model into evaluation mode
    pl_model.eval()

    # Preprocess the new data
    transforms = get_prediction_transforms()

    data_paths = [os.path.join(data_folder, f) for f in os.listdir(data_folder)]

    for data_path in data_paths:
        new_data = load_data_for_inference(
            data_path, transforms, device=torch.device("cpu")
        )
        new_data = new_data.to(torch.float32)

        # Perform test time augmentation (4-fold mirroring)
        predictions = torch.zeros_like(new_data)
        if test_time_augmentation:
            print("Performing 4-fold test-time augmentation.")
        for m in range(4 if test_time_augmentation else 1):
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    predictions += get_mirrored_data(
                        pl_model(
                            get_mirrored_data(new_data.clone(), m).to(device)
                        )[0], m
                    ).detach().cpu()

        if test_time_augmentation:
            predictions /= 4.0

        # Extract mask and store them in an output file.
        store_corrected_image(
            predictions,
            out_folder=out_folder,
            orig_data_path=data_path,
            ckpt_token=ckpt_token,
            store_probabilities=store_probabilities,
            mask_threshold=mask_threshold,
        )
