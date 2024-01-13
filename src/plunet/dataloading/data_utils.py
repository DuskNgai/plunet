import csv
import os
from typing import Callable

import mrcfile
import numpy as np
from PIL import Image
from torch import Tensor, device


def make_directory_if_not_exists(path: str):
    """
    Create a directory if it does not exist.

    Parameters
    ----------
    path : str
        Path to the directory to be created.

    """
    if not os.path.exists(path):
        os.makedirs(path)


def get_csv_data(csv_path, delimiter=",", with_header=False, return_header=False):
    """
    Load data from a CSV file.

    Parameters
    ----------
    csv_path : str
        Path to the CSV file.
    delimiter : str, optional
        Character used to separate fields. Default is ','.
    with_header : bool, optional
        If True, the function expects the CSV file to contain a header and will
        exclude it from the output data.
        Default is False.
    return_header : bool, optional
        If True, the function returns the header along with the data.
        Default is False.

    Returns
    -------
    out_array : numpy.ndarray
        Numpy array of data from the CSV file. If with_header or return_header is True,
        the first row (header) will be excluded from the array.
        If the CSV file is empty, a numpy array of shape (0, 13) will be returned.
    header : numpy.ndarray, optional
        Only returned if return_header is True. Numpy array containing the CSV
        file's header.

    Raises
    ------
    Exception
        If the data can't be converted to float numpy array, a numpy array with
        original type will be returned.

    """
    rows = []
    with open(csv_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=delimiter)
        for row in csv_reader:
            rows.append(row)
    assert len(rows) != 0
    out_array = np.stack(rows)
    if return_header:
        try:
            out_array = np.array(out_array[1:, :], dtype=np.float), out_array[0, :]
        finally:
            out_array = np.array(out_array[1:, :]), out_array[0, :]
        return out_array
    if with_header:
        try:
            out_array = np.array(out_array[1:, :], dtype=np.float)
        finally:
            out_array = np.array(out_array[1:, :])
        return out_array
    try:
        out_array = np.array(out_array, dtype=np.float)
    except Exception:
        out_array = np.array(out_array)
    return out_array


def load_data_for_inference(data_path: str, transforms: Callable, device: device):
    """
    Load tomogram for inference.

    This function loads the tomogram, normalizes it, and performs defined
    transforms on it (most likely just conversion to Torch.Tensor).
    Additionally moves tomogram to GPU if available.

    Parameters
    ----------
    data_path : str
        Path to the tomogram to be loaded.
    transforms : callable
        A function or transform that takes in an ndarray and returns a transformed
        version.
    device : torch.device
        The device to which the data should be transferred.

    Returns
    -------
    new_data : torch.Tensor
        The transformed data, ready for inference. It has an extra batch
        dimension added, and is moved to the appropriate device.

    """
    tomogram = read_image(data_path)
    new_data = np.expand_dims(tomogram.data, 0)

    new_data = transforms(new_data)
    new_data = new_data.unsqueeze(0)  # Add batch dimension
    new_data = new_data.to(device)
    return new_data


def store_corrected_image(
    network_output: Tensor,
    out_folder: str,
    orig_data_path: str,
    ckpt_token: str,
    store_probabilities: bool = True,
    mask_threshold: float = 0.0,
) -> None:
    """
    Helper function for storing output mask.

    Stores mask into
    os.path.join(out_folder, os.path.basename(orig_data_path))
    If specified, also logits are stored before thresholding.

    Parameters
    ----------
    network_output : torch.Tensor
        The output from the network.
    out_folder : str
        Directory path to store the output mask.
    orig_data_path : str
        Original data path.
    ckpt_token : str
        Checkpoint token.
    store_probabilities : bool, optional
        If True, probabilities are stored before thresholding.
    mask_threshold : float, optional
        Threshold for the mask. Default is 0.0.
    """
    # Create out directory if it doesn't exist yet
    make_directory_if_not_exists(out_folder)

    predictions = network_output[0]
    predictions_np = predictions.squeeze(0).cpu().numpy()

    if store_probabilities:
        out_file = os.path.join(
            out_folder, os.path.basename(orig_data_path)[:-4] + "_scores.npy"
        )
        write_npy(out_file, predictions_np)

    predictions_np_thres = (predictions_np > mask_threshold).astype(np.uint8) * 255
    out_file_thres = os.path.join(
        out_folder,
        os.path.basename(orig_data_path)[:-4] + "_" + ckpt_token + "_corrected.png",
    )

    write_image(out_file_thres, predictions_np_thres)
    print("Finished.")
    return out_file_thres


def read_image(image_file: str | os.PathLike) -> np.ndarray:
    """
    Read image file.

    Parameters
    ----------
    image_file : str | os.PathLike
        Path to the image file.

    Returns
    -------
    image : np.ndarray
        Numpy array representation of the image file.

    """
    with Image.open(image_file) as f:
        image = np.array(f)
    return image


def write_image(out_file: str | os.PathLike, image: np.ndarray) -> None:
    """
    Write image file.

    Parameters
    ----------
    out_file : str | os.PathLike
        Path to the image file. (Where should it be stored?)
    image: np.ndarray
        2D or 3D image that should be stored in the given file.

    Returns
    -------
    None

    """
    image = Image.fromarray(image)
    image.save(out_file)


def read_npy(npy_file: str | os.PathLike) -> np.ndarray:
    """
    Read npy file.

    Parameters
    ----------
    npy_file : str | os.PathLike
        Path to the npy file.

    Returns
    -------
    npy : np.ndarray
        Numpy array representation of the npy file.

    """
    npy = np.load(npy_file)
    return npy


def write_npy(out_file: str | os.PathLike, npy: np.ndarray) -> None:
    """
    Write npy file.

    Parameters
    ----------
    out_file : str | os.PathLike
        Path to the npy file. (Where should it be stored?)
    npy: np.ndarray
        2D or 3D npy that should be stored in the given file.

    Returns
    -------
    None

    """
    np.save(out_file, npy)


def read_mrc(mrc_file: str | os.PathLike) -> np.ndarray:
    """
    Read mrc file.

    Parameters
    ----------
    mrc_file : str | os.PathLike
        Path to the mrc file.

    Returns
    -------
    mrc : np.ndarray
        Numpy array representation of the mrc file.

    """
    with mrcfile.open(mrc_file, permissive=True) as file:
        volume = file.data
    return volume


def write_mrc(out_file: str | os.PathLike, mrc: np.ndarray) -> None:
    """
    Write mrc file.

    Parameters
    ----------
    out_file : str | os.PathLike
        Path to the mrc file. (Where should it be stored?)
    mrc: np.ndarray
        3D mrc that should be stored in the given file.

    Returns
    -------
    None

    """
    with mrcfile.new(out_file, overwrite=True) as file:
        file.set_data(mrc)
        file.update_header_from_data()
