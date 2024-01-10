from typing import Callable, List, Union

import numpy as np
import torch
from monai.transforms import (
    Compose,
    RandAxisFlipd,
    ThresholdIntensity,
    ThresholdIntensityd,
    RandRotate90d,
    ToTensor,
    ToTensord,
)

from plunet.dataloading.transforms import (
    DownsampleSegForDeepSupervisionTransform,
)

### Hard-coded area

# Hard-coding kernel sizes for pooling operations (work also with smaller network sizes)
pool_op_kernel_sizes = [
    [2, 2, 2],
    [2, 2, 2],
    [2, 2, 2],
    [2, 2, 2],
    [2, 2, 2],
]  # hard-coded
net_num_pool_op_kernel_sizes = pool_op_kernel_sizes
deep_supervision_scales = [[1, 1, 1]] + [
    list(i) for i in 1 / np.cumprod(np.vstack(net_num_pool_op_kernel_sizes), axis=0)
][:-1]

# Which axes should be used for mirroring?
mirror_axes = (0, 1, 2)


def get_mirrored_data(img: torch.Tensor, mirror_idx: int) -> torch.Tensor:
    """
    Get mirrored data for test time augmentation.

    There are 8 possible cases for 3d data, enumerated from 0 to 7.
    The function supports mirroring across two axes,
    and combinations thereof.

    Parameters
    ----------
    img : torch.Tensor
        Input tensor to be mirrored.
    mirror_idx : int
        Integer index to select mirror case.
        Should be within the range [0, 7] inclusive.

    Returns
    -------
    torch.Tensor
        The mirrored image tensor.

    Raises
    ------
    AssertionError
        If the mirror index is not in the range [0, 7].

    """
    assert 0 <= mirror_idx < 8
    if mirror_idx == 0:
        return img

    if mirror_idx == 1 and (2 in mirror_axes):
        return torch.flip(img, (4,))

    if mirror_idx == 2 and (1 in mirror_axes):
        return torch.flip(img, (3,))

    if mirror_idx == 3 and (2 in mirror_axes) and (1 in mirror_axes):
        return torch.flip(img, (4, 3))

    if mirror_idx == 4 and (0 in mirror_axes):
        return torch.flip(img, (2,))

    if mirror_idx == 5 and (0 in mirror_axes) and (2 in mirror_axes):
        return torch.flip(img, (4, 2))

    if mirror_idx == 6 and (0 in mirror_axes) and (1 in mirror_axes):
        return torch.flip(img, (3, 2))

    if (
        mirror_idx == 7
        and (0 in mirror_axes)
        and (1 in mirror_axes)
        and (2 in mirror_axes)
    ):
        return torch.flip(img, (4, 3, 2))


def get_training_transforms(
    prob_to_one: bool = False, return_as_list: bool = False
) -> Union[List[Callable], Compose]:
    """
    Returns the data augmentation transforms for training phase.

    The function sets up an augmentation sequence containing a variety of
    transformations such as rotations, zooms, 90 degree rotations,
    Gaussian noise, brightness adjustments, and more.
    If desired, the sequence can be returned as a list.

    Parameters
    ----------
    prob_to_one : bool, optional
        If True, the probability of applying the transformation is set to 1.0 for
            all transformations in the sequence.
        If False, the probability is lower (specified within each transformation).
        Default is False.
    return_as_list : bool, optional
        If True, the sequence of transformations is returned as a list.
        If False, the sequence is returned as a Compose object. Default is False.

    Returns
    -------
    List[Callable] or Compose
        If return_as_list is True, the function returns a list of
            transformation functions.
        If return_as_list is False, the function returns a Compose object
            containing the sequence of transformations.

    """
    aug_sequence = [
        ThresholdIntensityd(
            keys=("image", "label"),
            threshold=128,
            above=False,
            cval=1
        ),
        RandRotate90d(
            keys=("image", "label"),
            prob=(1.0 if prob_to_one else 0.5),
            max_k=3,
            spatial_axes=(0, 1),
        ),
        RandAxisFlipd(keys=("image", "label"), prob=(0.5)),
        DownsampleSegForDeepSupervisionTransform(
            keys=["label"], ds_scales=deep_supervision_scales, order="nearest"
        ),
        ToTensord(keys=["image"], dtype=torch.float),
    ]
    if return_as_list:
        return aug_sequence
    return Compose(aug_sequence)


def get_validation_transforms(
    return_as_list: bool = False,
) -> Union[List[Callable], Compose]:
    """
    Returns the data augmentation transforms for the validation phase.

    The function sets up a sequence of transformations including downsampling
    and tensor conversion. If desired, the sequence can be returned as a list.

    Parameters
    ----------
    return_as_list : bool, optional
        If True, the sequence of transformations is returned as a list.
        If False, the sequence is returned as a Compose object. Default is False.

    Returns
    -------
    List[Callable] or Compose
        If return_as_list is True, the function returns a list of
            transformation functions.
        If return_as_list is False, the function returns a Compose object
            containing the sequence of transformations.

    """
    aug_sequence = [
        ThresholdIntensityd(
            keys=("image", "label"),
            threshold=128,
            above=False,
            cval=1
        ),
        DownsampleSegForDeepSupervisionTransform(
            keys=["label"], ds_scales=deep_supervision_scales, order="nearest"
        ),
        ToTensord(keys=["image"], dtype=torch.float),
    ]
    if return_as_list:
        return aug_sequence
    return Compose(aug_sequence)


def get_prediction_transforms() -> Compose:
    """
    Returns the data augmentation transforms for the prediction phase.

    The function sets up a Compose object containing a transformation for
    converting data to tensors.

    Returns
    -------
    Compose
        A Compose object containing the sequence of transformations for
        the prediction phase.

    """
    transforms = Compose(
        [
            ThresholdIntensity(
                threshold=128,
                above=False,
                cval=1
            ),
            ToTensor(),
        ]
    )
    return transforms
