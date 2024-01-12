from typer import Option

from ..test import test as _test
from .cli import OPTION_PROMPT_KWARGS as PKWARGS
from .cli import cli


@cli.command(name="test", no_args_is_help=True)
def test(
    data_folder: str = Option(  # noqa: B008
        ...,
        help="Path to the folder of tested data.",
        **PKWARGS
    ),
    ckpt_path: str = Option(  # noqa: B008
        ...,
        help="Path to the pre-trained model checkpoint that should be used.",
        **PKWARGS,
    ),
    out_folder: str = Option(  # noqa: B008
        "./predictions",
        help="Path to the folder where mask should be stored."
    ),
    store_probabilities: bool = Option(  # noqa: B008
        True,
        help="Should probability maps be output in addition to mask?"
    ),
    test_time_augmentation: bool = Option(  # noqa: B008
        True,
        help="Use 4-fold test time augmentation (TTA)? "
             "TTA improves mask quality slightly, but also increases runtime.",
    ),
    mask_threshold: float = Option(  # noqa: B008
        0.0,
        help="Threshold for the mask. Only voxels with a score higher than "
             "this threshold will be corrected. (default: 0.0)",
    ),
):
    """Correct image using a trained model.

    Example
    -------
    plunet test --image-path <path-to-your-image>
    --ckpt-path <path-to-your-model>
    """
    _test(
        data_folder=data_folder,
        ckpt_path=ckpt_path,
        out_folder=out_folder,
        store_probabilities=store_probabilities,
        test_time_augmentation=test_time_augmentation,
        mask_threshold=mask_threshold,
    )
