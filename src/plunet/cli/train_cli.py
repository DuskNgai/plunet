from typer import Option

from ..train import train as _train
from .cli import OPTION_PROMPT_KWARGS as PKWARGS
from .cli import cli


@cli.command(name="train", no_args_is_help=True)
def train(
    data_dir: str = Option(  # noqa: B008
        ...,
        help='Data directory path, following the required structure. '
             'To learn more about the required data structure, '
             'type "plunet data_structure_help"',
        **PKWARGS,
    ),
    batch_size: int = Option(  # noqa: B008
        2,
        help="Batch size for training.",
    ),
    num_workers: int = Option(  # noqa: B008
        8,
        help="Number of worker threads for loading data",
    ),
    max_epochs: int = Option(  # noqa: B008
        1000,
        help="Maximum number of epochs for training",
    ),
    resume_ckpt_path: str = Option(  # noqa: B008
        None,
        help="Path to the checkpoint file to resume training from",
    ),
    project_name: str = Option(  # noqa: B008
        "unet_v0",
        help="Project name. This helps to find your model again.",
    ),
    sub_name: str = Option(  # noqa: B008
        "1",
        help="Subproject name. For multiple runs in the same project, "
             "please specify sub_names.",
    ),
    log_dir: str = Option(  # noqa: B008
        "logs/",
        help="Log directory path. Training logs will be stored here.",
    ),
    aug_prob_to_one: bool = Option(  # noqa: B008
        True,
        help='Whether to augment with a probability of one. This helps with the '
             'model\'s generalization, but also severely increases training time. '
             'Pass "True" or "False".',
    ),
    use_deep_supervision: bool = Option(  # noqa: B008
        True,
        help='Whether to use deep supervision. Pass "True" or "False".'
    ),
):
    """
    Initiates the Unet training routine with more advanced options.

    Parameters
    ----------
    data_dir : str
        Path to the data directory, structured as per the Unet's requirement.
        Use "plunet data_structure_help" for detailed information on the required
        data structure.
    batch_size : int
        Number of samples per batch, by default 2.
    num_workers : int
        Number of worker threads for data loading, by default 1.
    max_epochs : int
        Maximum number of training epochs, by default 1000.
    resume_ckpt_path : str
        Path to the checkpoint file to resume training from, by default None.
    project_name : str
        Name of the project for logging purposes, by default 'unet_v0'.
    sub_name : str
        Sub-name for the project, by default '1'.
    log_dir : str
        Path to the directory where logs will be stored, by default 'logs/'.
    aug_prob_to_one : bool
        Determines whether to apply very strong data augmentation, by default True.
        If set to False, data augmentation still happens, but not as frequently.
        More data augmentation can lead to a better performance, but also increases the
        training time substantially.
    use_deep_supervision : bool
        Determines whether to use deep supervision, by default True.

    Note

    ----

    The actual training logic resides in the function '_train'.
    """
    _train(
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        max_epochs=max_epochs,
        resume_ckpt_path=resume_ckpt_path,
        project_name=project_name,
        sub_name=sub_name,
        log_dir=log_dir,
        aug_prob_to_one=aug_prob_to_one,
        use_deep_supervision=use_deep_supervision,
    )


@cli.command(name="data_structure_help")
def data_dir_help():
    """
    Display information about the training data directory structure.

    Note:
    ----
        The file names of images and their corresponding labels should match.
        The segmentation algorithm uses this assumption to pair images with labels.
    """
    print(
        "The data directory structure should be as follows:\n"
        "data_dir/\n"
        "├── imagesTr/       # Directory containing training images\n"
        "│   ├── img1.glp.png    # Image file (currently requires .png format)\n"
        "│   ├── img2.glp.png    # Image file\n"
        "│   └── ...\n"
        "├── imagesVal/      # Directory containing validation images\n"
        "│   ├── img1.glp.png    # Image file\n"
        "│   ├── img2.glp.png    # Image file\n"
        "│   └── ...\n"
        "├── labelsTr/       # Directory containing training labels\n"
        "│   ├── label1.glpOPC.png  # Label file (currently requires .png format)\n"
        "│   ├── label2.glpOPC.png  # Label file\n"
        "│   └── ...\n"
        "└── labelsVal/      # Directory containing validation labels\n"
        "    ├── label1.glpOPC.png  # Label file\n"
        "    ├── label2.glpOPC.png  # Label file\n"
        "    └── ..."
    )
