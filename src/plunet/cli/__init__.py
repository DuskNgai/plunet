"""CLI init function."""
# These imports are necessary to register CLI commands. Do not remove!
from .cli import cli  # noqa: F401
from .test_cli import test  # noqa: F401
from .train_cli import data_dir_help, train  # noqa: F401
