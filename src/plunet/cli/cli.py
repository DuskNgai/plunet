import typer
from click import Context
from typer.core import TyperGroup


class OrderCommands(TyperGroup):
    """Return list of commands in the order appear."""

    def list_commands(self, ctx: Context):
        """Return list of commands in the order appear."""
        return list(self.commands)  # get commands using self.commands


cli = typer.Typer(
    cls=OrderCommands,
    add_completion=False,
    no_args_is_help=True,
    rich_markup_mode="rich",
)
OPTION_PROMPT_KWARGS = {"prompt": True, "prompt_required": True}
PKWARGS = OPTION_PROMPT_KWARGS


@cli.callback()
def callback():
    """
    Unet's training / prediction module.

    You can choose between the different options listed below.
    To see the help for a specific command, run:

    unet <command> --help

    -------

    Example:
    -------
    unet predict --tomogram-path <path-to-your-tomo>
        --ckpt-path <path-to-model-checkpoint>
        --out-folder ./segmentations

    -------
    """
