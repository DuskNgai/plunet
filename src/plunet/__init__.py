from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("unet")
except PackageNotFoundError:
    __version__ = "uninstalled"

__author__ = "Haizhao Dai"
__email__ = "daihzhz2023@shanghaitech.edu.cn"
