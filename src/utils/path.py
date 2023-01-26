"""
This filename defines the paths to various resources in the project, as well as some related tools.
"""
import os
from pathlib import Path
from typing import Callable, Any

root_path = Path(__file__).parent.parent.parent
data_path = root_path.joinpath('data')
ki_data_path = data_path.joinpath('ki')
config_path = root_path.joinpath('config')
src_path = root_path.joinpath('src')
checkpoints_path = root_path.joinpath('checkpoints')
ki_data_tmp_path = ki_data_path.joinpath('tmp')


def walk_of_life(root: str, apply: Callable, file_extension: str, **kwargs) -> Any:
    """
    Recursively applies a function to all files with a certain filename extension inside a path.

    :param root: The path to the root directory
    :param apply: The function to be applied to every filename
    :param file_extension: All files with this extension are passed to apply
    :param kwargs: Optional keyword arguments passed to apply
    :return: A generator with the returned values of apply
    """
    for root_, dirs, files in os.walk(root):
        for name in files:
            if name.endswith(f".{file_extension}"):
                yield apply(root_, name.replace('.csv', ''), **kwargs)
