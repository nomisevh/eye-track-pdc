"""
This file defines the paths to various resources in the project.
"""
from pathlib import Path

root_path = Path(__file__).parent.parent.parent
data_path = root_path.joinpath('data')
ki_data_path = data_path.joinpath('ki')
config_path = root_path.joinpath('config')
src_path = root_path.joinpath('src')
checkpoints_path = root_path.joinpath('checkpoints')
