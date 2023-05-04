"""
This filename holds KI specific utilities and constants
"""
import math
import os
from re import Pattern as RegexPattern, compile as compile_regex
from typing import Dict, Iterable

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pandas import DataFrame
from tqdm.auto import tqdm

from utils.data import interpolate_outliers
from utils.path import walk_of_life, ki_data_path

LABELS: Dict[str, int] = {'HC': 0, 'PDOFF': 1, 'PDON': 2}
AXIS: Dict[str, int] = {'horiz': 0, 'vert': 1}
SACCADE: Dict[str, int] = {'pro': 0, 'anti': 1}
FILENAME_REGEX: RegexPattern = compile_regex(r'(\d+?)_(\w+?)_\w+?_(\w+?)_\d+?_(\w+)?[.csv]?')
SAMPLE_RATE = 300
# The length of the time window where the target is invisible prior to a saccade.
INVISIBLE_TARGET_DURATION = 0.2


def load_data(train: bool, sources: Iterable[str]):
    data_path = ki_data_path.joinpath('train' if train else 'test').resolve().__str__()

    def apply(root: str, filename: str):
        # Extract the time series from the raw file
        frame = extract(filename=filename, root=root, plot=False)
        # Rename the columns to consolidate trials on different axes
        return rename_columns(frame, filename), filename

    # Accumulate returns from walk of life
    return zip(*list(tqdm(walk_of_life(data_path, apply, 'csv', sources), desc='loading files', unit='files')))


def extract(filename: str, root: str, plot: bool = True, head_norm_n: int = 300) -> DataFrame:
    """
    Extracts the data from a raw KI trial-file and centers the time series around the mean of the first datapoints.
    :param filename: The base name of the raw file (excl. file extensions)
    :param root: The path to the directory of the file
    :param plot: Whether to plot
    :param head_norm_n: The number of initial datapoints to use for centering the time series
    :return: A dataframe holding the time series.
    """
    screen_x = 478  # mm
    screen_y = 269
    res_x = 1920  # pxl
    res_y = 1080
    dist = 600
    deg_cols = ['RY (deg)', 'LY (deg)', 'LX (deg)', 'RX (deg)']
    final_cols = ['xpos', 'ypos', 'xpos_diff', 'ypos_diff']
    df = pd.read_csv(os.path.join(root, filename + ".csv"), skiprows=19, delimiter=";",
                     usecols=['Time (ms)', 'RX (deg)', 'LX (deg)', 'RY (deg)', 'LY (deg)', 'Label'])

    df.ffill(inplace=True)
    index = df[df['Label'].isnull()].index
    df.drop(index, inplace=True)
    lx = df["LX (deg)"].head(head_norm_n).mean()
    rx = df["RX (deg)"].head(head_norm_n).mean()
    ly = df["LY (deg)"].head(head_norm_n).mean()
    ry = df["RY (deg)"].head(head_norm_n).mean()

    if lx > 0:
        df["LX (deg)"] = df["LX (deg)"] - lx
    elif lx < 0:
        df["LX (deg)"] = df["LX (deg)"] + lx
    if rx > 0:
        df["RX (deg)"] = df["RX (deg)"] - rx
    elif rx < 0:
        df["RX (deg)"] = df["RX (deg)"] + rx
    if ly > 0:
        df["LY (deg)"] = df["LY (deg)"] - ly
    elif ly < 0:
        df["LY (deg)"] = df["LY (deg)"] + ly
    if ry > 0:
        df["RY (deg)"] = df["RY (deg)"] - ry
    elif ry < 0:
        df["RY (deg)"] = df["RY (deg)"] + ry

    df = interpolate_outliers(df, deg_cols)

    conditions = [
        (df['Label'] == 'cible = S:gap_Dir:horiz_A:-20_C:pro'),
        (df['Label'] == 'cible = S:gap_Dir:horiz_A:20_C:pro'),
        (df['Label'] == 'cible = S:gap_Dir:horiz_A:-20_C:anti'),
        (df['Label'] == 'cible = S:gap_Dir:horiz_A:20_C:anti'),
        (df['Label'] == 'cible = S:step_Dir:vert_A:-12_C:pro'),
        (df['Label'] == 'cible = S:step_Dir:vert_A:12_C:pro')
    ]

    values = [(res_x / screen_x) * dist * math.tan(-20 * math.pi / 180),
              (res_x / screen_x) * dist * math.tan(20 * math.pi / 180),
              (res_x / screen_x) * dist * math.tan(-20 * math.pi / 180),
              (res_x / screen_x) * dist * math.tan(20 * math.pi / 180),
              (res_y / screen_y) * dist * math.tan(-12 * math.pi / 180),
              (res_y / screen_y) * dist * math.tan(12 * math.pi / 180)]
    df['target'] = np.select(conditions, values)

    # Add explicit L/R signals
    df['xpos_L'] = (res_x / screen_x) * dist * np.tan(np.radians(df['LX (deg)']))
    df['ypos_L'] = (res_y / screen_y) * dist * np.tan(np.radians(df['LY (deg)']))
    df['xpos_R'] = (res_x / screen_x) * dist * np.tan(np.radians(df['RX (deg)']))
    df['ypos_R'] = (res_y / screen_y) * dist * np.tan(np.radians(df['RY (deg)']))

    # Compute mean signals
    df['xpos'] = (res_x / screen_x) * dist * np.tan(((df['LX (deg)'] + df['RX (deg)']) / 2) * np.pi / 180)
    df['ypos'] = (res_y / screen_y) * dist * np.tan(((df['LY (deg)'] + df['RY (deg)']) / 2) * np.pi / 180)

    df['xpos'] -= df['xpos'].head(head_norm_n).mean()
    df['ypos'] -= df['ypos'].head(head_norm_n).mean()

    # Compute diff signals
    df['xpos_diff'] = df['xpos_R'] - df['xpos_L']
    df['ypos_diff'] = df['ypos_R'] - df['ypos_L']

    df = interpolate_outliers(df, final_cols)

    if plot:
        fig = plt.figure(figsize=(6, 3), dpi=150)
        plt.plot(df["Time (ms)"], df["xpos"])
        plt.plot(df["Time (ms)"], df["ypos"])
        plt.plot(df["Time (ms)"], df["target"])
        fig.savefig(os.path.join(root, "clean_int_{}.svg".format(filename)), dpi=200)
        plt.close(fig)

    return df


def rename_columns(frame: DataFrame, filename: str):
    """
    Renames the columns of a dataframe depending on if it's a horizontal or vertical trial.
    """
    if filename.find('vert') != -1:
        # vertical saccades experiment
        frame.rename(
            columns={'xpos': 'drift', 'xpos_diff': 'drift_diff', 'ypos': 'position', 'ypos_diff': 'position_diff'},
            inplace=True
        )
    else:
        # horizontal saccades experiment
        frame.rename(
            columns={'ypos': 'drift', 'ypos_diff': 'drift_diff', 'xpos': 'position', 'xpos_diff': 'position_diff'},
            inplace=True
        )
    return frame
