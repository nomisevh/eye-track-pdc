"""
This file holds KI specific utilities and constants
"""
import os
import re
from typing import Callable, Dict, List, Pattern, Tuple

from pandas import DataFrame

from data.KI.gothrough_clean import extract
from preprocess.base import Trial

KI_LABELS: Dict[str, int] = {'HC': 0, 'PDOFF': 1, 'PDON': 2}
KI_AXIS: Dict[str, int] = {'horiz': 0, 'vert': 1}
KI_SACCADE: Dict[str, int] = {'pro': 0, 'anti': 1}
KI_FILENAME_REGEX: Pattern[str] = re.compile(r'(\d+?)_(\w+?)_\w+?_(\w+?)_\d+?_(\w+)?[.csv]?')
KI_SAMPLE_RATE = 300


def load_file(root: str,
              basename: str,
              read_meta: bool = False,
              rename_fields: bool = False) -> Tuple[DataFrame, str, int] or DataFrame:
    """
    Loads a CSV file as pandas DataFrame instance.

    Parameters
    ----------
    root          : str
                    Absolute path to the directory containing the file.
    basename      : str
                    File basename.
    read_meta     : bool
                    Set to True to read and return file metadata information.
    rename_fields : bool
                    Set to True to change field names from "{x/y}_pos" to "position" and "drift" based on the type of
                    the experiment.

    Returns
    -------
    tuple
        a tuple containing the DataFrame instance, the metadata and the sample rate
    """
    basename = basename.strip('.csv')
    df = extract(root=root, file=basename, plot=False)
    if rename_fields:
        if basename.find('vert') != -1:
            # vertical saccades experiment
            df.rename(
                columns={'xpos': 'drift', 'xpos_diff': 'drift_diff', 'ypos': 'position', 'ypos_diff': 'position_diff'},
                inplace=True
            )
        else:
            # horizontal saccades experiment
            df.rename(
                columns={'ypos': 'drift', 'ypos_diff': 'drift_diff', 'xpos': 'position', 'xpos_diff': 'position_diff'},
                inplace=True
            )
        df[['position', 'drift', 'position_diff', 'drift_diff']] = \
            df[['position', 'drift', 'position_diff', 'drift_diff']].fillna(0)
    if read_meta:
        def read_metadata() -> str:
            """
            Reads the first 22 lines of a file. These lines contain text describing the experiment.
            :return: the metadata as string
            """
            with open(os.path.join(root, basename + '.csv')) as reader:
                lines = reader.readlines()
                return "\n".join(lines[:23])

        meta = read_metadata()
        # Sample rate is always 300 for the data we have
        sample_rate_regex = re.compile(r'SampleRate\(Hz\): (\d*)')
        sample_rate = int(sample_rate_regex.findall(meta)[0])
        return df, meta, sample_rate
    return df


def load_ki_trials(trials_dir: str) -> List[Trial]:
    print(f'\t[KIDataset][__init__] Loading trials from: {trials_dir}')
    trials = []
    for df, filename in walk_of_life(trials_dir, load_file, rename_fields=True):
        individual, group, axis, saccade = KI_FILENAME_REGEX.findall(filename)[0]
        trials.append(
            Trial(mts=df, label=KI_LABELS[group], person_id=individual, **dict(
                group_str=group, axis_str=axis, saccade_str=saccade,
                group=KI_LABELS[group], axis=KI_AXIS[axis], saccade=KI_SACCADE[saccade],
            ))
        )
    return trials


def walk_of_life(root: str,
                 parse: Callable,
                 file_extension: str = 'csv',
                 **parser_kwargs) -> Tuple[List[DataFrame], List[int], List[str]]:
    """
    Iterates over all CSV files inside path. For KI: expects root to end at /HC, /PD_OFF or /PD_ON.
    Courtesy of Gonzalo.

    Parameters
    ----------
    root          : str
                    Absolute path to the data root.
    file_extension: str
                    The format of the files to be read.
    parse         : Callable
                    Function to parse the data, with the signature (root, filename) -> (Dataframe, sample rate)
    parser_kwargs :
                    Parse keyword arguments.

    Returns
    -------
    Generator
        Generator with return signature (Dataframe, filename)
    """
    for root_, dirs, files in os.walk(root):
        for name in files:
            if name.endswith(f".{file_extension}"):
                yield parse(root_, name.replace('.csv', ''), **parser_kwargs), name
