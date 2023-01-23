"""
This file holds KI specific utilities and constants
"""
from re import Pattern as RegexPattern, compile as compile_regex
from typing import Dict

LABELS: Dict[str, int] = {'HC': 0, 'PDOFF': 1, 'PDON': 2}
AXIS: Dict[str, int] = {'horiz': 0, 'vert': 1}
SACCADE: Dict[str, int] = {'pro': 0, 'anti': 1}
FILENAME_REGEX: RegexPattern = compile_regex(r'(\d+?)_(\w+?)_\w+?_(\w+?)_\d+?_(\w+)?[.csv]?')
SAMPLE_RATE = 300
