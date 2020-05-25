from pathlib import Path
from typing import Union


def get_path(path: str) -> Path:
    """Get Path of a resource given by relative path"""
    return Path(__file__).parent / path


def read_data(path: str, mode='r') -> Union[str, bytes]:
    """Read a test data file"""
    with get_path(path).open(mode=mode) as f:
        return f.read()
