from pathlib import Path


def get_path(path: str) -> Path:
    """Get Path of a resource given by relative path"""
    return Path(__file__).parent / path
