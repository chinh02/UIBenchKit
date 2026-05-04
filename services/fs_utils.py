#!/usr/bin/env python3
"""
Filesystem and path utilities.
"""

import os


def get_image_files(directory: str, exclude=None):
    """Get sorted PNG files in directory, excluding known helper files."""
    exclude = exclude or ["placeholder", "bbox"]
    files = []
    for name in os.listdir(directory):
        if name.endswith(".png") and not any(marker in name for marker in exclude):
            files.append(os.path.join(directory, name))
    return sorted(files)


def ensure_dir(path: str):
    """Create directory if it does not exist."""
    os.makedirs(path, exist_ok=True)


def sanitize_for_filename(name: str) -> str:
    """Sanitize text for safe filesystem naming."""
    invalid_chars = [":", "*", "?", '"', "<", ">", "|", "\\", "/"]
    result = name
    for char in invalid_chars:
        result = result.replace(char, "-")
    result = result.replace(".", "-")
    while "--" in result:
        result = result.replace("--", "-")
    return result.strip("-")


def resolve_path(path: str, data_dir: str) -> str:
    """Resolve relative path against data directory."""
    if os.path.isabs(path):
        return path
    return os.path.join(data_dir, path)


def create_path_resolver(data_dir: str):
    """Create a single-arg resolver(path) bound to a data directory."""
    def _resolve_path(path: str) -> str:
        return resolve_path(path, data_dir)

    return _resolve_path
