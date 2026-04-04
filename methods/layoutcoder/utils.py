"""
LayoutCoder Utilities
=====================

Common utility functions used across the LayoutCoder pipeline.
Includes nested structure operations, image encoding, and white-page detection.
"""

import base64
import json
import re
import os
import math
from collections import defaultdict
from functools import reduce
from typing import Iterable


# ============================================================
# Nested Structure Operations (from LayoutCoder common.py)
# ============================================================

def nested_dict():
    """Create an infinitely nestable defaultdict."""
    return defaultdict(nested_dict)


def get_value(nested_structure, path):
    """Navigate a nested dict/list by a path of keys/indices."""
    if not path:
        return None

    key = path[0]
    if len(path) == 1:
        return nested_structure[key]

    if isinstance(nested_structure, dict):
        if key not in nested_structure:
            return None
        return get_value(nested_structure[key], path[1:])
    elif isinstance(nested_structure, list):
        if not isinstance(key, int) or key >= len(nested_structure):
            return None
        return get_value(nested_structure[key], path[1:])
    else:
        raise TypeError("Unsupported nested structure type.")


def set_value(nested_structure, path, value):
    """Set a value deep inside a nested dict/list, creating containers as needed."""
    if not path:
        return value

    key = path[0]

    if len(path) == 1:
        if isinstance(nested_structure, dict):
            nested_structure[key] = value
        elif isinstance(nested_structure, list):
            while len(nested_structure) <= key:
                nested_structure.append(None)
            nested_structure[key] = value
        return nested_structure

    if isinstance(nested_structure, dict):
        if key not in nested_structure:
            nested_structure[key] = {} if isinstance(path[1], str) else []
        nested_structure[key] = set_value(nested_structure[key], path[1:], value)
    elif isinstance(nested_structure, list):
        if not isinstance(key, int) or key >= len(nested_structure):
            while len(nested_structure) <= key:
                nested_structure.append({} if isinstance(path[1], str) else [])
        nested_structure[key] = set_value(nested_structure[key], path[1:], value)
    else:
        raise TypeError("Unsupported nested structure type.")

    return nested_structure


# ============================================================
# Portion Normalization
# ============================================================

def numbers_to_portions(numbers: Iterable) -> list:
    """Normalize portion values using GCD so they become small integers."""
    def gcd_multiple(nums):
        return reduce(math.gcd, nums)

    portions = list(map(lambda d: d["portion"], numbers))
    result = gcd_multiple(portions)
    return list(map(lambda d: {**d, "portion": d["portion"] // result}, numbers))


# ============================================================
# Image Encoding
# ============================================================

def encode_image_file(image_path: str) -> str:
    """Encode an image file to base64 string."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def pil_to_base64(image) -> str:
    """Convert a PIL Image to base64 string."""
    import io
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


# ============================================================
# White Page Detection
# ============================================================

def is_white_page(image_path: str, threshold: float = 0.08) -> bool:
    """
    Detect if an image is mostly a blank white page using histogram comparison.
    Based on vivo Blog's white-screen detection approach.
    """
    import cv2
    import numpy as np

    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        return True

    white_img = np.ones_like(img, dtype=img.dtype) * 255
    dst = cv2.cvtColor(white_img, cv2.COLOR_BGR2GRAY)
    dst1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(dst1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    hist_base = cv2.calcHist([dst], [0], None, [256], (0, 256), accumulate=False)
    hist_test1 = cv2.calcHist([th], [0], None, [256], (0, 256), accumulate=False)

    cv2.normalize(hist_base, hist_base, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    cv2.normalize(hist_test1, hist_test1, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

    base_test1 = cv2.compareHist(hist_base, hist_test1, 3)
    print(f"White page score: {base_test1}")
    return base_test1 <= threshold


# ============================================================
# HTML Extraction
# ============================================================

def extract_div_from_response(response: str) -> str:
    """Extract <div>...</div> content from LLM response."""
    html_pattern = re.compile(r'<div[^>]*>.*</div>', re.DOTALL)
    matched = html_pattern.findall(response)
    if matched:
        return matched[0]
    return response


def extract_html_from_response(response: str) -> str:
    """Extract <html>...</html> content from LLM response."""
    response = response.strip()
    if response.startswith("```html"):
        response = response[len("```html"):].strip()
    if response.startswith("```"):
        response = response[3:].strip()
    if response.endswith("```"):
        response = response[:-3].strip()

    html_pattern = re.compile(r'<html[^>]*>.*</html>', re.DOTALL)
    matched = html_pattern.findall(response)
    if matched:
        return matched[0]
    return response


# ============================================================
# JSON I/O
# ============================================================

def read_json_file(path: str) -> dict:
    """Read a JSON file and return the parsed data."""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def write_json_file(path: str, data, is_np: bool = False, is_box: bool = False):
    """Write data to a JSON file with optional numpy/Box support."""
    import numpy as np

    class NpEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)

    class ReprJSONEncoder(json.JSONEncoder):
        def default(self, obj):
            if obj.__class__.__name__ == 'Box':
                return repr(obj)
            return super().default(obj)

    kwargs = {"obj": data, "indent": 4, "ensure_ascii": False}

    if is_np:
        kwargs["cls"] = NpEncoder
    if is_box:
        kwargs["cls"] = ReprJSONEncoder

    with open(path, 'w', encoding='utf-8') as f:
        kwargs["fp"] = f
        json.dump(**kwargs)
