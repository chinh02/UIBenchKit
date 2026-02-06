"""
LatCoder Utilities
==================

Helper functions for code extraction and image processing.
"""

import re
import io
import base64
from typing import Optional, List
from PIL import Image


def remove_code_markers(code: str) -> str:
    """Remove markdown code markers from HTML code."""
    cleaned = re.sub(r'^```html\s*', '', code, flags=re.MULTILINE)
    cleaned = re.sub(r'\s*```$', '', cleaned, flags=re.MULTILINE)
    return cleaned.strip()


def extract_html_from_response(text: str) -> Optional[str]:
    """Extract HTML code from LLM response."""
    pattern = r'```html(.*?)```'
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        return matches[0].strip()
    if '<html' in text.lower() or '<!doctype' in text.lower():
        return text.strip()
    return None


def crop_image(image: Image.Image, bbox: List[float]) -> Image.Image:
    """Crop image using proportional bbox coordinates [x1, y1, x2, y2]."""
    width, height = image.size
    left = bbox[0] * width
    top = bbox[1] * height
    right = bbox[2] * width
    bottom = bbox[3] * height
    return image.crop((left, top, right, bottom))


def pil_to_base64(image: Image.Image) -> str:
    """Convert PIL Image to base64 string."""
    buffered = io.BytesIO()
    if image.mode in ('RGBA', 'LA', 'P'):
        image = image.convert('RGB')
    image.save(buffered, format='PNG')
    return base64.b64encode(buffered.getvalue()).decode('utf-8')
