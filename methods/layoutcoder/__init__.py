"""
LayoutCoder Method Integration
==============================

A layout-guided HTML generation approach that:
1. Detects UI elements using UIED (OCR + component detection) [optional]
2. Analyzes spatial layout relationships and detects separator lines
3. Divides the page into a hierarchical structure via recursive cutting
4. Generates HTML code for each atomic block using an LLM
5. Assembles blocks using flex-based CSS layout

Based on the LayoutCoder paper methodology.
When the full LayoutCoder project is available, uses UIED preprocessing for
accurate layout masks. Otherwise falls back to direct screenshot analysis.
"""

from .pipeline import (
    generate_layoutcoder,
    pipeline,
    run_uied_preprocessing,
    extract_structure,
    generate_partial_codes,
    assemble_html,
    LAYOUTCODER_PROJECT_PATH,
)
from .structure import (
    recursive_cut_draw,
    mask2json,
    json_to_html_css,
    add_html_template,
    prettify_html,
    tag_and_get_atomic_components,
    soft_separation_lines,
)
from .prompts import PROMPT_LOCAL, PROMPT_LOCAL_SIMPLE, PROMPT_GLOBAL
from .utils import (
    is_white_page,
    extract_div_from_response,
    extract_html_from_response,
    encode_image_file,
    pil_to_base64,
    nested_dict,
    get_value,
    set_value,
    numbers_to_portions,
    read_json_file,
    write_json_file,
)

__all__ = [
    # Main API function (matches generate_dcgen / generate_latcoder signature)
    'generate_layoutcoder',
    # Pipeline functions
    'pipeline',
    'run_uied_preprocessing',
    'extract_structure',
    'generate_partial_codes',
    'assemble_html',
    # Structure analysis
    'recursive_cut_draw',
    'mask2json',
    'tag_and_get_atomic_components',
    'soft_separation_lines',
    # HTML generation
    'json_to_html_css',
    'add_html_template',
    'prettify_html',
    # Utilities
    'is_white_page',
    'extract_div_from_response',
    'extract_html_from_response',
    'encode_image_file',
    'pil_to_base64',
    'nested_dict',
    'get_value',
    'set_value',
    'numbers_to_portions',
    'read_json_file',
    'write_json_file',
    # Prompts
    'PROMPT_LOCAL',
    'PROMPT_LOCAL_SIMPLE',
    'PROMPT_GLOBAL',
    # Configuration
    'LAYOUTCODER_PROJECT_PATH',
]
