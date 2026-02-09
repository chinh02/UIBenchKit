"""
LatCoder Method Integration
===========================

A block-based HTML generation approach that:
1. Divides the design image into multiple blocks using smart splitting
2. Generates HTML code for each block using an LLM
3. Assembles blocks using either absolute positioning or agent-based assembly
4. Selects the best result using visual similarity scoring

Based on the LatCoder paper methodology.
"""

from .pipeline import (
    generate_latcoder,
    pipeline,
    generate_module_code,
    agent_assemble,
    absolute_assemble,
    refine,
    MAX_BLOCKS_LIMIT,
)
from .blocker import blocker
from .html2shot import html2shot
from .scoring import get_best, mae_score, clip_similarity, verify_score
from .prompts import PROMPT_GENERATE, PROMPT_GENERATE_ELF, PROMPT_ASSEMBLE, PROMPT_REFINE, PROMPT_GET_TEXT
from .utils import remove_code_markers, extract_html_from_response, crop_image, pil_to_base64

__all__ = [
    # Main API function (matches generate_dcgen signature)
    'generate_latcoder',
    # Pipeline functions
    'pipeline',
    'blocker',
    'generate_module_code',
    'agent_assemble',
    'absolute_assemble',
    'refine',
    # Utilities
    'html2shot',
    'get_best',
    'mae_score',
    'clip_similarity',
    'verify_score',
    'remove_code_markers',
    'extract_html_from_response',
    'crop_image',
    'pil_to_base64',
    # Constants
    'MAX_BLOCKS_LIMIT',
    # Prompts
    'PROMPT_GENERATE',
    'PROMPT_GENERATE_ELF',
    'PROMPT_ASSEMBLE',
    'PROMPT_REFINE',
    'PROMPT_GET_TEXT',
]
