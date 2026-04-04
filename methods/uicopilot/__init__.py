"""
UICopilot Method Integration
=============================

A multi-stage HTML generation approach that:
1. Uses a fine-tuned Pix2Struct model to predict DOM structure with bounding boxes
2. Generates HTML for each leaf node using an LLM agent
3. Assembles the tree into a full HTML document
4. Optimizes the result using an LLM agent

Based on the UICopilot/UICoder methodology.
"""

from .pipeline import generate_uicopilot

__all__ = [
    'generate_uicopilot',
]
