"""
DCGen Methods Package
=====================

This package contains different HTML generation methods:
- dcgen: Grid-based segmentation with hierarchical generation
- direct: Direct prompting without segmentation
- latcoder: Block-based generation with module assembly
- uicopilot: Pix2Struct bbox prediction + LLM leaf generation + optimization
- layoutcoder: Layout-guided hierarchical division + per-block LLM generation + flex assembly
"""

# Methods are imported on demand to avoid loading unused dependencies
