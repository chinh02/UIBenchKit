"""
LayoutCoder UIED Preprocessing (Subprocess Runner)
===================================================

This script is designed to be called as a SUBPROCESS from the DCGen pipeline.
It runs LayoutCoder's stages 1-4 (UIED, line detection, layout analysis, division)
which have heavy dependencies (PaddleOCR, custom CV modules) that might conflict
with DCGen's own imports.

Usage (called by pipeline.py):
    python preprocess.py <layoutcoder_project_path> <input_image_path> <output_root>

Stages executed:
    1. UIED: text detection (OCR) + component detection (IP) + merge
    2. Line detection: separator line detection using LSD
    3. Layout analysis: spatial relationship modeling + layout search
    4. Layout division: projection-based gap splitting -> mask image
"""

import sys
import os
import time


def run_preprocessing(layoutcoder_path, input_path_img, output_root):
    """Run LayoutCoder stages 1-4 on a single image."""

    # Add LayoutCoder project and UIED to Python path
    sys.path.insert(0, layoutcoder_path)
    sys.path.insert(0, os.path.join(layoutcoder_path, "UIED"))

    # Import LayoutCoder modules
    from run_single import uied
    from utils import layout, detect_lines, page_layout_divider

    start = time.process_time()

    # Stage 1: UIED - detect text and non-text components
    print("[LayoutCoder Preprocess] Stage 1: UIED detection...")
    uied(input_path_img, output_root)

    # Stage 2: Detect separator lines using LSD
    print("[LayoutCoder Preprocess] Stage 2: Line detection...")
    detect_lines.detect_sep_lines_with_lsd(input_path_img, output_root)

    # Stage 3: Layout analysis (depends on lines to filter bad layouts)
    print("[LayoutCoder Preprocess] Stage 3: Layout analysis...")
    layout.process_layout(
        input_path_img, output_root,
        use_uied_img=True, is_detail_print=False, use_sep_line=True
    )

    # Draw lines on layout image (needed by division stage)
    detect_lines.draw_lines(input_path_img, output_root)

    # Stage 4: Layout division -> produces mask image
    print("[LayoutCoder Preprocess] Stage 4: Layout division...")
    page_layout_divider.divide_layout(input_path_img, output_root)

    elapsed = time.process_time() - start
    print(f"[LayoutCoder Preprocess] Completed in {elapsed:.3f}s")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python preprocess.py <layoutcoder_project_path> <input_image_path> <output_root>")
        sys.exit(1)

    layoutcoder_path = sys.argv[1]
    input_path_img = sys.argv[2]
    output_root = sys.argv[3]

    # Set ablation config defaults (all features enabled)
    sys.path.insert(0, layoutcoder_path)
    import ablation_config
    ablation_config.is_ui_group = True
    ablation_config.is_gap_sort = True
    ablation_config.is_custom_prompt = True

    run_preprocessing(layoutcoder_path, input_path_img, output_root)
