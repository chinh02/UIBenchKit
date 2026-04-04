"""
LayoutCoder Pipeline
====================

Main orchestration for the LayoutCoder method within the DCGen framework.

LayoutCoder's approach:
  1. Preprocess: UIED (OCR + component detection) + line detection + layout analysis + division
     -> produces a mask image with separation lines
  2. Structure extraction: recursive cutting on the mask -> hierarchical layout structure
  3. Code generation: for each atomic block, crop the ORIGINAL image and call the LLM
  4. Assembly: insert generated code into the flex-based layout structure -> final HTML

Key design decisions:
  - Stages 1 (UIED preprocessing) runs via subprocess to isolate LayoutCoder's heavy
    dependencies (PaddleOCR, custom CV modules) from DCGen's module namespace.
  - If UIED preprocessing fails or the LayoutCoder project is not available,
    the pipeline falls back to running recursive_cut_draw() directly on the screenshot
    with tuned parameters. This produces a simpler layout structure but avoids
    external dependencies.
  - Stage 3 replaces LayoutCoder's hardcoded OpenAI client with DCGen's bot interface,
    allowing any model (GPT-4, Claude, Gemini, etc.) to be used.
"""

import json
import logging
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Tuple

from PIL import Image

from .prompts import PROMPT_LOCAL
from .structure import (
    mask2json,
    json_to_html_css,
    add_html_template,
    prettify_html,
    recursive_cut_draw,
    tag_and_get_atomic_components,
)
from .utils import (
    is_white_page,
    extract_div_from_response,
    pil_to_base64,
    read_json_file,
    write_json_file,
)

logger = logging.getLogger(__name__)

# Path to the LayoutCoder project (parent repo)
# This is auto-detected relative to the DCGen project, or can be set via env var
LAYOUTCODER_PROJECT_PATH = os.environ.get(
    "LAYOUTCODER_PROJECT_PATH",
    os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "papers", "LayoutCoder"))
)


# ============================================================
# Stage 1: UIED Preprocessing (via subprocess)
# ============================================================

def run_uied_preprocessing(image_path: str, output_root: str,
                           layoutcoder_path: str = None) -> bool:
    """
    Run LayoutCoder stages 1-4 (UIED + lines + layout + division) as a subprocess.

    Uses the LayoutCoder project's own Python venv if available (it has
    PaddleOCR and other heavy dependencies installed), otherwise falls back
    to the current interpreter.

    Returns True if preprocessing succeeded, False otherwise.
    """
    lc_path = layoutcoder_path or LAYOUTCODER_PROJECT_PATH

    if not os.path.isdir(lc_path):
        logger.warning(f"LayoutCoder project not found at {lc_path}, skipping UIED preprocessing")
        return False

    preprocess_script = os.path.join(os.path.dirname(__file__), "preprocess.py")

    # Prefer LayoutCoder's own venv Python (has PaddleOCR etc.)
    lc_python = os.path.join(lc_path, ".venv", "Scripts", "python.exe")
    if not os.path.isfile(lc_python):
        lc_python = os.path.join(lc_path, ".venv", "bin", "python")
    if not os.path.isfile(lc_python):
        lc_python = sys.executable  # fallback to current interpreter

    try:
        logger.info(f"Running UIED preprocessing via subprocess (python={lc_python})...")
        result = subprocess.run(
            [lc_python, preprocess_script, lc_path, image_path, output_root],
            capture_output=True, text=True, timeout=300,  # 5 min timeout
            cwd=lc_path,  # Run from LayoutCoder project directory
        )

        if result.returncode != 0:
            logger.warning(f"UIED preprocessing failed:\nstdout: {result.stdout[-500:]}\nstderr: {result.stderr[-500:]}")
            return False

        logger.info("UIED preprocessing completed successfully")
        return True

    except subprocess.TimeoutExpired:
        logger.warning("UIED preprocessing timed out (>300s)")
        return False
    except Exception as e:
        logger.warning(f"UIED preprocessing error: {e}")
        return False


# ============================================================
# Stage 2: Structure Extraction
# ============================================================

def extract_structure(image_path: str, output_root: str,
                      use_uied: bool = True) -> dict:
    """
    Extract hierarchical layout structure from the image.

    If UIED preprocessing was successful, reads the mask image and runs
    recursive_cut_draw() on it (high accuracy).
    Otherwise, runs recursive_cut_draw() directly on the screenshot (fallback).

    Returns:
        {"structure": dict, "page_size": (w, h)}
    """
    name = os.path.splitext(os.path.basename(image_path))[0]
    # Windows path compatibility: handle both / and \ in path
    if '/' in name:
        name = name.split('/')[-1]
    if '\\' in name:
        name = name.split('\\')[-1]

    sep_root = os.path.join(output_root, "sep")
    struct_root = os.path.join(output_root, "struct")
    os.makedirs(struct_root, exist_ok=True)

    mask_path = os.path.join(sep_root, f"{name}_mask.png")

    if use_uied and os.path.exists(mask_path):
        # Best path: use UIED-generated mask
        logger.info(f"Using UIED mask for structure extraction: {mask_path}")
        data = mask2json(sep_root, struct_root, name)
    else:
        # Fallback: run recursive cutting directly on the screenshot
        logger.info("Falling back to direct screenshot analysis (no UIED mask)")
        result_img, data = recursive_cut_draw(image_path, depth=3)
        # Save visualization
        os.makedirs(struct_root, exist_ok=True)
        result_img.save(os.path.join(struct_root, f"{name}_sep.png"))

    return data


# ============================================================
# Stage 3: Code Generation for Atomic Components
# ============================================================

def generate_partial_codes(bot, image_path: str, structure_data: dict,
                           output_root: str) -> dict:
    """
    Generate HTML code for each atomic component in the layout structure.
    Uses the DCGen bot to replace LayoutCoder's hardcoded OpenAI client.

    This traverses the structure tree, crops the original image for each
    atomic block, sends it to the LLM, and inserts the generated code
    back into the structure.

    Args:
        bot: DCGen bot instance (GPT4, Claude, Gemini, etc.)
        image_path: Path to the original full-page screenshot
        structure_data: Layout structure dict from stage 2
        output_root: Directory to save cropped images

    Returns:
        Updated structure with 'code' fields populated
    """
    full_img = Image.open(image_path)
    partial_img_root = os.path.join(output_root, "struct", "partial")
    os.makedirs(partial_img_root, exist_ok=True)

    name = os.path.splitext(os.path.basename(image_path))[0]
    if '/' in name:
        name = name.split('/')[-1]
    if '\\' in name:
        name = name.split('\\')[-1]

    def add_ids_and_codes(structure, current_id=1, depth=0):
        """Recursively traverse, crop image regions, and generate code via LLM."""
        if structure['type'] == 'atomic':
            position = structure["position"]
            bbox = (
                max(0, position["column_min"]),
                max(0, position["row_min"]),
                min(full_img.width, position["column_max"]),
                min(full_img.height, position["row_max"]),
            )

            # Crop the region from the original image
            cropped_path = os.path.join(partial_img_root, f'{name}_part_{current_id}.png')
            cropped_img = full_img.crop(bbox)
            cropped_img.save(cropped_path)

            structure['id'] = current_id

            # Skip blank regions to save tokens
            if is_white_page(cropped_path):
                structure['code'] = "    "
                logger.info(f"  Block {current_id}: blank (skipped)")
            else:
                # Call LLM via DCGen bot
                img_b64 = pil_to_base64(cropped_img)
                try:
                    response = bot.ask(PROMPT_LOCAL, img_b64, verbose=False)
                    code = extract_div_from_response(response)
                    structure['code'] = code
                    logger.info(f"  Block {current_id}: generated ({len(code)} chars)")
                except Exception as e:
                    logger.error(f"  Block {current_id}: LLM call failed: {e}")
                    structure['code'] = "    "

            structure["size"] = (bbox[2] - bbox[0], bbox[3] - bbox[1])
            return current_id + 1

        # Recurse into children
        if 'value' in structure:
            for item in structure['value']:
                current_id = add_ids_and_codes(item, current_id, depth + 1)
        return current_id

    add_ids_and_codes(structure_data)
    return structure_data


# ============================================================
# Stage 4: Assembly -> Final HTML
# ============================================================

def assemble_html(structure_data: dict, page_size: tuple) -> str:
    """
    Convert the code-populated structure into final HTML.

    Args:
        structure_data: Structure dict with 'code' fields
        page_size: (width, height) of the original image

    Returns:
        Complete HTML string
    """
    page_width, page_height = page_size
    html_output = json_to_html_css(structure_data)
    html_template = add_html_template(
        html_output,
        ratio=page_width / page_height,
        border=True,
        margin="2px",
        bg_color="white",
    )
    return prettify_html(html_template)


# ============================================================
# Full Pipeline
# ============================================================

def pipeline(bot, image_path: str, output_root: str,
             use_uied: bool = True,
             layoutcoder_path: str = None) -> str:
    """
    Run the complete LayoutCoder pipeline on a single image.

    Args:
        bot: DCGen bot instance
        image_path: Path to input design screenshot
        output_root: Temp directory for intermediate artifacts
        use_uied: Whether to attempt UIED preprocessing
        layoutcoder_path: Override path to LayoutCoder project

    Returns:
        Final HTML code string
    """
    os.makedirs(output_root, exist_ok=True)
    name = os.path.splitext(os.path.basename(image_path))[0]

    # Stage 1: UIED Preprocessing (optional, via subprocess)
    uied_success = False
    if use_uied:
        uied_success = run_uied_preprocessing(image_path, output_root, layoutcoder_path)

    # Stage 2: Extract hierarchical layout structure
    logger.info("Stage 2: Extracting layout structure...")
    layout_data = extract_structure(image_path, output_root, use_uied=uied_success)

    # Check if structure extraction found anything useful
    structure = layout_data.get("structure", {})
    if not structure or (isinstance(structure, dict) and structure.get("type") is None):
        logger.warning("Structure extraction produced empty result, falling back to single-block")
        # Create a single atomic block for the entire page
        img = Image.open(image_path)
        structure = {
            "type": "atomic",
            "portion": 1,
            "value": "     ",
            "position": {
                "column_min": 0,
                "row_min": 0,
                "column_max": img.width,
                "row_max": img.height,
            }
        }
        layout_data = {"structure": structure, "page_size": img.size}

    # Stage 3: Generate code for each atomic block
    logger.info("Stage 3: Generating code for atomic blocks...")
    atomic_count = len(tag_and_get_atomic_components(structure))
    logger.info(f"  Found {atomic_count} atomic blocks")
    coded_structure = generate_partial_codes(bot, image_path, structure, output_root)

    # Save the structure with code (for debugging/caching)
    struct_root = os.path.join(output_root, "struct")
    os.makedirs(struct_root, exist_ok=True)
    struct_json_path = os.path.join(struct_root, f"{name}_sep.json")
    try:
        write_json_file(struct_json_path, {
            **layout_data,
            "structure": coded_structure,
        })
    except Exception as e:
        logger.warning(f"Failed to save structure JSON: {e}")

    # Stage 4: Assemble into final HTML
    logger.info("Stage 4: Assembling final HTML...")
    page_size = layout_data.get("page_size", Image.open(image_path).size)
    final_html = assemble_html(coded_structure, page_size)

    # Save the assembled HTML
    html_path = os.path.join(struct_root, f"{name}_sep.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(final_html)

    return final_html


# ============================================================
# DCGen API Integration
# ============================================================

def generate_layoutcoder(bot, img_path: str, save_path: str = None,
                         use_uied: bool = True) -> str:
    """
    Generate HTML from image using the LayoutCoder method.

    This function matches the signature of generate_dcgen(), generate_latcoder(), etc.
    for easy integration with the DCGen API.

    Args:
        bot: DCGen bot instance (any model: GPT4, Claude, Gemini, etc.)
        img_path: Path to input design image
        save_path: Optional path to save output HTML (e.g., results/run_id/0.html)
        use_uied: Whether to run UIED preprocessing (requires LayoutCoder project).
                  If False or if LayoutCoder is not available, falls back to direct
                  screenshot analysis.

    Returns:
        Generated HTML code string
    """
    # Create artifacts directory
    # e.g., results/run_id/0.html -> results/run_id/0_layoutcoder/
    if save_path:
        save_dir = Path(save_path)
        sample_id = save_dir.stem  # e.g., "0" from "0.html"
        artifacts_dir = str(save_dir.parent / f'{sample_id}_layoutcoder')
    else:
        artifacts_dir = tempfile.mkdtemp(prefix="layoutcoder_")

    # Run the full pipeline
    html_code = pipeline(
        bot=bot,
        image_path=img_path,
        output_root=artifacts_dir,
        use_uied=use_uied,
    )

    # Save final output
    if save_path:
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(html_code)

        # Also take a screenshot if possible
        try:
            from playwright.sync_api import sync_playwright
            png_path = save_path.replace('.html', '.png')
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                page = browser.new_page()
                page.set_viewport_size({"width": 1920, "height": 1080})
                page.goto("file://" + os.path.abspath(save_path))
                page.wait_for_load_state("networkidle")
                page.screenshot(path=png_path, full_page=True)
                browser.close()
        except Exception:
            pass  # Screenshot is optional; pipeline handles it separately

    return html_code
