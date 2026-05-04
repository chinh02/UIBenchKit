#!/usr/bin/env python3
"""
Generation and screenshot helpers.
"""

import os
import re
import time
import base64
import io

from PIL import Image


def _extract_error_details(error: Exception) -> str:
    status_code = getattr(error, "status_code", None)
    body = getattr(error, "body", None)
    details = str(error)
    if status_code:
        details = f"HTTP {status_code} - {details}"
    if body:
        details = f"{details} | body={body}"
    return details


def _is_non_retryable_client_error(error: Exception) -> bool:
    status_code = getattr(error, "status_code", None)
    return isinstance(status_code, int) and 400 <= status_code < 500 and status_code != 429


def _encode_image_with_size_guard(img_path: str, max_raw_bytes: int = 3 * 1024 * 1024, max_side: int = 2048) -> str:
    """
    Encode image as base64. If input is large, downscale/re-encode to reduce 400 risk
    from upstream providers that enforce payload limits on vision requests.
    """
    file_size = os.path.getsize(img_path)
    if file_size <= max_raw_bytes:
        from utils import encode_image
        return encode_image(img_path)

    with Image.open(img_path) as img:
        img = img.convert("RGB")
        width, height = img.size
        scale = min(1.0, max_side / max(width, height))
        if scale < 1.0:
            new_size = (int(width * scale), int(height * scale))
            img = img.resize(new_size, Image.LANCZOS)

        buffer = io.BytesIO()
        # JPEG significantly reduces request payload while preserving enough visual detail.
        img.save(buffer, format="JPEG", quality=88, optimize=True)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")


def generate_single(prompt: str, bot, img_path: str, save_path: str = None, max_retries: int = 5):
    """Generate HTML from one image using direct prompting."""
    last_error = None
    image_b64 = _encode_image_with_size_guard(img_path)
    for index in range(max_retries):
        try:
            html = bot.ask(prompt, image_b64)
            code = re.findall(r"```html([^`]+)```", html)
            if code:
                html = code[0]
            if len(html) < 10:
                preview = html[:200] if html else "empty"
                raise Exception(f"No HTML code found in response. Response was: {preview}")
            if save_path:
                with open(save_path, "w", encoding="utf-8") as file:
                    file.write(html)
            return html
        except Exception as error:
            last_error = error
            error_details = _extract_error_details(error)
            # 4xx (except rate limit) are usually deterministic and should fail fast.
            if _is_non_retryable_client_error(error):
                model_name = getattr(bot, "model", getattr(bot, "model_version", "unknown"))
                raise Exception(
                    "Vision request rejected by upstream provider (non-retryable). "
                    f"model={model_name}, image={os.path.basename(img_path)}, error={error_details}. "
                    "Check model vision support and base URL compatibility."
                ) from error
            wait_time = 2 ** (index + 1)
            file_name = os.path.basename(img_path)
            print(
                f"Retry {index + 1}/{max_retries} for {file_name}: "
                f"{error_details[:200]}... waiting {wait_time}s"
            )
            time.sleep(wait_time)

    raise Exception(
        f"Failed to generate HTML for {img_path} after {max_retries} retries. "
        f"Last error: {_extract_error_details(last_error)}"
    )


def create_dcgen_generator(*, prompt_dcgen: dict, seg_params_default: dict):
    """Create a generate_dcgen(bot, img_path, save_path=None, seg_params=None) function."""
    def generate_dcgen(bot, img_path: str, save_path: str = None, seg_params: dict = None):
        from utils import ImgSegmentation, DCGenGrid

        params = seg_params or seg_params_default
        img_seg = ImgSegmentation(img_path, **params)
        dcgen_grid = DCGenGrid(
            img_seg,
            prompt_seg=prompt_dcgen["prompt_leaf"],
            prompt_refine=prompt_dcgen["prompt_root"],
        )
        dcgen_grid.generate_code(bot, multi_thread=True)

        if save_path:
            with open(save_path, "w", encoding="utf-8", errors="ignore") as file:
                file.write(dcgen_grid.code)

        return dcgen_grid.code

    return generate_dcgen


def take_screenshots_task(directory: str, replace: bool = False):
    """Generate screenshots for HTML files using Playwright full-page capture."""
    from playwright.sync_api import sync_playwright

    html_files = [name for name in os.listdir(directory) if name.endswith(".html")]
    if not html_files:
        return

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page()
        page.set_viewport_size({"width": 1920, "height": 1080})

        for filename in html_files:
            html_path = os.path.join(directory, filename)
            png_path = html_path.replace(".html", ".png")
            if os.path.exists(png_path) and not replace:
                continue

            try:
                page.goto("file://" + os.path.abspath(html_path))
                page.wait_for_load_state("networkidle")
                page.screenshot(path=png_path, full_page=True)
            except Exception:
                pass

        browser.close()
