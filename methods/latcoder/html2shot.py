"""
LatCoder HTML2Shot
==================

HTML rendering to screenshot using Playwright.
"""

import io
import time
import logging
from PIL import Image

logger = logging.getLogger(__name__)


def html2shot(html_content: str, output_file: str = None) -> Image.Image:
    """Render HTML to screenshot using Playwright."""
    try:
        from playwright.sync_api import sync_playwright
        
        with sync_playwright() as p:
            browser = p.chromium.launch()
            context = browser.new_context()
            page = context.new_page()
            
            page.set_content(html_content, timeout=50000, wait_until='networkidle')
            
            # Wait for images to load
            images = page.query_selector_all('img')
            t_end = time.time() + 3
            for img in images:
                while time.time() < t_end:
                    try:
                        loaded = page.evaluate('(img) => img.complete', img)
                        if loaded:
                            break
                    except:
                        break
                    time.sleep(0.05)
            
            screenshot_bytes = page.screenshot(full_page=True, animations="disabled", timeout=50000)
            image = Image.open(io.BytesIO(screenshot_bytes))
            
            if output_file:
                image.save(output_file)
            
            context.close()
            browser.close()
            
        return image
        
    except Exception as e:
        logger.error(f"Failed to render HTML: {e}, returning blank image")
        image = Image.new('RGB', (1280, 960), color='white')
        if output_file:
            image.save(output_file)
        return image
