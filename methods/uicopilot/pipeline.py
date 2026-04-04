"""
UICopilot Pipeline
==================

Main orchestration for the UICopilot method.
Combines a fine-tuned Pix2Struct bbox model with LLM agents
to convert webpage screenshots into HTML code.

Pipeline:
1. Pix2Struct predicts DOM structure with bounding boxes
2. Leaf nodes are cropped and HTML is generated per-leaf via LLM
3. Full page is assembled from the bbox tree
4. LLM agent refines/optimizes the assembled HTML
"""

import logging
import re
import time
from pathlib import Path
from PIL import Image

from .prompts import PROMPT_I2C, PROMPT_OPTIMIZE
from .utils import (
    Html2BboxTree, BboxTree2Html, BboxTree2StyleList,
    move_to_device, add_special_tokens, pil_to_base64,
)

logger = logging.getLogger(__name__)

# ============================================================
# Lazy-loaded Pix2Struct bbox model (module-level cache)
# ============================================================
_bbox_model = None
_processor = None
_device = None


def _load_bbox_model(device='cuda'):
    """
    Lazy-load the Pix2Struct bbox model and processor.
    Downloads ~3.5GB from HuggingFace on first call.
    """
    global _bbox_model, _processor, _device
    if _bbox_model is not None:
        return _bbox_model, _processor, _device

    import torch
    from transformers import (
        AutoTokenizer, Pix2StructProcessor,
        Pix2StructForConditionalGeneration,
    )

    logger.info("Loading Pix2Struct bbox model (xcodemind/uicopilot_structure)...")
    _device = device

    # The fine-tuned model repo lacks preprocessor_config.json, so we load
    # the image processor from the base Pix2Struct model and the tokenizer
    # from the fine-tuned checkpoint, then compose them.
    base_proc = Pix2StructProcessor.from_pretrained("google/pix2struct-base")
    tokenizer = AutoTokenizer.from_pretrained("xcodemind/uicopilot_structure")
    _processor = Pix2StructProcessor(
        image_processor=base_proc.image_processor, tokenizer=tokenizer,
    )

    _bbox_model = Pix2StructForConditionalGeneration.from_pretrained(
        "xcodemind/uicopilot_structure",
        is_encoder_decoder=True,
        device_map=device,
        torch_dtype=torch.float16,
    )
    add_special_tokens(_bbox_model, _processor.tokenizer)
    logger.info("Pix2Struct bbox model loaded successfully.")
    return _bbox_model, _processor, _device


# ============================================================
# Core Functions
# ============================================================
def infer_bbox(image, model, processor, device):
    """
    Run the Pix2Struct model to predict bbox-annotated HTML.

    Args:
        image: PIL Image of the webpage
        model: Loaded Pix2Struct model
        processor: Pix2Struct processor
        device: torch device string

    Returns:
        Bbox-annotated HTML string (e.g. '<body bbox=[0,0,1,1]>...')
    """
    import torch

    model.eval()
    with torch.no_grad():
        input_text = '<body bbox=['
        decoder_input_ids = processor.tokenizer.encode(
            input_text, return_tensors='pt', add_special_tokens=True
        )[..., :-1]
        encoding = processor(
            images=[image], text=[""], max_patches=1024, return_tensors='pt'
        )
        item = {
            'decoder_input_ids': decoder_input_ids,
            'flattened_patches': encoding['flattened_patches'].half(),
            'attention_mask': encoding['attention_mask'],
        }
        item = move_to_device(item, device)

        outputs = model.generate(
            **item,
            max_new_tokens=2560,
            eos_token_id=processor.tokenizer.eos_token_id,
            do_sample=True,
        )
        prediction_html = processor.tokenizer.batch_decode(
            outputs, skip_special_tokens=True
        )[0]

    return prediction_html


def pruning(node, now_depth, max_depth, min_area):
    """Prune the bbox tree by depth and minimum area."""
    bbox = node['bbox']
    area = abs((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
    if area < min_area:
        return None
    if now_depth >= max_depth:
        node['children'] = []
    else:
        new_children = []
        for cnode in node['children']:
            pruned = pruning(cnode, now_depth + 1, max_depth, min_area)
            if pruned is not None:
                new_children.append(pruned)
        node['children'] = new_children
    return node


def extract_html(html):
    """Strip markdown code block markers from LLM response."""
    if '```' in html:
        html = html.split('```')[1]
    if html[:4] == 'html':
        html = html[4:]
    html = html.strip()
    return html


def locate_by_index(bbox_tree, index):
    """Navigate a bbox tree to a node specified by dash-separated index."""
    target = bbox_tree
    for i in filter(lambda x: x, index.split('-')):
        target = target['children'][int(i)]
    return target


def gen(bot, image, model, processor, device,
        max_depth=100, min_area=100, retries=3, retry_delay=2):
    """
    Core UICopilot generation logic.

    Args:
        bot: DCGen Bot instance (GPT4, Gemini, Claude, etc.)
        image: PIL Image of the webpage
        model: Pix2Struct model
        processor: Pix2Struct processor
        device: torch device string
        max_depth: Max tree depth for pruning
        min_area: Min node area for pruning
        retries: Number of retry attempts
        retry_delay: Delay between retries in seconds

    Returns:
        Tuple of (html_before_optimize, html_after_optimize, extracted_images)
    """
    for attempt in range(retries):
        try:
            imgs = []

            # Stage 1: Predict bbox tree
            logger.info("UICopilot Stage 1: Predicting bbox tree...")
            prediction_html = infer_bbox(image, model, processor, device)
            bbox_tree = Html2BboxTree(prediction_html, size=image.size)

            if bbox_tree is None:
                raise ValueError("Failed to parse bbox tree from model output")

            # Prune
            pruning(bbox_tree, 1, max_depth, min_area)

            # Re-parse for generation (fresh copy)
            bbox_tree = Html2BboxTree(prediction_html, size=image.size)
            index_list = BboxTree2StyleList(bbox_tree, skip_leaf=False)
            # Keep only leaf nodes
            index_list = [item for item in index_list if not len(item['children'])]

            # Stage 2: Generate HTML for each leaf node
            logger.info(f"UICopilot Stage 2: Generating code for {len(index_list)} leaf nodes...")
            img_count = 0
            for item in index_list:
                bbox = item['bbox']
                index = item['index']

                image_crop = image.crop((
                    bbox[0], bbox[1],
                    bbox[0] + bbox[2], bbox[1] + bbox[3]
                ))

                if item['type'] == 'img':
                    imgs.append(image_crop)
                    part_html = f'{img_count}.png'
                    img_count += 1
                else:
                    # Use DCGen bot with system prompt (matching original uicopilot message structure)
                    crop_b64 = pil_to_base64(image_crop)
                    response = bot.ask("", crop_b64, verbose=False, system_prompt=PROMPT_I2C)
                    part_html = extract_html(response)

                target = locate_by_index(bbox_tree, index)
                target['children'] = [part_html]

            # Stage 3: Assemble
            html = BboxTree2Html(bbox_tree, style=True)

            # Stage 4: Optimize via LLM
            logger.info("UICopilot Stage 4: Optimizing assembled HTML...")
            full_b64 = pil_to_base64(image)
            response = bot.ask(html, full_b64, verbose=False, system_prompt=PROMPT_OPTIMIZE)
            html2 = extract_html(response)

            return html, html2, imgs

        except Exception as e:
            logger.warning(f"UICopilot attempt {attempt + 1} failed: {e}")
            if attempt < retries - 1:
                time.sleep(retry_delay)
            else:
                logger.error(f"UICopilot failed after {retries} attempts")
                raise


# ============================================================
# DCGen API Integration
# ============================================================
def generate_uicopilot(bot, img_path: str, save_path: str = None,
                       device: str = 'cuda') -> str:
    """
    Generate HTML from image using the UICopilot method.

    This function matches the signature of generate_dcgen() and generate_latcoder()
    for easy integration with the DCGen benchmark.

    Args:
        bot: DCGen Bot instance
        img_path: Path to input design image
        save_path: Optional path to save output HTML
        device: CUDA device for Pix2Struct model (default: 'cuda')

    Returns:
        Generated HTML code
    """
    # Load bbox model lazily
    model, processor, dev = _load_bbox_model(device)

    # Load image
    image = Image.open(img_path).convert('RGB')

    # Run pipeline
    html_before, html_after, imgs = gen(
        bot=bot,
        image=image,
        model=model,
        processor=processor,
        device=dev,
    )

    # Use the optimized version as the main output
    html_code = html_after

    # Save output
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # Save main output (after optimize) as {id}.html
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(html_code)

        # Save artifacts in a subfolder: {id}_uicopilot/
        sample_id = save_path.stem  # e.g. "0" from "0.html"
        artifacts_dir = save_path.parent / f'{sample_id}_uicopilot'
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        # Save before/after optimization HTML
        with open(artifacts_dir / 'before_optimize.html', 'w', encoding='utf-8') as f:
            f.write(html_before)
        with open(artifacts_dir / 'after_optimize.html', 'w', encoding='utf-8') as f:
            f.write(html_after)

        # Save input image
        image.save(str(artifacts_dir / 'input.png'))

        # Save extracted images (cropped img nodes)
        for idx, img in enumerate(imgs):
            img.save(str(artifacts_dir / f'{idx}.png'))

    return html_code
