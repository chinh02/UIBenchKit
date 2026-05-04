"""
UICopilot Pipeline
==================

Main orchestration for the UICopilot method.
"""

import importlib.util
import logging
import time
from pathlib import Path

from PIL import Image

from .prompts import PROMPT_I2C, PROMPT_OPTIMIZE
from .utils import (
    BboxTree2Html,
    BboxTree2StyleList,
    Html2BboxTree,
    add_special_tokens,
    move_to_device,
    pil_to_base64,
)

logger = logging.getLogger(__name__)


# Lazy-loaded Pix2Struct bbox model (module-level cache)
_bbox_model = None
_processor = None
_device = None


def _load_bbox_model(device: str = "cuda"):
    """
    Lazy-load the Pix2Struct bbox model and processor.
    """
    global _bbox_model, _processor, _device
    if _bbox_model is not None:
        return _bbox_model, _processor, _device

    import torch
    from transformers import AutoTokenizer, Pix2StructForConditionalGeneration, Pix2StructProcessor

    logger.info("Loading Pix2Struct bbox model (xcodemind/uicopilot_structure)...")

    requested_device = device
    if requested_device == "cuda" and not torch.cuda.is_available():
        logger.warning("UICopilot requested CUDA but no GPU is available. Falling back to CPU.")
        _device = "cpu"
    else:
        _device = requested_device

    dtype = torch.float16 if _device == "cuda" else torch.float32

    # The fine-tuned checkpoint misses preprocessor_config.json, so we compose:
    # image processor from pix2struct-base + tokenizer from finetuned checkpoint.
    base_proc = Pix2StructProcessor.from_pretrained("google/pix2struct-base")
    tokenizer = AutoTokenizer.from_pretrained("xcodemind/uicopilot_structure")
    _processor = Pix2StructProcessor(image_processor=base_proc.image_processor, tokenizer=tokenizer)

    model_kwargs = {
        "is_encoder_decoder": True,
        "torch_dtype": dtype,
    }
    has_accelerate = importlib.util.find_spec("accelerate") is not None
    if has_accelerate:
        model_kwargs["device_map"] = _device
    else:
        logger.warning(
            "Package 'accelerate' not found. Loading UICopilot model without device_map "
            f"and moving to '{_device}' directly."
        )

    _bbox_model = Pix2StructForConditionalGeneration.from_pretrained(
        "xcodemind/uicopilot_structure",
        **model_kwargs,
    )
    if not has_accelerate:
        _bbox_model = _bbox_model.to(_device)

    add_special_tokens(_bbox_model, _processor.tokenizer)
    logger.info("Pix2Struct bbox model loaded successfully.")
    return _bbox_model, _processor, _device


def infer_bbox(image, model, processor, device):
    """
    Run the Pix2Struct model to predict bbox-annotated HTML.
    """
    import torch

    model.eval()
    with torch.no_grad():
        input_text = "<body bbox=["
        decoder_input_ids = processor.tokenizer.encode(
            input_text,
            return_tensors="pt",
            add_special_tokens=True,
        )[..., :-1]
        encoding = processor(images=[image], text=[""], max_patches=1024, return_tensors="pt")

        patch_dtype = torch.float16 if device == "cuda" else torch.float32
        item = {
            "decoder_input_ids": decoder_input_ids,
            "flattened_patches": encoding["flattened_patches"].to(dtype=patch_dtype),
            "attention_mask": encoding["attention_mask"],
        }
        item = move_to_device(item, device)

        outputs = model.generate(
            **item,
            max_new_tokens=2560,
            eos_token_id=processor.tokenizer.eos_token_id,
            do_sample=True,
        )
        prediction_html = processor.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

    return prediction_html


def pruning(node, now_depth, max_depth, min_area):
    """Prune a bbox tree by depth and minimum area."""
    bbox = node["bbox"]
    area = abs((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
    if area < min_area:
        return None
    if now_depth >= max_depth:
        node["children"] = []
    else:
        new_children = []
        for cnode in node["children"]:
            pruned = pruning(cnode, now_depth + 1, max_depth, min_area)
            if pruned is not None:
                new_children.append(pruned)
        node["children"] = new_children
    return node


def extract_html(html):
    """Strip markdown code-fence markers from an LLM response."""
    if "```" in html:
        parts = html.split("```")
        if len(parts) > 1:
            html = parts[1]
    if html[:4].lower() == "html":
        html = html[4:]
    return html.strip()


def locate_by_index(bbox_tree, index):
    """Navigate a bbox tree to a node by dash-separated index."""
    target = bbox_tree
    for position in filter(lambda x: x, index.split("-")):
        target = target["children"][int(position)]
    return target


def gen(bot, image, model, processor, device, max_depth=100, min_area=100, retries=3, retry_delay=2):
    """
    Core UICopilot generation logic.
    """
    for attempt in range(retries):
        try:
            imgs = []

            logger.info("UICopilot Stage 1: Predicting bbox tree...")
            prediction_html = infer_bbox(image, model, processor, device)
            parsed_tree = Html2BboxTree(prediction_html, size=image.size)
            if parsed_tree is None:
                raise ValueError("Failed to parse bbox tree from model output")

            pruning(parsed_tree, 1, max_depth, min_area)

            # Use a fresh parse for generation to match existing method behavior.
            bbox_tree = Html2BboxTree(prediction_html, size=image.size)
            if bbox_tree is None:
                raise ValueError("Failed to parse bbox tree for generation")

            index_list = BboxTree2StyleList(bbox_tree, skip_leaf=False)
            index_list = [item for item in index_list if not len(item["children"])]

            logger.info("UICopilot Stage 2: Generating code for %s leaf nodes...", len(index_list))
            img_count = 0
            for item in index_list:
                bbox = item["bbox"]
                index = item["index"]

                image_crop = image.crop((bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]))
                if item["type"] == "img":
                    imgs.append(image_crop)
                    part_html = f"{img_count}.png"
                    img_count += 1
                else:
                    crop_b64 = pil_to_base64(image_crop)
                    response = bot.ask("", crop_b64, verbose=False, system_prompt=PROMPT_I2C)
                    part_html = extract_html(response)

                target = locate_by_index(bbox_tree, index)
                target["children"] = [part_html]

            logger.info("UICopilot Stage 3: Assembling HTML...")
            html_before = BboxTree2Html(bbox_tree, style=True)

            logger.info("UICopilot Stage 4: Optimizing assembled HTML...")
            full_b64 = pil_to_base64(image)
            response = bot.ask(html_before, full_b64, verbose=False, system_prompt=PROMPT_OPTIMIZE)
            html_after = extract_html(response)

            return html_before, html_after, imgs
        except Exception as error:
            logger.warning("UICopilot attempt %s failed: %s", attempt + 1, error)
            if attempt < retries - 1:
                time.sleep(retry_delay)
            else:
                logger.error("UICopilot failed after %s attempts", retries)
                raise


def generate_uicopilot(bot, img_path: str, save_path: str = None, device: str = "cuda") -> str:
    """
    Generate HTML from image using the UICopilot method.

    Signature intentionally matches other methods for API/runner integration.
    """
    model, processor, model_device = _load_bbox_model(device)

    image = Image.open(img_path).convert("RGB")
    html_before, html_after, imgs = gen(
        bot=bot,
        image=image,
        model=model,
        processor=processor,
        device=model_device,
    )
    html_code = html_after or html_before

    if save_path:
        save_path_obj = Path(save_path)
        save_path_obj.parent.mkdir(parents=True, exist_ok=True)

        with open(save_path_obj, "w", encoding="utf-8") as file:
            file.write(html_code)

        sample_id = save_path_obj.stem
        artifacts_dir = save_path_obj.parent / f"{sample_id}_uicopilot"
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        with open(artifacts_dir / "before_optimize.html", "w", encoding="utf-8") as file:
            file.write(html_before)
        with open(artifacts_dir / "after_optimize.html", "w", encoding="utf-8") as file:
            file.write(html_after)

        image.save(str(artifacts_dir / "input.png"))
        for idx, cropped in enumerate(imgs):
            cropped.save(str(artifacts_dir / f"{idx}.png"))

    return html_code

