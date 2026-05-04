"""
LatCoder Scoring
================

Visual similarity scoring using MAE and CLIP metrics.
Matches original latcoder implementation exactly.
"""

import logging
from typing import List, Tuple
import importlib.util
from PIL import Image
import numpy as np
import torch

logger = logging.getLogger(__name__)

# Global CLIP model cache (matches original)
CLIP_MODEL = None
CLIP_PREPROCESS = None
CLIP_BACKEND = None
_CLIP_DISABLED_REASON = None


def process_imgs(image1: Image.Image, image2: Image.Image, max_size: int = 512):
    """
    Process images for comparison - pad to same size and resize.
    Matches original latcoder/evaluation/mrweb/emd_similarity.py
    """
    # Convert to RGB
    image1 = image1.convert('RGB')
    image2 = image2.convert('RGB')
    
    # Get the original sizes
    width1, height1 = image1.size
    width2, height2 = image2.size

    # Determine the new dimensions (max of both images' width and height)
    new_width = max(width1, width2)
    new_height = max(height1, height2)

    # Pad images to the new dimensions with random values
    def pad_image(image, new_width, new_height):
        random_padding = np.random.randint(0, 256, (new_height, new_width, 3), dtype=np.uint8)
        padded_image = Image.fromarray(random_padding)
        padded_image.paste(image, (0, 0))
        return padded_image

    padded_image1 = pad_image(image1, new_width, new_height)
    padded_image2 = pad_image(image2, new_width, new_height)

    # Calculate the aspect ratio for resizing to the max size
    aspect_ratio = min(max_size / new_width, max_size / new_height)
    new_size = (int(new_width * aspect_ratio), int(new_height * aspect_ratio))

    # Resize the padded images to the specified max size
    resized_image1 = padded_image1.resize(new_size, Image.LANCZOS)
    resized_image2 = padded_image2.resize(new_size, Image.LANCZOS)

    # Convert the images to numpy arrays with dtype int16
    array1 = np.array(resized_image1).astype(np.int16)
    array2 = np.array(resized_image2).astype(np.int16)

    return array1, array2


def mae_score(img1: Image.Image, img2: Image.Image) -> float:
    """
    Calculate Mean Absolute Error between two images.
    Matches original latcoder/evaluation/mrweb/study.py
    """
    arr1, arr2 = process_imgs(img1, img2, 512)
    mae = np.mean(np.abs(arr1 - arr2))
    return float(mae)


def clip_encode(ims: list, device: str = 'cuda'):
    """
    Encode images using CLIP.
    Matches original latcoder/evaluation/metrics.py
    """
    global CLIP_MODEL, CLIP_PREPROCESS, CLIP_BACKEND, _CLIP_DISABLED_REASON
    
    if CLIP_MODEL is None:
        # Backend 1: OpenAI CLIP package (module name: clip)
        if importlib.util.find_spec("clip") is not None:
            try:
                import clip

                CLIP_MODEL, CLIP_PREPROCESS = clip.load("ViT-B/32", device=device)
                CLIP_BACKEND = "clip"
            except Exception as error:
                logger.warning(f"Failed to initialize OpenAI CLIP backend: {error}")

        # Backend 2: open_clip_torch package (module name: open_clip)
        if CLIP_MODEL is None and importlib.util.find_spec("open_clip") is not None:
            try:
                import open_clip

                model, _, preprocess = open_clip.create_model_and_transforms(
                    "ViT-B-32",
                    pretrained="openai",
                    device=device,
                )
                model.eval()
                CLIP_MODEL = model
                CLIP_PREPROCESS = preprocess
                CLIP_BACKEND = "open_clip"
            except Exception as error:
                logger.warning(f"Failed to initialize open_clip backend: {error}")

        if CLIP_MODEL is None:
            _CLIP_DISABLED_REASON = (
                "Neither OpenAI CLIP ('clip') nor open_clip ('open_clip_torch') "
                "is usable in this environment."
            )
            raise RuntimeError(_CLIP_DISABLED_REASON)
    
    with torch.no_grad():
        img_tmps = torch.stack([CLIP_PREPROCESS(im) for im in ims]).to(device)
        img_feas = CLIP_MODEL.encode_image(img_tmps).cpu()
    return img_feas


def clip_similarity(img1: Image.Image, img2: Image.Image, device: str = 'cuda') -> float:
    """
    Calculate CLIP cosine similarity between two images.
    Matches original latcoder/evaluation/metrics.py clip_sim function.
    """
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'
    
    # Convert to RGB for CLIP
    img1_rgb = img1.convert('RGB')
    img2_rgb = img2.convert('RGB')
    
    feas = clip_encode([img1_rgb, img2_rgb], device)
    return torch.nn.functional.cosine_similarity(feas[0], feas[1], dim=0).item()


def mae_only_score(mae: float) -> float:
    """
    Fallback score when CLIP is unavailable.
    Higher is better, normalized to roughly [0, 1].
    """
    return max(0.0, 1.0 - (mae / 255.0))


def verify_score(mae: float, clip_sim: float, weights: Tuple[float, float] = (0.5, 0.5)) -> float:
    """Compute composite similarity score from MAE and CLIP."""
    w1, w2 = weights
    composite = w1 * (1 - mae / 255) + w2 * (clip_sim ** 0.5)
    return composite


def evaluate_images(ref_img: Image.Image, cand_img: Image.Image) -> Tuple[float, float]:
    """Evaluate candidate image against reference."""
    mae = mae_score(ref_img, cand_img)
    clip = None
    try:
        clip = clip_similarity(ref_img, cand_img)
    except Exception as error:
        logger.warning(f"CLIP similarity unavailable, falling back to MAE-only scoring: {error}")
    return mae, clip


def get_best(ref_img: Image.Image, cand_imgs: List[Image.Image]) -> Tuple[int, List[float]]:
    """Select the best candidate image based on composite score."""
    if len(cand_imgs) == 1:
        return 0, [1.0]
    
    scores_data = {'MAE': [], 'CLIP': []}
    for cand in cand_imgs:
        mae, clip = evaluate_images(ref_img, cand)
        scores_data['MAE'].append(mae)
        scores_data['CLIP'].append(clip)
    
    # If CLIP failed for any candidate, use MAE-only scoring for consistent ranking.
    if any(score is None for score in scores_data['CLIP']):
        final_scores = [mae_only_score(mae) for mae in scores_data['MAE']]
        logger.info("Using MAE-only ranking because CLIP is unavailable.")
    else:
        weights = (0.5, 0.5)
        final_scores = [
            verify_score(scores_data['MAE'][i], scores_data['CLIP'][i], weights)
            for i in range(len(cand_imgs))
        ]
    
    logger.info(f"Candidate scores: {final_scores}")
    best_idx = final_scores.index(max(final_scores))
    return best_idx, final_scores
