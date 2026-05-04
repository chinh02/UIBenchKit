"""
LatCoder Pipeline
=================

Main orchestration functions for the LatCoder method.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Tuple
from PIL import Image
from bs4 import BeautifulSoup
from tqdm import tqdm

from .prompts import PROMPT_GENERATE, PROMPT_ASSEMBLE, PROMPT_REFINE, PROMPT_GET_TEXT
from .utils import remove_code_markers, extract_html_from_response, crop_image, pil_to_base64
from .scoring import get_best
from .html2shot import html2shot
from .blocker import blocker

logger = logging.getLogger(__name__)

# ============================================================
# Configuration
# ============================================================
MAX_BLOCKS_LIMIT = 25


# ============================================================
# Absolute Assembly
# ============================================================
def absolute_assemble(image: Image.Image, code_plans: List[Dict]) -> str:
    """Assemble modules using absolute CSS positioning."""
    framework_html = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <script src="https://cdn.tailwindcss.com"></script>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.3/css/all.min.css">
</head>
<body>
</body>
</html>"""
    
    soup = BeautifulSoup(framework_html, 'html.parser')
    body = soup.find('body')
    
    for node in code_plans:
        bbox = node['module_position']
        module_code = node['module_code']
        
        # Parse module code
        module_soup = BeautifulSoup(module_code, 'html.parser')
        body_tag = module_soup.find('body')
        
        if body_tag:
            body_tag.name = "div"
            inner_content = str(body_tag)
        else:
            inner_content = module_code
        
        # Calculate absolute position
        left = round(bbox[0] * image.width)
        top = round(bbox[1] * image.height)
        width = round((bbox[2] - bbox[0]) * image.width)
        height = round((bbox[3] - bbox[1]) * image.height)
        
        wrapper = f'<div style="position: absolute; overflow: hidden; border: 1px solid white; left: {left}px; top: {top}px; width: {width}px; height: {height}px;">{inner_content}</div>'
        new_content = BeautifulSoup(wrapper, 'html.parser')
        body.append(new_content)
    
    return soup.prettify()


# ============================================================
# Main Pipeline Functions
# ============================================================
def generate_module_code(bot, image: Image.Image, plans: List[List[float]], 
                         samples: int = 1, temperature: float = 0.0,
                         artifacts_dir: Path = None) -> List[Dict]:
    """
    Generate HTML code for each module/block.
    
    Args:
        bot: DCGen bot instance
        image: Full design image
        plans: List of bbox proportions
        samples: Number of samples per module (for best selection)
        temperature: LLM temperature
        artifacts_dir: Directory to save module artifacts
        
    Returns:
        List of {"module_position": bbox, "module_code": html}
    """
    code_plans = []
    
    # Create modules subfolder
    modules_dir = None
    if artifacts_dir:
        modules_dir = artifacts_dir / 'modules'
        modules_dir.mkdir(exist_ok=True)
    
    for index, plan in enumerate(tqdm(plans, desc="Generating modules")):
        module_image = crop_image(image, plan)
        module_b64 = pil_to_base64(module_image)
        
        # Save module input image
        if modules_dir:
            module_image.save(str(modules_dir / f'module_{index}_input.png'))
        
        try:
            if samples > 1:
                # Generate multiple samples and select best
                codes = []
                imgs = []
                for s in range(samples):
                    response = bot.ask(PROMPT_GENERATE, module_b64, verbose=False)
                    code = extract_html_from_response(response)
                    
                    # Validate response - retry if model failed to return HTML
                    if not code or 'unable to view' in response.lower() or '<html' not in response.lower():
                        logger.warning(f"Module {index} sample {s}: Invalid response, retrying...")
                        response = bot.ask(PROMPT_GENERATE, module_b64, verbose=False)
                        code = extract_html_from_response(response)
                    
                    if not code:
                        logger.warning(f"Module {index} sample {s}: No valid HTML extracted, using raw response")
                        code = response
                    
                    code = remove_code_markers(code)
                    codes.append(code)
                    
                    try:
                        pred_img = html2shot(code)
                        imgs.append(pred_img)
                        # Save each sample
                        if modules_dir:
                            pred_img.save(str(modules_dir / f'module_{index}_sample_{s}.png'))
                            with open(modules_dir / f'module_{index}_sample_{s}.html', 'w', encoding='utf-8') as f:
                                f.write(code)
                    except:
                        imgs.append(Image.new('RGB', (100, 100), 'white'))
                
                if len(imgs) > 1:
                    best_id, scores = get_best(module_image, imgs)
                    logger.info(f"Module {index + 1} best sample: {best_id}, scores: {scores}")
                    code = codes[best_id]
                    pred_img = imgs[best_id]
                else:
                    code = codes[0]
                    pred_img = imgs[0] if imgs else None
            else:
                response = bot.ask(PROMPT_GENERATE, module_b64, verbose=False)
                code = extract_html_from_response(response)
                
                # Validate response - retry if model failed to return HTML
                if not code or 'unable to view' in response.lower() or '<html' not in response.lower():
                    logger.warning(f"Module {index}: Invalid response, retrying...")
                    response = bot.ask(PROMPT_GENERATE, module_b64, verbose=False)
                    code = extract_html_from_response(response)
                
                if not code:
                    logger.warning(f"Module {index}: No valid HTML extracted, skipping module")
                    continue
                    
                code = remove_code_markers(code)
                
                # Render and save
                try:
                    pred_img = html2shot(code)
                except:
                    pred_img = Image.new('RGB', (100, 100), 'white')
            
            # Save final module output
            if modules_dir:
                if pred_img:
                    pred_img.save(str(modules_dir / f'module_{index}_output.png'))
                with open(modules_dir / f'module_{index}_output.html', 'w', encoding='utf-8') as f:
                    f.write(code)
            
            code_plans.append({
                "module_position": plan,
                "module_code": code
            })
            
        except Exception as e:
            logger.error(f"Failed to generate block {index + 1}: {e}")
            continue
    
    return code_plans


def agent_assemble(bot, target_image: Image.Image, code_plans: List[Dict],
                   samples: int = 1, temperature: float = 0.0,
                   artifacts_dir: Path = None) -> List[Dict]:
    """
    Use LLM agent to assemble modules into complete webpage.
    
    Args:
        bot: DCGen bot instance
        target_image: Original design image
        code_plans: Module codes and positions
        samples: Number of assembly samples
        temperature: LLM temperature
        artifacts_dir: Directory to save assembly artifacts
        
    Returns:
        List of {"html": str, "image": Image}
    """
    results = []
    target_b64 = pil_to_base64(target_image)
    
    # Format module data
    user_text = json.dumps(code_plans, indent=2)
    full_prompt = f"{PROMPT_ASSEMBLE}\n\nModule data:\n{user_text}"
    
    for s in range(samples):
        try:
            response = bot.ask(full_prompt, target_b64, verbose=False)
            html = extract_html_from_response(response)
            
            if html:
                html = remove_code_markers(html)
                img = html2shot(html)
                results.append({"html": html, "image": img})
                
                # Save assembly attempt
                if artifacts_dir:
                    img.save(str(artifacts_dir / f'agent_assembly_{s}.png'))
                    with open(artifacts_dir / f'agent_assembly_{s}.html', 'w', encoding='utf-8') as f:
                        f.write(html)
        except Exception as e:
            logger.warning(f"Agent assembly failed: {e}")
            continue
    
    return results


def refine(bot, design: Image.Image, pred_img: Image.Image, pred_code: str) -> str:
    """
    Refine generated HTML to better match design.
    
    Args:
        bot: DCGen bot instance
        design: Original design image
        pred_img: Screenshot of current generated HTML
        pred_code: Current HTML code
        
    Returns:
        Refined HTML code
    """
    # Get text content from design
    design_b64 = pil_to_base64(design)
    text_response = bot.ask(PROMPT_GET_TEXT, design_b64, verbose=False)
    
    # Refine with both images
    user_text = json.dumps({
        "task": "Based on the design image (Image 1) and the screenshot of the generated webpage (Image 2), "
                "combined with the textual_content extracted from the original webpage, "
                "refine the webpage_code according to the requirements in the prompt.",
        "webpage_code": pred_code,
        "textual_content": text_response,
    })
    
    full_prompt = f"{PROMPT_REFINE}\n\n{user_text}"
    
    # Note: Most DCGen bots support single image; for multi-image we use the design
    response = bot.ask(full_prompt, design_b64, verbose=False)
    refined = extract_html_from_response(response)
    return remove_code_markers(refined) if refined else pred_code


def pipeline(bot, design_image: Image.Image, artifacts_dir: Path = None,
             generate_samples: int = 1, assembly_samples: int = 1,
             temperature: float = 0.0, use_agent_assembly: bool = True) -> Tuple[str, Image.Image]:
    """
    Main LatCoder pipeline.
    
    Args:
        bot: DCGen bot instance
        design_image: Input design image
        artifacts_dir: Optional directory to save intermediate results
        generate_samples: Samples per module
        assembly_samples: Assembly attempts
        temperature: LLM temperature
        use_agent_assembly: Whether to use agent-based assembly
        
    Returns:
        Tuple of (final HTML code, rendered screenshot)
    """
    logger.info("Step 1: Splitting design image into blocks...")
    plans, blocks_image = blocker(design_image)
    
    if artifacts_dir:
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        blocks_image.save(str(artifacts_dir / 'blocks.png'))
        # Save block positions as JSON
        with open(artifacts_dir / 'block_positions.json', 'w', encoding='utf-8') as f:
            json.dump(plans, f, indent=2)
    
    if len(plans) > MAX_BLOCKS_LIMIT:
        raise ValueError(f"Too many blocks ({len(plans)}), max is {MAX_BLOCKS_LIMIT}")
    
    logger.info(f"Created {len(plans)} blocks")
    
    # Step 2: Generate code for each block
    logger.info("Step 2: Generating code for each block...")
    codes = generate_module_code(bot, design_image, plans, generate_samples, temperature, artifacts_dir)
    
    if not codes:
        raise ValueError("Failed to generate any module codes")
    
    pred_html = None
    pred_img = None
    
    # Step 3a: Absolute assembly
    logger.info("Step 3a: Assembling with absolute positioning...")
    assemble_res_abs = []
    abs_html = absolute_assemble(design_image, codes)
    abs_img = html2shot(abs_html)
    assemble_res_abs.append({"html": abs_html, "image": abs_img})
    
    # Save absolute assembly
    if artifacts_dir:
        abs_img.save(str(artifacts_dir / 'absolute_assembly.png'))
        with open(artifacts_dir / 'absolute_assembly.html', 'w', encoding='utf-8') as f:
            f.write(abs_html)
    
    # Step 3b: Agent assembly (optional)
    assemble_res_agent = []
    if use_agent_assembly:
        logger.info("Step 3b: Assembling with LLM agent...")
        assemble_res_agent = agent_assemble(bot, design_image, codes, assembly_samples, temperature, artifacts_dir)
    
    # Step 4: Select best result
    all_results = assemble_res_agent + assemble_res_abs
    
    if len(all_results) == 1:
        pred_html = all_results[0]['html']
        pred_img = all_results[0]['image']
    elif len(all_results) == 0:
        # Fallback: use absolute assembly directly
        pred_html = abs_html
        pred_img = abs_img
    else:
        logger.info("Step 4: Selecting best result...")
        all_images = [r['image'] for r in all_results]
        best_idx, scores = get_best(design_image, all_images)
        
        # Prefer agent result if available (matches main.py logic)
        if assemble_res_agent and len(assemble_res_agent) > 0:
            # Get best among agent results only (exclude absolute which is last)
            agent_scores = scores[:len(assemble_res_agent)]
            agent_best_idx = agent_scores.index(max(agent_scores))
            logger.info(f"Scores: {scores}, overall best: {best_idx}, agent best: {agent_best_idx}")
            pred_html = assemble_res_agent[agent_best_idx]['html']
            pred_img = assemble_res_agent[agent_best_idx]['image']
        else:
            pred_html = all_results[best_idx]['html']
            pred_img = all_results[best_idx]['image']
    
    # Save selection info to artifacts
    if artifacts_dir and len(all_results) > 1:
        with open(artifacts_dir / 'selection_info.json', 'w', encoding='utf-8') as f:
            json.dump({
                'scores': scores,
                'best_idx': best_idx,
                'selected': 'agent' if assemble_res_agent and best_idx < len(assemble_res_agent) else 'absolute'
            }, f, indent=2)
    
    return pred_html, pred_img


# ============================================================
# UIBenchKit API Integration
# ============================================================
def generate_latcoder(bot, img_path: str, save_path: str = None,
                      generate_samples: int = 1, assembly_samples: int = 1,
                      use_agent_assembly: bool = True) -> str:
    """
    Generate HTML from image using LatCoder method.
    
    This function matches the signature of generate_dcgen() and generate_single()
    for easy integration with the UIBenchKit API.
    
    Args:
        bot: DCGen bot instance
        img_path: Path to input design image
        save_path: Optional path to save output HTML (e.g., results/run_id/0.html)
        generate_samples: Number of samples per module
        assembly_samples: Number of assembly attempts
        use_agent_assembly: Whether to use agent-based assembly
        
    Returns:
        Generated HTML code
    """
    # Load image
    design_image = Image.open(img_path)
    
    # Create artifacts directory if save_path is provided
    # e.g., results/run_id/0.html -> results/run_id/0_latcoder/
    artifacts_dir = None
    if save_path:
        save_path = Path(save_path)
        sample_id = save_path.stem  # e.g., "0" from "0.html"
        artifacts_dir = save_path.parent / f'{sample_id}_latcoder'
        artifacts_dir.mkdir(parents=True, exist_ok=True)
    
    # Run pipeline
    html_code, pred_img = pipeline(
        bot=bot,
        design_image=design_image,
        artifacts_dir=artifacts_dir,
        generate_samples=generate_samples,
        assembly_samples=assembly_samples,
        temperature=0.0,
        use_agent_assembly=use_agent_assembly
    )
    
    # Save final output to main folder
    if save_path:
        # Save HTML
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(html_code)
        # Save PNG (same name with .png extension)
        png_path = save_path.with_suffix('.png')
        if pred_img:
            pred_img.save(str(png_path))
    
    return html_code

