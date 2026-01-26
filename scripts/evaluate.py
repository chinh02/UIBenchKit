import os
import sys
from rapidfuzz import fuzz

from tqdm import tqdm
import json
sys.path.append("../")

# original_reference_dir = "../data/original"
original_reference_dir = "../data/demo"
test_dirs = {
    "dcgen": "../data/dcgen_demo",
    "direct": "../data/direct_demo",
}
exp_name = "test"



def code_similarity(file1, file2):
    with open(file1, 'r', encoding='utf-8') as f:
        html1 = f.read()
    with open(file2, 'r', encoding='utf-8') as f:
        html2 = f.read()
    similarity = fuzz.ratio(html1, html2)
    return similarity

def code_sim_for_dir(ref_dir, test_dir):
    test_files = os.listdir(test_dir)
    results = {}
    for file in tqdm(test_files):
        try:
            if file.endswith(".html"):
                ref_file = os.path.join(ref_dir, file)
                test_file = os.path.join(test_dir, file)
                similarity = code_similarity(ref_file, test_file)
                results[file] = similarity
        except Exception as e:
            print(f"Error processing file {file}: {e}")
            continue
    return results

def code_sim_experiment(ref_dir, test_dirs, experiment_name):
    results = {}
    for key in test_dirs:
        results[key] = code_sim_for_dir(ref_dir, test_dirs[key])
    with open(f"{experiment_name}_code_sim.json", "w") as f:
        json.dump(results, f, indent=4)

    return results
    


import os
from tqdm import tqdm
import open_clip
import torch
from PIL import Image

class CLIPScorer:
    def __init__(self, model_name='ViT-B-32-quickgelu', pretrained='openai'):
        """
        Initializes the CLIPScorer with the specified model.

        Args:
            model_name (str): The name of the CLIP model to use.
            pretrained (str): Specifies whether to load pre-trained weights.
        """
        self.device = "cuda" if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else "cpu"
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        self.model.to(self.device)

    def score(self, img1: Image.Image, img2: Image.Image) -> float:
        """
        Calculates the CLIP score (cosine similarity) between two images.

        Args:
            img1 (Image.Image): The first image as a PIL Image.
            img2 (Image.Image): The second image as a PIL Image.

        Returns:
            float: The cosine similarity score between the two images.
        """
        # Preprocess the images
        image1 = self.preprocess(img1).unsqueeze(0).to(self.device)
        image2 = self.preprocess(img2).unsqueeze(0).to(self.device)

        # Get the image features from CLIP using openclip
        with torch.no_grad():
            image1_features = self.model.encode_image(image1)
            image2_features = self.model.encode_image(image2)

        # Normalize the features to unit length
        image1_features /= image1_features.norm(dim=-1, keepdim=True)
        image2_features /= image2_features.norm(dim=-1, keepdim=True)

        # Calculate cosine similarity between the two image features
        cosine_similarity = torch.nn.functional.cosine_similarity(image1_features, image2_features)
        return cosine_similarity.item()
    

def clip_for_dir(ref_dir, test_dir):
    test_files = os.listdir(test_dir)
    results = {}
    clip_scorer = CLIPScorer()
    for file in tqdm(test_files):
        if file.endswith(".png") and not "placeholder" in file:
            ref_file = os.path.join(ref_dir, file)
            test_file = os.path.join(test_dir, file)
            ref_img = Image.open(ref_file)
            test_img = Image.open(test_file)
            similarity = clip_scorer.score(ref_img, test_img)
            results[file] = similarity
    return results

def clip_experiment(ref_dir, test_dirs, experiment_name):
    # take_screenshots_for_dir(ref_dir, replace=True)
    results = {}
    for key in test_dirs:
        results[key] = clip_for_dir(ref_dir, test_dirs[key])
    with open(f"{experiment_name}_clip.json", "w") as f:
        json.dump(results, f, indent=4)

    return results


########## Fine-grained Visual Metrics ##########
def fine_grained_experiment(ref_dir, test_dirs, experiment_name):
    """
    Run fine-grained visual evaluation using Design2Code metrics.
    
    Evaluates:
    - Block-Match: How well the generated blocks match the reference
    - Text: Text content similarity  
    - Position: Positional accuracy of elements
    - Color: Color accuracy of text elements
    - CLIP: Visual similarity using CLIP embeddings
    
    Args:
        ref_dir: Directory containing reference HTML files
        test_dirs: Dictionary mapping method names to test directories
        experiment_name: Name for the output file
    
    Returns:
        Dictionary with per-method scores
    """
    import platform
    import numpy as np
    
    if platform.system() not in ["Linux", "Darwin"]:
        raise RuntimeError(f"Fine-grained metrics only available on Linux/Mac. Current platform: {platform.system()}")
    
    # Add metric path and import
    metric_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "metric")
    if metric_path not in sys.path:
        sys.path.insert(0, metric_path)
    
    from Design2Code.metrics.visual_score import visual_eval_v3_multi
    
    # Change to metrics directory for proper relative imports
    original_cwd = os.getcwd()
    metrics_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "metric", "Design2Code")
    
    try:
        os.chdir(metrics_dir)
        
        # Get list of reference files
        ref_files = [f for f in os.listdir(ref_dir) if f.endswith(".html")]
        
        results = {}
        for method_name, test_dir in test_dirs.items():
            print(f"\nEvaluating {method_name}...")
            
            scores = {
                "block_match": {},
                "text": {},
                "position": {},
                "color": {},
                "clip": {},
                "overall": {}
            }
            
            for filename in tqdm(ref_files):
                ref_html = os.path.join(ref_dir, filename)
                test_html = os.path.join(test_dir, filename)
                
                # Convert to absolute paths from metrics_dir
                ref_html = os.path.abspath(os.path.join(original_cwd, ref_html))
                test_html = os.path.abspath(os.path.join(original_cwd, test_html))
                
                if not os.path.exists(test_html):
                    continue
                
                try:
                    input_list = [[test_html], ref_html]
                    return_score_list = visual_eval_v3_multi(input_list, debug=False)
                    
                    if return_score_list and len(return_score_list) > 0:
                        result = return_score_list[0]
                        if len(result) >= 3 and result[2]:
                            multi_score = result[2]
                            final_score = result[1]
                            
                            instance_id = filename.replace(".html", "")
                            scores["block_match"][instance_id] = float(multi_score[0])
                            scores["text"][instance_id] = float(multi_score[1])
                            scores["position"][instance_id] = float(multi_score[2])
                            scores["color"][instance_id] = float(multi_score[3])
                            scores["clip"][instance_id] = float(multi_score[4])
                            scores["overall"][instance_id] = float(final_score)
                            
                except Exception as e:
                    print(f"Error evaluating {filename}: {e}")
                    continue
            
            # Calculate averages
            method_results = {}
            for metric_name, metric_scores in scores.items():
                if metric_scores:
                    method_results[metric_name] = {
                        "scores": metric_scores,
                        "average": sum(metric_scores.values()) / len(metric_scores)
                    }
            
            results[method_name] = method_results
        
        # Save results
        with open(f"{experiment_name}_visual.json", "w") as f:
            json.dump(results, f, indent=4)
        
        return results
        
    finally:
        os.chdir(original_cwd)


if __name__ == "__main__":

    print(original_reference_dir)

    results = clip_experiment(original_reference_dir, test_dirs, exp_name)
    print("CLIP Results:")
    for key in results:
        print(key)
        print(sum(results[key].values()) / len(results[key]))


    results = code_sim_experiment(original_reference_dir, test_dirs, exp_name)
    print("Code Similarity Results:")
    for key in results:
        print(key)
        print(sum(results[key].values()) / len(results[key]))

    ########## For calculating fine-grained metrics ##########
    ########## Please run install.sh to install necessary dependencies ##########
    ########## This code can only be run on linux or mac ##########
    import platform
    if platform.system() in ["Linux", "Darwin"]:
        try:
            fine_grained_results = fine_grained_experiment(original_reference_dir, test_dirs, exp_name)
            print("\nFine-Grained Visual Results:")
            for key in fine_grained_results:
                print(f"\n{key}:")
                for metric, data in fine_grained_results[key].items():
                    print(f"  {metric}: {data['average']:.4f}")
        except Exception as e:
            print(f"\nFine-grained evaluation failed: {e}")
            print("Make sure you have run: pip install -e metric && playwright install")
    else:
        print(f"\nFine-grained metrics skipped (only available on Linux/Mac, current: {platform.system()})")