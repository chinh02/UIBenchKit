#!/usr/bin/env python3
"""
CLIP Score Evaluator
====================

Evaluates visual similarity using CLIP embeddings.
Uses open_clip with ViT-B-32-quickgelu model for consistency with Design2Code metrics.
"""

import os
from typing import Dict, Any, Optional

from .base import BaseEvaluator, EvaluationResult


class CLIPScoreEvaluator(BaseEvaluator):
    """
    Evaluator using CLIP for visual similarity.
    
    Computes cosine similarity between CLIP embeddings of
    reference and generated screenshots.
    
    Uses open_clip with ViT-B-32-quickgelu model (same as Design2Code).
    """
    
    metric_name = "clip_score"
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.model = None
        self.preprocess = None
        self.device = None
        self.model_name = config.get('model_name', 'ViT-B-32-quickgelu') if config else 'ViT-B-32-quickgelu'
        self.pretrained = config.get('pretrained', 'openai') if config else 'openai'
    
    def initialize(self) -> None:
        """Load CLIP model using open_clip."""
        try:
            import torch
            import open_clip
            
            # Device detection: CUDA > MPS (Apple Silicon) > CPU
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = "cpu"
            
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                self.model_name, pretrained=self.pretrained
            )
            self.model.to(self.device)
            self._initialized = True
            
        except ImportError as e:
            raise ImportError(
                "CLIP evaluation requires open_clip and torch. "
                "Install with: pip install open_clip_torch torch"
            ) from e
    
    def cleanup(self) -> None:
        """Release model from memory."""
        if self.model is not None:
            del self.model
            self.model = None
        if self.preprocess is not None:
            del self.preprocess
            self.preprocess = None
        
        # Clear CUDA cache if available
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        
        self._initialized = False
    
    def _get_image_embedding(self, image_path: str):
        """Get CLIP embedding for an image."""
        import torch
        from PIL import Image
        
        image = Image.open(image_path)
        image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            image_features = self.model.encode_image(image_tensor)
            # Normalize
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        return image_features
    
    def evaluate_sample(
        self,
        generated_html_path: str,
        reference_image_path: str,
        generated_screenshot_path: Optional[str] = None,
        **kwargs
    ) -> EvaluationResult:
        """
        Evaluate CLIP similarity between reference and generated screenshots.
        
        Args:
            generated_html_path: Path to generated HTML (used for sample ID)
            reference_image_path: Path to reference screenshot
            generated_screenshot_path: Path to generated screenshot
        
        Returns:
            EvaluationResult with CLIP similarity score
        """
        import torch
        
        sample_id = os.path.splitext(os.path.basename(generated_html_path))[0]
        
        if not self._initialized:
            self.initialize()
        
        # Determine generated screenshot path if not provided
        if not generated_screenshot_path:
            # Try common naming patterns
            html_dir = os.path.dirname(generated_html_path)
            possible_paths = [
                os.path.join(html_dir, f"{sample_id}.png"),
                os.path.join(html_dir, f"{sample_id}_screenshot.png"),
                generated_html_path.replace('.html', '.png')
            ]
            for path in possible_paths:
                if os.path.exists(path):
                    generated_screenshot_path = path
                    break
        
        if not generated_screenshot_path or not os.path.exists(generated_screenshot_path):
            return EvaluationResult(
                sample_id=sample_id,
                metric_name=self.metric_name,
                success=False,
                error="Generated screenshot not found"
            )
        
        if not os.path.exists(reference_image_path):
            return EvaluationResult(
                sample_id=sample_id,
                metric_name=self.metric_name,
                success=False,
                error="Reference image not found"
            )
        
        try:
            ref_embedding = self._get_image_embedding(reference_image_path)
            gen_embedding = self._get_image_embedding(generated_screenshot_path)
            
            # Cosine similarity (embeddings are already normalized)
            similarity = torch.nn.functional.cosine_similarity(
                ref_embedding, gen_embedding
            ).item()
            
            return EvaluationResult(
                sample_id=sample_id,
                metric_name=self.metric_name,
                scores={
                    "clip_score": similarity
                },
                metadata={
                    "reference_image": os.path.basename(reference_image_path),
                    "generated_image": os.path.basename(generated_screenshot_path),
                    "model": self.model_name
                }
            )
            
        except Exception as e:
            return EvaluationResult(
                sample_id=sample_id,
                metric_name=self.metric_name,
                success=False,
                error=str(e)
            )
