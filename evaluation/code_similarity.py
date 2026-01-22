#!/usr/bin/env python3
"""
Code Similarity Evaluator
=========================

Evaluates generated HTML code similarity against reference.
Uses text-based metrics like edit distance, token overlap, and fuzzy matching.
"""

import os
import re
from typing import Dict, Any, Optional
from difflib import SequenceMatcher

from .base import BaseEvaluator, EvaluationResult


class CodeSimilarityEvaluator(BaseEvaluator):
    """
    Evaluator for code similarity metrics.
    
    Computes:
    - Fuzzy ratio (rapidfuzz) - 0-100 scale, same as original api.py
    - Normalized edit distance
    - Token-level similarity
    - Structure similarity (tag-based)
    """
    
    metric_name = "code_similarity"
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.normalize_whitespace = config.get('normalize_whitespace', True) if config else True
    
    def _normalize_html(self, html: str) -> str:
        """Normalize HTML for comparison."""
        if self.normalize_whitespace:
            # Remove excessive whitespace
            html = re.sub(r'\s+', ' ', html)
            # Normalize around tags
            html = re.sub(r'>\s+<', '><', html)
        return html.strip().lower()
    
    def _extract_tags(self, html: str) -> list:
        """Extract HTML tags for structure comparison."""
        return re.findall(r'<[^>]+>', html)
    
    def _tokenize(self, html: str) -> list:
        """Tokenize HTML for token-level comparison."""
        # Split by whitespace, tags, and punctuation
        tokens = re.findall(r'<[^>]+>|[\w]+|[^\s\w]', html)
        return tokens
    
    def evaluate_sample(
        self,
        generated_html_path: str,
        reference_html_path: Optional[str] = None,
        reference_image_path: str = None,  # Not used but required by interface
        generated_screenshot_path: Optional[str] = None,
        **kwargs
    ) -> EvaluationResult:
        """
        Evaluate code similarity between generated and reference HTML.
        
        Args:
            generated_html_path: Path to generated HTML file
            reference_html_path: Path to reference HTML file
            reference_image_path: Not used for code similarity
            generated_screenshot_path: Not used for code similarity
        
        Returns:
            EvaluationResult with similarity scores
        """
        sample_id = os.path.splitext(os.path.basename(generated_html_path))[0]
        
        # If no reference HTML, we can't compute code similarity
        if not reference_html_path or not os.path.exists(reference_html_path):
            return EvaluationResult(
                sample_id=sample_id,
                metric_name=self.metric_name,
                success=False,
                error="Reference HTML not provided or not found"
            )
        
        try:
            with open(generated_html_path, 'r', encoding='utf-8') as f:
                generated_html = f.read()
            
            with open(reference_html_path, 'r', encoding='utf-8') as f:
                reference_html = f.read()
            
            # Calculate fuzz ratio using rapidfuzz (0-100 scale, same as original api.py)
            try:
                from rapidfuzz import fuzz
                fuzz_ratio = fuzz.ratio(generated_html, reference_html)
            except ImportError:
                # Fallback to SequenceMatcher if rapidfuzz not available
                fuzz_ratio = SequenceMatcher(None, generated_html, reference_html).ratio() * 100
            
            # Normalize for detailed metrics
            gen_normalized = self._normalize_html(generated_html)
            ref_normalized = self._normalize_html(reference_html)
            
            # Calculate sequence similarity (edit distance based)
            sequence_similarity = SequenceMatcher(
                None, gen_normalized, ref_normalized
            ).ratio()
            
            # Token-level similarity
            gen_tokens = self._tokenize(gen_normalized)
            ref_tokens = self._tokenize(ref_normalized)
            token_similarity = SequenceMatcher(
                None, gen_tokens, ref_tokens
            ).ratio()
            
            # Structure similarity (tag-level)
            gen_tags = self._extract_tags(gen_normalized)
            ref_tags = self._extract_tags(ref_normalized)
            structure_similarity = SequenceMatcher(
                None, gen_tags, ref_tags
            ).ratio()
            
            # Jaccard similarity for tokens
            gen_token_set = set(gen_tokens)
            ref_token_set = set(ref_tokens)
            intersection = gen_token_set & ref_token_set
            union = gen_token_set | ref_token_set
            jaccard_similarity = len(intersection) / len(union) if union else 0
            
            return EvaluationResult(
                sample_id=sample_id,
                metric_name=self.metric_name,
                scores={
                    "fuzz_ratio": fuzz_ratio,  # 0-100 scale (primary, same as original)
                    "sequence_similarity": sequence_similarity * 100,  # 0-100 scale
                    "token_similarity": token_similarity * 100,  # 0-100 scale
                    "structure_similarity": structure_similarity * 100,  # 0-100 scale
                    "jaccard_similarity": jaccard_similarity * 100,  # 0-100 scale
                    "overall": fuzz_ratio  # Use fuzz_ratio as primary for backward compatibility
                },
                metadata={
                    "generated_tokens": len(gen_tokens),
                    "reference_tokens": len(ref_tokens),
                    "generated_tags": len(gen_tags),
                    "reference_tags": len(ref_tags)
                }
            )
            
        except Exception as e:
            return EvaluationResult(
                sample_id=sample_id,
                metric_name=self.metric_name,
                success=False,
                error=str(e)
            )
