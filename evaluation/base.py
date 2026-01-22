#!/usr/bin/env python3
"""
Base Evaluator
==============

Abstract base class for all evaluation metrics.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
import os


@dataclass
class EvaluationResult:
    """
    Container for evaluation results.
    """
    sample_id: str
    metric_name: str
    scores: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    success: bool = True
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "metric_name": self.metric_name,
            "scores": self.scores,
            "metadata": self.metadata,
            "success": self.success,
            "error": self.error
        }


class BaseEvaluator(ABC):
    """
    Abstract base class for evaluation metrics.
    
    All evaluators should inherit from this class and implement
    the `evaluate_sample` method.
    """
    
    metric_name: str = "base"
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the evaluator.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self._initialized = False
    
    def initialize(self) -> None:
        """
        Initialize any resources needed for evaluation.
        Override this method to set up models, load weights, etc.
        """
        self._initialized = True
    
    def cleanup(self) -> None:
        """
        Clean up any resources.
        Override this method to release resources.
        """
        self._initialized = False
    
    @abstractmethod
    def evaluate_sample(
        self,
        generated_html_path: str,
        reference_image_path: str,
        generated_screenshot_path: Optional[str] = None,
        **kwargs
    ) -> EvaluationResult:
        """
        Evaluate a single sample.
        
        Args:
            generated_html_path: Path to the generated HTML file
            reference_image_path: Path to the reference screenshot
            generated_screenshot_path: Optional path to the generated screenshot
            **kwargs: Additional arguments for specific evaluators
        
        Returns:
            EvaluationResult with scores and metadata
        """
        pass
    
    def evaluate_batch(
        self,
        samples: List[Dict[str, str]],
        **kwargs
    ) -> List[EvaluationResult]:
        """
        Evaluate a batch of samples.
        
        Args:
            samples: List of dicts with 'generated_html_path', 'reference_image_path', etc.
            **kwargs: Additional arguments passed to evaluate_sample
        
        Returns:
            List of EvaluationResult objects
        """
        results = []
        
        for sample in samples:
            try:
                result = self.evaluate_sample(**sample, **kwargs)
                results.append(result)
            except Exception as e:
                sample_id = os.path.basename(sample.get('generated_html_path', 'unknown'))
                results.append(EvaluationResult(
                    sample_id=sample_id,
                    metric_name=self.metric_name,
                    success=False,
                    error=str(e)
                ))
        
        return results
    
    def aggregate_results(
        self,
        results: List[EvaluationResult]
    ) -> Dict[str, Any]:
        """
        Aggregate results across multiple samples.
        
        Args:
            results: List of evaluation results
        
        Returns:
            Dictionary with aggregated statistics
        """
        successful = [r for r in results if r.success]
        
        if not successful:
            return {
                "metric_name": self.metric_name,
                "sample_count": len(results),
                "success_count": 0,
                "averages": {},
                "error": "No successful evaluations"
            }
        
        # Collect all score keys
        all_keys = set()
        for r in successful:
            all_keys.update(r.scores.keys())
        
        # Calculate averages
        averages = {}
        for key in all_keys:
            values = [r.scores.get(key, 0) for r in successful if key in r.scores]
            if values:
                averages[key] = sum(values) / len(values)
        
        return {
            "metric_name": self.metric_name,
            "sample_count": len(results),
            "success_count": len(successful),
            "failure_count": len(results) - len(successful),
            "averages": averages
        }
    
    def __enter__(self):
        self.initialize()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
        return False
