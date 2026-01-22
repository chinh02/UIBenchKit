#!/usr/bin/env python3
"""
Dataset Routes
==============

Dataset listing, information, and sample management.
"""

import os
import json
from flask import Blueprint, request, jsonify

from dataset_manager import DatasetManager, DATASETS_CONFIG, get_dataset_manager

datasets_bp = Blueprint('datasets', __name__)


# ============================================================
# Routes
# ============================================================

@datasets_bp.route("/datasets", methods=["GET"])
def list_datasets():
    """
    List all available datasets.
    
    Response:
        {
            "datasets": {
                "design2code": {...},
                "dcgen": {...}
            }
        }
    """
    result = {}
    for name, config in DATASETS_CONFIG.items():
        dm = get_dataset_manager(name)
        result[name] = {
            "name": config.get("name", name),
            "description": config.get("description", ""),
            "sample_count": dm.get_sample_count() if dm else 0,
            "available": dm.is_downloaded() if dm else False
        }
    
    return jsonify({"datasets": result})


@datasets_bp.route("/datasets/<dataset_name>", methods=["GET"])
def get_dataset(dataset_name: str):
    """
    Get detailed information about a specific dataset.
    
    Response:
        {
            "name": "design2code",
            "description": "...",
            "sample_count": 500,
            "samples": ["sample1", "sample2", ...]
        }
    """
    if dataset_name not in DATASETS_CONFIG:
        return jsonify({"error": f"Unknown dataset: {dataset_name}"}), 404
    
    dm = get_dataset_manager(dataset_name)
    if not dm:
        return jsonify({"error": f"Dataset {dataset_name} not configured"}), 500
    
    config = DATASETS_CONFIG[dataset_name]
    
    result = {
        "name": config.get("name", dataset_name),
        "description": config.get("description", ""),
        "sample_count": dm.get_sample_count(),
        "available": dm.is_downloaded(),
        "samples": dm.list_samples()[:100]  # Limit to first 100
    }
    
    return jsonify(result)


@datasets_bp.route("/datasets/<dataset_name>/samples", methods=["GET"])
def list_samples(dataset_name: str):
    """
    List samples in a dataset with pagination.
    
    Query params:
        - limit: Max samples to return (default 100)
        - offset: Number of samples to skip (default 0)
    
    Response:
        {
            "dataset": "design2code",
            "samples": [...],
            "total": 500,
            "limit": 100,
            "offset": 0
        }
    """
    if dataset_name not in DATASETS_CONFIG:
        return jsonify({"error": f"Unknown dataset: {dataset_name}"}), 404
    
    dm = get_dataset_manager(dataset_name)
    if not dm:
        return jsonify({"error": f"Dataset {dataset_name} not configured"}), 500
    
    limit = request.args.get("limit", 100, type=int)
    offset = request.args.get("offset", 0, type=int)
    
    all_samples = dm.list_samples()
    samples = all_samples[offset:offset + limit]
    
    return jsonify({
        "dataset": dataset_name,
        "samples": samples,
        "total": len(all_samples),
        "limit": limit,
        "offset": offset
    })


@datasets_bp.route("/datasets/<dataset_name>/samples/<sample_id>", methods=["GET"])
def get_sample(dataset_name: str, sample_id: str):
    """
    Get information about a specific sample.
    
    Response:
        {
            "sample_id": "sample1",
            "has_image": true,
            "has_html": true,
            "image_path": "/path/to/image.png",
            "html_path": "/path/to/sample.html"
        }
    """
    if dataset_name not in DATASETS_CONFIG:
        return jsonify({"error": f"Unknown dataset: {dataset_name}"}), 404
    
    dm = get_dataset_manager(dataset_name)
    if not dm:
        return jsonify({"error": f"Dataset {dataset_name} not configured"}), 500
    
    sample_info = dm.get_sample_info(sample_id)
    if not sample_info:
        return jsonify({"error": f"Sample {sample_id} not found"}), 404
    
    return jsonify(sample_info)
