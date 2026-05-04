#!/usr/bin/env python3
"""
Dataset routes.
"""

from flask import Blueprint, jsonify, request


def create_datasets_blueprint(*, supported_datasets: list, get_dataset_manager):
    """Create dataset blueprint with injected dependencies."""
    bp = Blueprint("datasets", __name__)

    @bp.route("/datasets", methods=["GET"])
    def list_datasets():
        dm = get_dataset_manager()
        datasets = dm.list_available_datasets()
        return jsonify({"datasets": datasets, "supported": supported_datasets})

    @bp.route("/datasets/<dataset_name>", methods=["GET"])
    def get_dataset_info(dataset_name: str):
        if dataset_name not in supported_datasets:
            return (
                jsonify(
                    {
                        "message": f"Unknown dataset: {dataset_name}. Available: {', '.join(supported_datasets)}"
                    }
                ),
                400,
            )

        dm = get_dataset_manager()
        info = dm.get_dataset_info(dataset_name)

        if not info:
            return jsonify(
                {
                    "message": f"Dataset {dataset_name} not downloaded. Please contact administrator.",
                    "downloaded": False,
                }
            )

        return jsonify({"downloaded": True, "info": info})

    @bp.route("/datasets/<dataset_name>/samples", methods=["GET"])
    def get_dataset_samples(dataset_name: str):
        if dataset_name not in supported_datasets:
            return jsonify({"message": f"Unknown dataset: {dataset_name}"}), 400

        dm = get_dataset_manager()

        try:
            limit = request.args.get("limit", type=int)
            offset = request.args.get("offset", 0, type=int)
            samples = dm.get_samples(dataset_name, limit=limit, offset=offset)
            total = len(dm.get_sample_ids(dataset_name))
            return jsonify({"samples": samples, "total": total, "offset": offset, "limit": limit})
        except ValueError as e:
            return jsonify({"message": str(e)}), 400

    return bp

