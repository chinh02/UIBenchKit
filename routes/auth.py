#!/usr/bin/env python3
"""
Authentication and system routes.
"""

import datetime
import hashlib
import uuid
from flask import Blueprint, jsonify, request, g


def create_auth_blueprint(
    *,
    require_api_key,
    api_keys: dict,
    api_version: str,
    supported_datasets: list,
    supported_models: list,
    model_families: dict,
    supported_methods: list,
    provider_env_status: dict = None,
):
    """Create the auth/system blueprint with injected dependencies."""
    bp = Blueprint("auth", __name__)

    @bp.route("/health", methods=["GET"])
    def health():
        return jsonify(
            {
                "status": "healthy",
                "version": api_version,
                "timestamp": datetime.datetime.now().isoformat(),
                "supported_datasets": supported_datasets,
                "supported_model_families": supported_models,
                "supported_model_versions": model_families,
                "supported_methods": supported_methods,
                "provider_env_status": provider_env_status or {},
            }
        )

    @bp.route("/gen-api-key", methods=["POST"])
    def gen_api_key():
        data = request.get_json()
        if not data or "email" not in data:
            return jsonify({"message": "Email is required"}), 400

        email = data["email"]
        key_raw = f"{email}-{datetime.datetime.now().isoformat()}-{uuid.uuid4()}"
        api_key = hashlib.sha256(key_raw.encode()).hexdigest()[:32]

        api_keys[api_key] = {
            "email": email,
            "verified": True,
            "created_at": datetime.datetime.now().isoformat(),
        }

        return jsonify(
            {
                "message": f"API key generated for {email}. Key is auto-verified for development.",
                "api_key": api_key,
            }
        )

    @bp.route("/verify-api-key", methods=["POST"])
    @require_api_key
    def verify_api_key_endpoint():
        return jsonify(
            {
                "message": "API key is valid and verified.",
                "email": api_keys[g.api_key].get("email"),
            }
        )

    @bp.route("/get-quotas", methods=["GET"])
    @require_api_key
    def get_quotas():
        return jsonify(
            {
                "remaining_quotas": {
                    "gemini": {"runs": 1000},
                    "gpt4": {"runs": 1000},
                    "claude": {"runs": 1000},
                    "qwen": {"runs": 1000},
                }
            }
        )

    return bp
