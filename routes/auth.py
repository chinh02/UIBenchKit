#!/usr/bin/env python3
"""
Authentication Routes
=====================

API key generation, verification, and quota management.
"""

import os
import json
import hashlib
import datetime
from functools import wraps
from flask import Blueprint, request, jsonify, g

auth_bp = Blueprint('auth', __name__)

# ============================================================
# API Keys Storage
# ============================================================
API_KEYS = {}
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")

# Default API key for development
DEFAULT_API_KEY = os.getenv("DCGEN_API_KEY", "dev-api-key-12345")
API_KEYS[DEFAULT_API_KEY] = {
    "email": "dev@localhost",
    "verified": True,
    "created_at": datetime.datetime.now().isoformat()
}


def require_api_key(f):
    """Decorator to require API key authentication."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        api_key = request.headers.get('X-API-Key') or request.headers.get('Authorization', '').replace('Bearer ', '')
        
        if not api_key:
            return jsonify({"error": "API key required. Set X-API-Key header."}), 401
        
        # Check if API key is valid
        if api_key not in API_KEYS and api_key != DEFAULT_API_KEY:
            return jsonify({"error": "Invalid API key"}), 403
        
        g.api_key = api_key
        return f(*args, **kwargs)
    
    return decorated_function


def get_api_key():
    """Get the current API key from the request context."""
    return getattr(g, 'api_key', None)


# ============================================================
# Routes
# ============================================================

@auth_bp.route("/gen-api-key", methods=["POST"])
def gen_api_key():
    """
    Generate a new API key.
    
    Request:
        {
            "email": "user@example.com"
        }
    
    Response:
        {
            "api_key": "dcgen-xxxxx",
            "email": "user@example.com",
            "created_at": "2024-01-01T00:00:00"
        }
    """
    data = request.json or {}
    email = data.get("email", "anonymous@localhost")
    
    # Generate API key from email + timestamp
    timestamp = datetime.datetime.now().isoformat()
    raw = f"{email}:{timestamp}:{os.urandom(16).hex()}"
    api_key = f"dcgen-{hashlib.sha256(raw.encode()).hexdigest()[:32]}"
    
    API_KEYS[api_key] = {
        "email": email,
        "verified": True,
        "created_at": timestamp
    }
    
    return jsonify({
        "api_key": api_key,
        "email": email,
        "created_at": timestamp
    })


@auth_bp.route("/verify-api-key", methods=["POST"])
@require_api_key
def verify_api_key():
    """
    Verify an API key.
    
    Response:
        {"valid": true, "email": "user@example.com"}
    """
    api_key = g.api_key
    key_info = API_KEYS.get(api_key, {})
    
    return jsonify({
        "valid": True,
        "email": key_info.get("email", "unknown")
    })


@auth_bp.route("/get-quotas", methods=["GET"])
@require_api_key
def get_quotas():
    """
    Get API usage quotas for the current user.
    
    Response:
        {
            "runs_total": 100,
            "runs_used": 5,
            "runs_remaining": 95
        }
    """
    api_key = g.api_key
    
    # Count runs for this API key
    runs_used = 0
    for run_id in os.listdir(RESULTS_DIR) if os.path.exists(RESULTS_DIR) else []:
        run_dir = os.path.join(RESULTS_DIR, run_id)
        if not os.path.isdir(run_dir):
            continue
        
        meta_path = os.path.join(run_dir, "run_metadata.json")
        if os.path.exists(meta_path):
            try:
                with open(meta_path, 'r') as f:
                    meta = json.load(f)
                if meta.get("api_key") == api_key:
                    runs_used += 1
            except:
                pass
    
    return jsonify({
        "runs_total": 1000,
        "runs_used": runs_used,
        "runs_remaining": 1000 - runs_used
    })
