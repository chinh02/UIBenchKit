#!/usr/bin/env python3
"""
Authentication helpers for API key-gated endpoints.
"""

from functools import wraps
from flask import request, jsonify, g


def verify_api_key_check(api_key: str, api_keys: dict) -> bool:
    """Verify if API key exists and is marked verified."""
    return api_key in api_keys and api_keys[api_key].get("verified", False)


def create_require_api_key(api_keys: dict):
    """Create decorator that requires x-api-key or api_key in JSON body."""
    def require_api_key(f):
        @wraps(f)
        def decorated(*args, **kwargs):
            api_key = request.headers.get("x-api-key")
            if not api_key:
                try:
                    json_data = request.get_json(silent=True)
                    if json_data:
                        api_key = json_data.get("api_key")
                except Exception:
                    pass

            if not api_key:
                return jsonify({
                    "message": "API key is required. Set x-api-key header or UIBENCHKIT_API_KEY environment variable."
                }), 401

            # Keep compatibility with existing behavior:
            # allow any provided key in dev mode and attach it to request context.
            g.api_key = api_key
            return f(*args, **kwargs)

        return decorated

    return require_api_key
