#!/usr/bin/env python3
"""
DCGen Routes Package
====================

Flask route blueprints for the DCGen API.
"""

from flask import Blueprint

# Import all blueprints
from .auth import auth_bp
from .datasets import datasets_bp
from .runs import runs_bp
from .evaluation import evaluation_bp

__all__ = [
    'auth_bp',
    'datasets_bp',
    'runs_bp',
    'evaluation_bp',
]


def register_all_blueprints(app):
    """Register all blueprints with a Flask app."""
    app.register_blueprint(auth_bp)
    app.register_blueprint(datasets_bp)
    app.register_blueprint(runs_bp)
    app.register_blueprint(evaluation_bp)
