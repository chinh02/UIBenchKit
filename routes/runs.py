#!/usr/bin/env python3
"""
Runs Routes
===========

Routes for submitting, polling, listing, and managing runs.
"""

import os
import json
import shutil
import datetime
from flask import Blueprint, request, jsonify, g

from routes.auth import require_api_key

runs_bp = Blueprint('runs', __name__)


# ============================================================
# Note: These routes use the RUNS_DB from api.py
# This file defines the route structure for future full modularization
# Currently, the routes remain in api.py due to shared state (RUNS_DB)
# ============================================================

# The following is a reference implementation that can be used
# when the codebase is fully refactored to use dependency injection
# or a proper service layer

"""
Example modular structure (for future reference):

from services.run_service import RunService
from models import Run

run_service = RunService()

@runs_bp.route("/submit", methods=["POST"])
@require_api_key
def submit():
    data = request.get_json()
    result = run_service.submit_run(data, g.api_key)
    return jsonify(result)

@runs_bp.route("/poll-jobs", methods=["GET"])
@require_api_key
def poll_jobs():
    run_id = request.args.get("run_id")
    result = run_service.poll_run(run_id, g.api_key)
    return jsonify(result)
"""

# For now, this module serves as a placeholder and documentation
# The actual routes are still in api.py to maintain backward compatibility
# and avoid breaking the shared RUNS_DB state

def create_runs_blueprint(runs_db, run_service):
    """
    Factory function to create runs blueprint with dependencies.
    
    This pattern allows for proper dependency injection when
    the codebase is ready for full modularization.
    
    Usage:
        from routes.runs import create_runs_blueprint
        runs_bp = create_runs_blueprint(RUNS_DB, run_service)
        app.register_blueprint(runs_bp)
    """
    bp = Blueprint('runs_injected', __name__)
    
    @bp.route("/submit", methods=["POST"])
    @require_api_key
    def submit():
        # Implementation would go here
        pass
    
    @bp.route("/poll-jobs", methods=["GET"])
    @require_api_key
    def poll_jobs():
        # Implementation would go here
        pass
    
    @bp.route("/get-report", methods=["POST"])
    @require_api_key
    def get_report():
        # Implementation would go here
        pass
    
    @bp.route("/list-runs", methods=["POST"])
    @require_api_key
    def list_runs():
        # Implementation would go here
        pass
    
    @bp.route("/delete-run", methods=["DELETE", "POST"])
    @require_api_key
    def delete_run():
        # Implementation would go here
        pass
    
    @bp.route("/stop-run", methods=["POST"])
    @require_api_key
    def stop_run():
        # Implementation would go here
        pass
    
    @bp.route("/resume-run", methods=["POST"])
    @require_api_key
    def resume_run():
        # Implementation would go here
        pass
    
    @bp.route("/retry-failed", methods=["POST"])
    @require_api_key
    def retry_failed():
        # Implementation would go here
        pass
    
    @bp.route("/rerun-evaluation", methods=["POST"])
    @require_api_key
    def rerun_evaluation():
        # Implementation would go here
        pass
    
    return bp
