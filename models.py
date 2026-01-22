#!/usr/bin/env python3
"""
DCGen Run Model
===============

Data model for experiment runs with persistence operations.
"""

import os
import json
import time
from datetime import datetime
from typing import Optional, List, Dict, Any


class RunStatus:
    """Enum-like class for run statuses."""
    PENDING = "pending"
    RUNNING = "running"
    EVALUATING = "evaluating"
    COMPLETED = "completed"
    FAILED = "failed"
    STOPPED = "stopped"


class Run:
    """
    Represents a single experiment run with persistence capabilities.
    """
    
    def __init__(self, run_id: str, results_dir: str):
        self.run_id = run_id
        self.results_dir = results_dir
        self.run_dir = os.path.join(results_dir, run_id)
        self._data = None
    
    @property
    def run_json_path(self) -> str:
        return os.path.join(self.run_dir, "run.json")
    
    @property
    def evaluation_json_path(self) -> str:
        return os.path.join(self.run_dir, "evaluation.json")
    
    @property
    def token_usage_path(self) -> str:
        return os.path.join(self.run_dir, "token_usage.json")
    
    @property
    def exists(self) -> bool:
        return os.path.exists(self.run_json_path)
    
    def load(self) -> Optional[Dict[str, Any]]:
        """Load run data from disk."""
        if not self.exists:
            return None
        
        with open(self.run_json_path, 'r') as f:
            self._data = json.load(f)
        return self._data
    
    def save(self, data: Optional[Dict[str, Any]] = None) -> None:
        """Save run data to disk."""
        if data is not None:
            self._data = data
        
        if self._data is None:
            raise ValueError("No data to save")
        
        os.makedirs(self.run_dir, exist_ok=True)
        
        with open(self.run_json_path, 'w') as f:
            json.dump(self._data, f, indent=2)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a value from run data."""
        if self._data is None:
            self.load()
        return self._data.get(key, default) if self._data else default
    
    def set(self, key: str, value: Any, persist: bool = True) -> None:
        """Set a value in run data."""
        if self._data is None:
            self.load() or {}
            if self._data is None:
                self._data = {}
        
        self._data[key] = value
        
        if persist:
            self.save()
    
    def update(self, updates: Dict[str, Any], persist: bool = True) -> None:
        """Update multiple values in run data."""
        if self._data is None:
            self.load()
            if self._data is None:
                self._data = {}
        
        self._data.update(updates)
        
        if persist:
            self.save()
    
    @property
    def status(self) -> Optional[str]:
        return self.get("status")
    
    @status.setter
    def status(self, value: str) -> None:
        self.set("status", value)
    
    @property
    def progress(self) -> Dict[str, int]:
        return self.get("progress", {"completed": 0, "total": 0, "failed": 0})
    
    def update_progress(self, completed: int = None, total: int = None, 
                        failed: int = None, persist: bool = True) -> None:
        """Update progress counters."""
        current = self.progress
        
        if completed is not None:
            current["completed"] = completed
        if total is not None:
            current["total"] = total
        if failed is not None:
            current["failed"] = failed
        
        self.set("progress", current, persist=persist)
    
    def load_evaluation(self) -> Optional[Dict[str, Any]]:
        """Load evaluation data."""
        if not os.path.exists(self.evaluation_json_path):
            return None
        
        with open(self.evaluation_json_path, 'r') as f:
            return json.load(f)
    
    def save_evaluation(self, data: Dict[str, Any]) -> None:
        """Save evaluation data."""
        with open(self.evaluation_json_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load_token_usage(self) -> Optional[Dict[str, Any]]:
        """Load token usage data."""
        if not os.path.exists(self.token_usage_path):
            return None
        
        with open(self.token_usage_path, 'r') as f:
            return json.load(f)
    
    def save_token_usage(self, data: Dict[str, Any]) -> None:
        """Save token usage data."""
        with open(self.token_usage_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def get_html_files(self) -> List[str]:
        """Get list of generated HTML files in the run directory."""
        html_files = []
        
        if not os.path.exists(self.run_dir):
            return html_files
        
        for f in os.listdir(self.run_dir):
            if f.endswith('.html') and not f.startswith('_'):
                html_files.append(os.path.join(self.run_dir, f))
        
        return sorted(html_files)
    
    def get_sample_ids(self) -> List[str]:
        """Get list of sample IDs that have been processed."""
        sample_ids = set()
        
        for html_file in self.get_html_files():
            # Extract sample ID from filename (e.g., "sample_001.html" -> "sample_001")
            basename = os.path.basename(html_file)
            sample_id = os.path.splitext(basename)[0]
            sample_ids.add(sample_id)
        
        return sorted(list(sample_ids))
    
    def to_dict(self) -> Optional[Dict[str, Any]]:
        """Get the full run data as a dictionary."""
        if self._data is None:
            self.load()
        return self._data.copy() if self._data else None
    
    def __repr__(self) -> str:
        return f"Run(id={self.run_id}, status={self.status})"


class RunManager:
    """
    Manages multiple runs and provides query/list operations.
    """
    
    def __init__(self, results_dir: str):
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
    
    def get(self, run_id: str) -> Run:
        """Get a Run object by ID."""
        return Run(run_id, self.results_dir)
    
    def exists(self, run_id: str) -> bool:
        """Check if a run exists."""
        return self.get(run_id).exists
    
    def create(self, run_id: str, data: Dict[str, Any]) -> Run:
        """Create a new run."""
        run = Run(run_id, self.results_dir)
        run.save(data)
        return run
    
    def list_all(self, api_key: Optional[str] = None, 
                 status: Optional[str] = None,
                 model: Optional[str] = None,
                 method: Optional[str] = None,
                 limit: int = 100,
                 offset: int = 0) -> List[Dict[str, Any]]:
        """
        List all runs with optional filtering.
        
        Args:
            api_key: Filter by API key
            status: Filter by status
            model: Filter by model
            method: Filter by method
            limit: Maximum number of results
            offset: Number of results to skip
        
        Returns:
            List of run summaries
        """
        all_runs = []
        
        if not os.path.exists(self.results_dir):
            return all_runs
        
        for run_id in os.listdir(self.results_dir):
            run_dir = os.path.join(self.results_dir, run_id)
            run_json = os.path.join(run_dir, "run.json")
            
            if not os.path.isdir(run_dir) or not os.path.exists(run_json):
                continue
            
            try:
                with open(run_json, 'r') as f:
                    run_data = json.load(f)
                
                # Apply filters
                if api_key and run_data.get("api_key") != api_key:
                    continue
                if status and run_data.get("status") != status:
                    continue
                if model and run_data.get("model") != model:
                    continue
                if method and run_data.get("method") != method:
                    continue
                
                all_runs.append({
                    "run_id": run_id,
                    "status": run_data.get("status"),
                    "dataset": run_data.get("dataset"),
                    "model": run_data.get("model"),
                    "method": run_data.get("method"),
                    "progress": run_data.get("progress", {}),
                    "created_at": run_data.get("created_at"),
                    "started_at": run_data.get("started_at"),
                    "completed_at": run_data.get("completed_at")
                })
            except (json.JSONDecodeError, IOError):
                continue
        
        # Sort by creation time (newest first)
        all_runs.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        
        return all_runs[offset:offset + limit]
    
    def delete(self, run_id: str) -> bool:
        """Delete a run and all its files."""
        import shutil
        
        run = self.get(run_id)
        if not run.exists:
            return False
        
        shutil.rmtree(run.run_dir)
        return True
    
    def count_by_api_key(self, api_key: str) -> Dict[str, int]:
        """Count runs by status for an API key."""
        counts = {
            "total": 0,
            "pending": 0,
            "running": 0,
            "evaluating": 0,
            "completed": 0,
            "failed": 0,
            "stopped": 0
        }
        
        for run_id in os.listdir(self.results_dir):
            run_json = os.path.join(self.results_dir, run_id, "run.json")
            
            if not os.path.exists(run_json):
                continue
            
            try:
                with open(run_json, 'r') as f:
                    run_data = json.load(f)
                
                if run_data.get("api_key") == api_key:
                    counts["total"] += 1
                    status = run_data.get("status", "unknown")
                    if status in counts:
                        counts[status] += 1
            except (json.JSONDecodeError, IOError):
                continue
        
        return counts
    
    def cleanup_stale_runs(self, timeout_hours: int = 24) -> List[str]:
        """
        Mark runs that have been stuck as failed.
        
        Args:
            timeout_hours: Consider runs stale after this many hours
        
        Returns:
            List of run IDs that were marked as failed
        """
        stale_runs = []
        timeout_seconds = timeout_hours * 3600
        current_time = time.time()
        
        for run_id in os.listdir(self.results_dir):
            run = self.get(run_id)
            if not run.exists:
                continue
            
            run_data = run.load()
            if not run_data:
                continue
            
            status = run_data.get("status")
            
            # Only check running or evaluating runs
            if status not in [RunStatus.RUNNING, RunStatus.EVALUATING]:
                continue
            
            # Check if it's been too long
            started_at = run_data.get("started_at")
            if not started_at:
                continue
            
            try:
                start_time = datetime.fromisoformat(started_at.replace('Z', '+00:00'))
                elapsed = current_time - start_time.timestamp()
                
                if elapsed > timeout_seconds:
                    run.update({
                        "status": RunStatus.FAILED,
                        "error": f"Run timed out after {timeout_hours} hours",
                        "failed_at": datetime.now().isoformat()
                    })
                    stale_runs.append(run_id)
            except (ValueError, TypeError):
                continue
        
        return stale_runs
