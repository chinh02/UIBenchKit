#!/usr/bin/env python3
"""
Run domain model with persistence helpers.
"""

import datetime
import json
import os

from config import RESULTS_DIR, calculate_cost
from services.fs_utils import ensure_dir


class Run:
    """Represents an experiment run."""

    RUN_METADATA_FILE = "run_metadata.json"
    EVALUATION_FILE = "evaluation.json"
    RESULTS_FILE = "results.json"
    COST_REPORT_FILE = "cost_report.json"

    def __init__(
        self,
        run_id: str,
        model: str,
        method: str,
        input_dir: str,
        api_key: str,
        dataset: str = None,
        sample_ids: list = None,
        user_api_key: str = None,
        user_base_url: str = None,
    ):
        self.run_id = run_id
        self.model = model
        self.method = method
        self.input_dir = input_dir
        self.output_dir = os.path.join(RESULTS_DIR, run_id)
        self.api_key = api_key
        self.dataset = dataset
        self.sample_ids = sample_ids
        self.user_api_key = user_api_key
        self.user_base_url = user_base_url
        self.status = "pending"
        self.created_at = datetime.datetime.now().isoformat()
        self.completed_at = None
        self.instances = {}
        self.total_instances = 0
        self.error = None
        self.token_usage = None
        self.cost_estimate = None
        self.evaluation = None

    def save_to_disk(self):
        """Save run metadata, evaluation, results, and cost report to disk."""
        ensure_dir(self.output_dir)

        if self.token_usage:
            self.cost_estimate = calculate_cost(self.model, self.token_usage)

        metadata = {
            "run_id": self.run_id,
            "model": self.model,
            "method": self.method,
            "dataset": self.dataset,
            "input_dir": self.input_dir,
            "output_dir": self.output_dir,
            "api_key": self.api_key,
            "sample_ids": self.sample_ids,
            "status": self.status,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
            "total_instances": self.total_instances,
            "error": self.error,
            "token_usage": self.token_usage,
            "cost_estimate": self.cost_estimate,
        }
        metadata_path = os.path.join(self.output_dir, self.RUN_METADATA_FILE)
        with open(metadata_path, "w", encoding="utf-8") as file:
            json.dump(metadata, file, indent=2, ensure_ascii=False)

        results = {
            "dataset": self.dataset,
            "model": self.model,
            "method": self.method,
            "run_id": self.run_id,
            "instances": self.instances,
        }
        results_path = os.path.join(self.output_dir, self.RESULTS_FILE)
        with open(results_path, "w", encoding="utf-8") as file:
            json.dump(results, file, indent=2, ensure_ascii=False)

        if self.evaluation:
            eval_path = os.path.join(self.output_dir, self.EVALUATION_FILE)
            with open(eval_path, "w", encoding="utf-8") as file:
                json.dump(self.evaluation, file, indent=2, ensure_ascii=False)

        if self.cost_estimate:
            cost_report = {
                "run_id": self.run_id,
                "model": self.model,
                "method": self.method,
                "dataset": self.dataset,
                "created_at": self.created_at,
                "completed_at": self.completed_at,
                "total_instances": self.total_instances,
                "token_usage": self.token_usage,
                "cost_estimate": self.cost_estimate,
                "summary": {
                    "total_tokens": self.cost_estimate.get("total_tokens", 0),
                    "total_cost_usd": self.cost_estimate.get("total_cost_usd", 0),
                    "cost_per_instance_usd": round(
                        self.cost_estimate.get("total_cost_usd", 0) / max(self.total_instances, 1),
                        6,
                    ),
                },
            }
            cost_path = os.path.join(self.output_dir, self.COST_REPORT_FILE)
            with open(cost_path, "w", encoding="utf-8") as file:
                json.dump(cost_report, file, indent=2, ensure_ascii=False)

    @classmethod
    def load_from_disk(cls, run_dir: str) -> "Run":
        """Load a run from disk."""
        metadata_path = os.path.join(run_dir, cls.RUN_METADATA_FILE)
        if not os.path.exists(metadata_path):
            return None

        try:
            with open(metadata_path, "r", encoding="utf-8") as file:
                metadata = json.load(file)

            run = cls(
                run_id=metadata["run_id"],
                model=metadata["model"],
                method=metadata["method"],
                input_dir=metadata.get("input_dir", ""),
                api_key=metadata.get("api_key", ""),
                dataset=metadata.get("dataset"),
                sample_ids=metadata.get("sample_ids"),
            )

            run.status = metadata.get("status", "completed")
            run.created_at = metadata.get("created_at")
            run.completed_at = metadata.get("completed_at")
            run.total_instances = metadata.get("total_instances", 0)
            run.error = metadata.get("error")
            run.token_usage = metadata.get("token_usage")
            run.cost_estimate = metadata.get("cost_estimate")
            run.output_dir = run_dir

            results_path = os.path.join(run_dir, cls.RESULTS_FILE)
            if os.path.exists(results_path):
                with open(results_path, "r", encoding="utf-8") as file:
                    results = json.load(file)
                run.instances = results.get("instances", {})

            eval_path = os.path.join(run_dir, cls.EVALUATION_FILE)
            if os.path.exists(eval_path):
                with open(eval_path, "r", encoding="utf-8") as file:
                    run.evaluation = json.load(file)

            if not run.cost_estimate:
                cost_path = os.path.join(run_dir, cls.COST_REPORT_FILE)
                if os.path.exists(cost_path):
                    with open(cost_path, "r", encoding="utf-8") as file:
                        cost_report = json.load(file)
                    run.cost_estimate = cost_report.get("cost_estimate")

            return run
        except Exception as error:
            print(f"Error loading run from {run_dir}: {error}")
            return None

    def to_dict(self, include_details: bool = False):
        result = {
            "run_id": self.run_id,
            "model": self.model,
            "method": self.method,
            "dataset": self.dataset,
            "status": self.status,
            "total_instances": self.total_instances,
            "completed_instances": len([item for item in self.instances.values() if item.get("status") == "completed"]),
            "pending_instances": len([item for item in self.instances.values() if item.get("status") == "pending"]),
            "running_instances": len([item for item in self.instances.values() if item.get("status") == "running"]),
            "failed_instances": len([item for item in self.instances.values() if item.get("status") == "failed"]),
            "created_at": self.created_at,
            "completed_at": self.completed_at,
        }
        if include_details:
            result["input_dir"] = self.input_dir
            result["output_dir"] = self.output_dir
            result["instances"] = self.instances
            result["token_usage"] = self.token_usage
            result["cost_estimate"] = self.cost_estimate
            result["evaluation"] = self.evaluation
            result["error"] = self.error
            result["sample_ids"] = self.sample_ids
        return result

    def get_poll_status(self):
        """Get status payload for polling endpoints."""
        running = [key for key, val in self.instances.items() if val.get("status") == "running"]
        completed = [key for key, val in self.instances.items() if val.get("status") == "completed"]
        pending = [key for key, val in self.instances.items() if val.get("status") == "pending"]
        failed = [key for key, val in self.instances.items() if val.get("status") == "failed"]
        failed_details = {
            key: val.get("error", "Unknown error")
            for key, val in self.instances.items()
            if val.get("status") == "failed"
        }
        result = {
            "run_id": self.run_id,
            "status": self.status,
            "dataset": self.dataset,
            "model": self.model,
            "method": self.method,
            "running": running,
            "completed": completed,
            "pending": pending,
            "failed": failed,
            "failed_details": failed_details,
        }
        if self.status == "completed":
            if self.evaluation:
                result["evaluation"] = self.evaluation
            if self.cost_estimate:
                result["cost_estimate"] = self.cost_estimate
        return result
