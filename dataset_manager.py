#!/usr/bin/env python3
"""
Dataset Manager for DCGen
=========================

Manages downloading and accessing datasets from Hugging Face for benchmarking.

Supported Datasets:
- design2code: SALT-NLP/Design2Code-hf (484 webpages from C4 validation set)
- dcgen: iforgott/DCGen (461 experiment images)

Usage:
    from dataset_manager import DatasetManager
    
    dm = DatasetManager()
    dm.download_dataset("design2code")
    dm.download_dataset("dcgen")
    
    # Get dataset info
    info = dm.get_dataset_info("design2code")
    
    # Get samples
    samples = dm.get_samples("design2code", limit=10)
"""

import os
import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from PIL import Image
import io


# Dataset configurations
DATASETS_CONFIG = {
    "design2code": {
        "hf_repo": "SALT-NLP/Design2Code-hf",
        "description": "484 webpages from C4 validation set for Design2Code benchmark",
        "image_column": "image",
        "html_column": "text",
        "split": "train",
        "size": 484,
        "placeholder_image": "rick.jpg"
    },
    "dcgen": {
        "hf_repo": "iforgott/DCGen",
        "description": "461 experiment images with HTML for DCGen benchmarking",
        "image_column": "image",
        "html_column": "html",  # Use full_data folder which has HTML files
        "split": "train",
        "size": 461,
        "placeholder_image": "placeholder.png",
        "use_raw_files": True,  # Download from full_data folder directly
        "raw_folder": "full_data"  # Folder containing paired .png and .html files
    }
}


class DatasetManager:
    """Manages downloading and accessing HuggingFace datasets."""
    
    def __init__(self, cache_dir: str = None):
        """
        Initialize the DatasetManager.
        
        Args:
            cache_dir: Directory to cache downloaded datasets. 
                       Defaults to ./data/hf_datasets
        """
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.cache_dir = cache_dir or os.path.join(self.base_dir, "data", "hf_datasets")
        self.datasets_info = {}
        
        # Ensure cache directory exists
        os.makedirs(self.cache_dir, exist_ok=True)
    
    def list_available_datasets(self) -> List[Dict[str, Any]]:
        """
        List all available datasets and their status.
        
        Returns:
            List of dataset info dictionaries
        """
        result = []
        for name, config in DATASETS_CONFIG.items():
            dataset_dir = os.path.join(self.cache_dir, name)
            is_downloaded = os.path.exists(dataset_dir) and os.path.exists(
                os.path.join(dataset_dir, "metadata.json")
            )
            
            result.append({
                "name": name,
                "hf_repo": config["hf_repo"],
                "description": config["description"],
                "size": config["size"],
                "downloaded": is_downloaded,
                "local_path": dataset_dir if is_downloaded else None
            })
        
        return result
    
    def download_dataset(self, dataset_name: str, force: bool = False) -> Dict[str, Any]:
        """
        Download a dataset from HuggingFace.
        
        Args:
            dataset_name: Name of the dataset (design2code, dcgen)
            force: Force re-download even if already exists
        
        Returns:
            Dictionary with download status and info
        """
        if dataset_name not in DATASETS_CONFIG:
            raise ValueError(
                f"Unknown dataset: {dataset_name}. "
                f"Available: {', '.join(DATASETS_CONFIG.keys())}"
            )
        
        config = DATASETS_CONFIG[dataset_name]
        dataset_dir = os.path.join(self.cache_dir, dataset_name)
        metadata_file = os.path.join(dataset_dir, "metadata.json")
        
        # Check if already downloaded
        if os.path.exists(metadata_file) and not force:
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            return {
                "status": "already_downloaded",
                "message": f"Dataset {dataset_name} already downloaded",
                "path": dataset_dir,
                "samples": metadata.get("num_samples", 0)
            }
        
        # Clean up existing directory if force
        if force and os.path.exists(dataset_dir):
            shutil.rmtree(dataset_dir)
        
        os.makedirs(dataset_dir, exist_ok=True)
        
        try:
            # Check if we should use raw file download (for repos with full_data folder)
            if config.get("use_raw_files"):
                return self._download_raw_files(dataset_name, config, dataset_dir, metadata_file)
            
            from datasets import load_dataset
            
            # Load dataset from HuggingFace
            print(f"Downloading {dataset_name} from {config['hf_repo']}...")
            hf_dataset = load_dataset(config["hf_repo"], split=config["split"])
            
            # Process and save samples
            num_samples = 0
            samples_info = []
            
            for idx, sample in enumerate(hf_dataset):
                sample_id = str(idx)
                sample_dir = os.path.join(dataset_dir, "samples")
                os.makedirs(sample_dir, exist_ok=True)
                
                # Save image
                image_path = os.path.join(sample_dir, f"{sample_id}.png")
                if config["image_column"] and config["image_column"] in sample:
                    img = sample[config["image_column"]]
                    if isinstance(img, Image.Image):
                        img.save(image_path)
                    elif isinstance(img, dict) and "bytes" in img:
                        img = Image.open(io.BytesIO(img["bytes"]))
                        img.save(image_path)
                
                # Save HTML if available
                html_path = None
                if config["html_column"] and config["html_column"] in sample:
                    html_path = os.path.join(sample_dir, f"{sample_id}.html")
                    html_content = sample[config["html_column"]]
                    with open(html_path, 'w', encoding='utf-8') as f:
                        f.write(html_content)
                
                samples_info.append({
                    "id": sample_id,
                    "image": image_path,
                    "html": html_path
                })
                num_samples += 1
            
            # Create placeholder image
            placeholder_path = os.path.join(sample_dir, "placeholder.png")
            if not os.path.exists(placeholder_path):
                # Create a simple gray placeholder
                placeholder = Image.new('RGB', (100, 100), color=(200, 200, 200))
                placeholder.save(placeholder_path)
            
            # Save metadata
            metadata = {
                "name": dataset_name,
                "hf_repo": config["hf_repo"],
                "description": config["description"],
                "num_samples": num_samples,
                "samples": samples_info,
                "samples_dir": os.path.join(dataset_dir, "samples")
            }
            
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            return {
                "status": "success",
                "message": f"Downloaded {num_samples} samples from {dataset_name}",
                "path": dataset_dir,
                "samples": num_samples
            }
            
        except ImportError:
            return {
                "status": "error",
                "message": "Please install the 'datasets' package: pip install datasets"
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Failed to download dataset: {str(e)}"
            }
    
    def _download_raw_files(self, dataset_name: str, config: Dict, 
                            dataset_dir: str, metadata_file: str) -> Dict[str, Any]:
        """
        Download raw files (PNG + HTML) from a HuggingFace repo folder.
        
        This is used for datasets like dcgen that have a full_data folder
        with paired .png and .html files rather than using the datasets library.
        
        Args:
            dataset_name: Name of the dataset
            config: Dataset configuration dict
            dataset_dir: Local directory to save files
            metadata_file: Path to metadata.json
        
        Returns:
            Status dictionary
        """
        try:
            from huggingface_hub import HfApi, hf_hub_download
            
            hf_repo = config["hf_repo"]
            raw_folder = config.get("raw_folder", "full_data")
            
            print(f"Downloading {dataset_name} raw files from {hf_repo}/{raw_folder}...")
            
            # List files in the full_data folder
            api = HfApi()
            files = api.list_repo_files(repo_id=hf_repo, repo_type="dataset")
            
            # Filter for files in the raw folder
            raw_files = [f for f in files if f.startswith(f"{raw_folder}/")]
            
            # Get unique sample IDs (filenames without extension)
            sample_ids = set()
            for f in raw_files:
                basename = os.path.basename(f)
                name_without_ext = os.path.splitext(basename)[0]
                sample_ids.add(name_without_ext)
            
            # Sort numerically if all IDs are digits, otherwise alphabetically
            def sort_key(x):
                try:
                    return (0, int(x))
                except ValueError:
                    return (1, x)
            sample_ids = sorted(sample_ids, key=sort_key)
            
            # Create samples directory
            sample_dir = os.path.join(dataset_dir, "samples")
            os.makedirs(sample_dir, exist_ok=True)
            
            # Download each sample
            num_samples = 0
            samples_info = []
            
            for sample_id in sample_ids:
                png_file = f"{raw_folder}/{sample_id}.png"
                html_file = f"{raw_folder}/{sample_id}.html"
                
                image_path = None
                html_path = None
                
                # Download PNG
                if png_file in raw_files:
                    try:
                        local_png = hf_hub_download(
                            repo_id=hf_repo,
                            filename=png_file,
                            repo_type="dataset",
                            local_dir=dataset_dir,
                            local_dir_use_symlinks=False
                        )
                        # Move/copy to samples directory with standardized name
                        dest_png = os.path.join(sample_dir, f"{sample_id}.png")
                        if local_png != dest_png:
                            shutil.copy2(local_png, dest_png)
                        image_path = dest_png
                    except Exception as e:
                        print(f"Warning: Failed to download {png_file}: {e}")
                
                # Download HTML
                if html_file in raw_files:
                    try:
                        local_html = hf_hub_download(
                            repo_id=hf_repo,
                            filename=html_file,
                            repo_type="dataset",
                            local_dir=dataset_dir,
                            local_dir_use_symlinks=False
                        )
                        # Move/copy to samples directory with standardized name
                        dest_html = os.path.join(sample_dir, f"{sample_id}.html")
                        if local_html != dest_html:
                            shutil.copy2(local_html, dest_html)
                        html_path = dest_html
                    except Exception as e:
                        print(f"Warning: Failed to download {html_file}: {e}")
                
                if image_path:  # Only add if we at least have the image
                    samples_info.append({
                        "id": sample_id,
                        "image": image_path,
                        "html": html_path
                    })
                    num_samples += 1
                    
                    if num_samples % 50 == 0:
                        print(f"  Downloaded {num_samples} samples...")
            
            # Create placeholder image
            placeholder_path = os.path.join(sample_dir, "placeholder.png")
            if not os.path.exists(placeholder_path):
                placeholder = Image.new('RGB', (100, 100), color=(200, 200, 200))
                placeholder.save(placeholder_path)
            
            # Save metadata
            metadata = {
                "name": dataset_name,
                "hf_repo": config["hf_repo"],
                "description": config["description"],
                "num_samples": num_samples,
                "samples": samples_info,
                "samples_dir": sample_dir,
                "source_folder": raw_folder
            }
            
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            print(f"Successfully downloaded {num_samples} samples with HTML")
            
            return {
                "status": "success",
                "message": f"Downloaded {num_samples} samples from {dataset_name}/{raw_folder}",
                "path": dataset_dir,
                "samples": num_samples
            }
            
        except ImportError:
            return {
                "status": "error",
                "message": "Please install huggingface_hub: pip install huggingface_hub"
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Failed to download raw files: {str(e)}"
            }
    
    def get_dataset_info(self, dataset_name: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a downloaded dataset.
        
        Args:
            dataset_name: Name of the dataset
        
        Returns:
            Dataset metadata or None if not downloaded
        """
        if dataset_name not in DATASETS_CONFIG:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        metadata_file = os.path.join(self.cache_dir, dataset_name, "metadata.json")
        
        if not os.path.exists(metadata_file):
            return None
        
        with open(metadata_file, 'r') as f:
            return json.load(f)
    
    def get_samples_dir(self, dataset_name: str) -> Optional[str]:
        """
        Get the samples directory for a dataset.
        
        Args:
            dataset_name: Name of the dataset
        
        Returns:
            Path to samples directory or None if not downloaded
        """
        info = self.get_dataset_info(dataset_name)
        if info:
            return info.get("samples_dir")
        return None
    
    def get_samples(self, dataset_name: str, limit: int = None, 
                   offset: int = 0) -> List[Dict[str, Any]]:
        """
        Get samples from a downloaded dataset.
        
        Args:
            dataset_name: Name of the dataset
            limit: Maximum number of samples to return
            offset: Starting offset
        
        Returns:
            List of sample dictionaries with id, image_path, html_path
        """
        info = self.get_dataset_info(dataset_name)
        if not info:
            raise ValueError(f"Dataset {dataset_name} not downloaded")
        
        samples = info.get("samples", [])
        
        if offset:
            samples = samples[offset:]
        if limit:
            samples = samples[:limit]
        
        return samples
    
    def get_sample_ids(self, dataset_name: str) -> List[str]:
        """
        Get all sample IDs from a dataset.
        
        Args:
            dataset_name: Name of the dataset
        
        Returns:
            List of sample IDs
        """
        info = self.get_dataset_info(dataset_name)
        if not info:
            return []
        
        return [s["id"] for s in info.get("samples", [])]
    
    def delete_dataset(self, dataset_name: str) -> Dict[str, Any]:
        """
        Delete a downloaded dataset.
        
        Args:
            dataset_name: Name of the dataset
        
        Returns:
            Status dictionary
        """
        if dataset_name not in DATASETS_CONFIG:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        dataset_dir = os.path.join(self.cache_dir, dataset_name)
        
        if os.path.exists(dataset_dir):
            shutil.rmtree(dataset_dir)
            return {
                "status": "success",
                "message": f"Dataset {dataset_name} deleted"
            }
        
        return {
            "status": "not_found",
            "message": f"Dataset {dataset_name} not found"
        }
    
    def prepare_benchmark_dir(self, dataset_name: str, 
                              sample_ids: List[str] = None) -> str:
        """
        Prepare a benchmark directory from a dataset.
        Creates a directory structure compatible with the DCGen experiment runner.
        
        Args:
            dataset_name: Name of the dataset
            sample_ids: Optional list of specific sample IDs to include.
                        If provided, creates a unique subdirectory with only those samples.
        
        Returns:
            Path to the benchmark directory
        """
        info = self.get_dataset_info(dataset_name)
        if not info:
            raise ValueError(f"Dataset {dataset_name} not downloaded")
        
        samples_dir = info.get("samples_dir")
        samples = info.get("samples", [])
        
        # If sample_ids is specified, create a unique subdirectory for this subset
        if sample_ids:
            # Create a unique benchmark directory for this subset
            subset_name = "_".join(sorted(sample_ids)[:5])  # Use first 5 IDs for naming
            if len(sample_ids) > 5:
                subset_name += f"_plus{len(sample_ids)-5}"
            benchmark_dir = os.path.join(self.cache_dir, dataset_name, f"benchmark_{subset_name}")
            
            # Clean the directory to ensure only requested samples are included
            if os.path.exists(benchmark_dir):
                shutil.rmtree(benchmark_dir)
            os.makedirs(benchmark_dir)
            
            samples = [s for s in samples if s["id"] in sample_ids]
        else:
            # Use default benchmark directory for full dataset
            benchmark_dir = os.path.join(self.cache_dir, dataset_name, "benchmark")
            os.makedirs(benchmark_dir, exist_ok=True)
        
        # Copy/link files to benchmark directory
        for sample in samples:
            sample_id = sample["id"]
            
            # Copy image
            src_img = os.path.join(samples_dir, f"{sample_id}.png")
            dst_img = os.path.join(benchmark_dir, f"{sample_id}.png")
            if os.path.exists(src_img):
                shutil.copy(src_img, dst_img)
            
            # Copy HTML reference if exists
            src_html = os.path.join(samples_dir, f"{sample_id}.html")
            dst_html = os.path.join(benchmark_dir, f"{sample_id}.html")
            if os.path.exists(src_html):
                shutil.copy(src_html, dst_html)
        
        # Copy placeholder
        placeholder_src = os.path.join(samples_dir, "placeholder.png")
        placeholder_dst = os.path.join(benchmark_dir, "placeholder.png")
        if os.path.exists(placeholder_src):
            shutil.copy(placeholder_src, placeholder_dst)
        
        return benchmark_dir


# Singleton instance for module-level access
_manager_instance = None


def get_dataset_manager() -> DatasetManager:
    """Get the global DatasetManager instance."""
    global _manager_instance
    if _manager_instance is None:
        _manager_instance = DatasetManager()
    return _manager_instance


if __name__ == "__main__":
    # CLI interface for dataset management
    import argparse
    
    parser = argparse.ArgumentParser(description="DCGen Dataset Manager")
    subparsers = parser.add_subparsers(dest="command")
    
    # List command
    list_parser = subparsers.add_parser("list", help="List available datasets")
    
    # Download command
    download_parser = subparsers.add_parser("download", help="Download a dataset")
    download_parser.add_argument("dataset", help="Dataset name (design2code, dcgen)")
    download_parser.add_argument("--force", action="store_true", help="Force re-download")
    
    # Info command
    info_parser = subparsers.add_parser("info", help="Get dataset info")
    info_parser.add_argument("dataset", help="Dataset name")
    
    # Delete command
    delete_parser = subparsers.add_parser("delete", help="Delete a dataset")
    delete_parser.add_argument("dataset", help="Dataset name")
    
    args = parser.parse_args()
    
    dm = DatasetManager()
    
    if args.command == "list":
        datasets = dm.list_available_datasets()
        print("\nAvailable Datasets:")
        print("-" * 60)
        for ds in datasets:
            status = "✓ Downloaded" if ds["downloaded"] else "✗ Not downloaded"
            print(f"  {ds['name']}: {ds['description']}")
            print(f"    HuggingFace: {ds['hf_repo']}")
            print(f"    Samples: {ds['size']}")
            print(f"    Status: {status}")
            print()
    
    elif args.command == "download":
        result = dm.download_dataset(args.dataset, force=args.force)
        print(f"\n{result['status']}: {result['message']}")
        if result.get("path"):
            print(f"Path: {result['path']}")
    
    elif args.command == "info":
        info = dm.get_dataset_info(args.dataset)
        if info:
            print(f"\nDataset: {info['name']}")
            print(f"Description: {info['description']}")
            print(f"Samples: {info['num_samples']}")
            print(f"Path: {info['samples_dir']}")
        else:
            print(f"Dataset {args.dataset} not downloaded")
    
    elif args.command == "delete":
        result = dm.delete_dataset(args.dataset)
        print(f"\n{result['status']}: {result['message']}")
    
    else:
        parser.print_help()
