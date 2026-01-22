#!/usr/bin/env python3
"""Download HuggingFace datasets for DCGen using the DatasetManager"""
import os
import sys

# Add the dcgen directory to path
sys.path.insert(0, os.path.expanduser("~/dcgen"))

from dataset_manager import DatasetManager

# Initialize dataset manager
dm = DatasetManager()

print("=== Downloading dcgen dataset ===")
result = dm.download_dataset("dcgen", force=True)
print(f"dcgen: {result}")

print("\n=== Downloading design2code dataset ===")
result = dm.download_dataset("design2code", force=True)
print(f"design2code: {result}")

print("\n=== Listing datasets ===")
for ds in dm.list_available_datasets():
    print(f"  {ds['name']}: downloaded={ds['downloaded']}, size={ds['size']}")
