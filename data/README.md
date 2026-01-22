# Data Directory

This directory should contain your datasets and experimental results. The directory structure should be:

```
data/
├── demo/           # Sample demo files
├── dcgen_demo/     # DCGen generated outputs  
├── direct_demo/    # Direct method outputs
├── original/       # Original reference data
└── hf_datasets/    # HuggingFace datasets
```

**Note**: The actual data files are not included in the repository due to size constraints. 
You can download the full dataset from [HuggingFace](https://huggingface.co/datasets/iforgott/DCGen).

For running experiments, place your test images and reference files in the appropriate subdirectories.