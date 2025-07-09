#!/usr/bin/env python3
"""
DEPRECATED: Legacy parallel extraction script
Use process_dataset.py with SLURM instead for better job management.

This script is kept for compatibility but the new SLURM-based approach
is recommended:
    sbatch submit_all_datasets.sbatch

See README_SLURM.md for detailed instructions.
"""

import sys
import os
from pathlib import Path
import warnings

# Add the python directory to the path
script_dir = Path(__file__).parent
python_dir = script_dir / 'python'
if str(python_dir) not in sys.path:
    sys.path.append(str(python_dir))

from nucleus_lineage_to_classification import nucleus_extractor
from lineage_tree import read_json_file


def main():
    """Legacy main function - use SLURM scripts instead."""
    
    warnings.warn(
        "This script is deprecated. Use 'sbatch submit_all_datasets.sbatch' instead.",
        DeprecationWarning,
        stacklevel=2
    )
    
    print("⚠️  WARNING: This script is deprecated!")
    print("🔄 Please use: sbatch submit_all_datasets.sbatch")
    print("📖 See README_SLURM.md for detailed instructions")
    print("")
    
    # Available datasets
    datasets = {
        "230212_stack6": "/mnt/ceph/users/lbrown/MouseData/Rebecca/230212_stack6/",
        "220321_stack11": "/mnt/ceph/users/lbrown/MouseData/Eszter1",
        "221016_FUCCI_Nanog_stack_3": "/mnt/ceph/users/lbrown/Labels3DMouse/Abhishek/RebeccaData/221016_FUCCI_Nanog_stack_3/",
    }
    
    output_dir = "/mnt/home/dchhantyal/3d-cnn-classification/data/nuclei_state_dataset/v2"
    
    print("🚀 Running legacy extraction for all datasets...")
    
    # Process each dataset
    for dataset_name, base_dir in datasets.items():
        print(f"\n📊 Processing dataset: {dataset_name}")
        print(f"📁 Base directory: {base_dir}")
        
        try:
            # Load lineage data
            lineage_file = Path(base_dir) / "LineageGraph.json"
            forest = read_json_file(lineage_file)
            
            # Extract nuclei
            results = nucleus_extractor(
                forest=forest,
                timeframe=1,
                base_dir=base_dir,
                output_dir=output_dir,
            )
            
            print(f"✅ Successfully processed {dataset_name}")
            
        except Exception as e:
            print(f"❌ Error processing {dataset_name}: {str(e)}")
            continue
    
    print(f"\n🎯 All datasets processed!")
    print(f"📂 Results saved to: {output_dir}")
    print("\n💡 For future runs, use the SLURM-based approach:")
    print("   sbatch submit_all_datasets.sbatch")


if __name__ == "__main__":
    main()