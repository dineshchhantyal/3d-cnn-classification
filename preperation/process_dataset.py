#!/usr/bin/env python3
"""
Command-line script to process a single dataset for nucleus extraction.
This script is designed to be used with SLURM for parallel processing of multiple datasets.

Usage:
    python process_dataset.py --dataset DATASET_NAME --timeframe TIMEFRAME --output_dir OUTPUT_DIR [--max_samples MAX_SAMPLES]

Example:
    python process_dataset.py --dataset 230212_stack6 --timeframe 1 --output_dir /path/to/output
"""

import sys
import os
import argparse
from pathlib import Path

# Add the python directory to the path
script_dir = Path(__file__).parent
python_dir = script_dir / 'python'
if str(python_dir) not in sys.path:
    sys.path.append(str(python_dir))

from nucleus_lineage_to_classification import nucleus_extractor
from lineage_tree import read_json_file


# Available datasets configuration
DATASETS = {
    "230212_stack6": "/mnt/ceph/users/lbrown/MouseData/Rebecca/230212_stack6/",
    "220321_stack11": "/mnt/ceph/users/lbrown/MouseData/Eszter1",
    "221016_FUCCI_Nanog_stack_3": "/mnt/ceph/users/lbrown/Labels3DMouse/Abhishek/RebeccaData/221016_FUCCI_Nanog_stack_3/",
    "David4EPI": "/mnt/ceph/users/lbrown/MouseData/David4EPI",
    "230101_Gata6Nanog_stack_19": "/mnt/ceph/users/lbrown/Labels3DMouse/Abhishek/RebeccaData/230101_Gata6Nanog_stack_19/",
}


def main():
    """Main function to process a single dataset."""
    parser = argparse.ArgumentParser(
        description="Process a single dataset for nucleus extraction",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--dataset",
        required=True,
        choices=list(DATASETS.keys()),
        help="Dataset name to process"
    )
    
    parser.add_argument(
        "--timeframe",
        type=int,
        default=1,
        help="Timeframe for extraction (±N frames around event)"
    )
    
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Output directory for extracted nuclei"
    )
    
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples per classification (None for unlimited)"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    args = parser.parse_args()
    
    # Get dataset configuration
    dataset_name = args.dataset
    base_dir = DATASETS[dataset_name]
    
    if args.verbose:
        print(f"🚀 Processing dataset: {dataset_name}")
        print(f"📁 Base directory: {base_dir}")
        print(f"📂 Output directory: {args.output_dir}")
        print(f"⏱️  Timeframe: ±{args.timeframe}")
        print(f"📊 Max samples: {args.max_samples if args.max_samples else 'unlimited'}")
        print("=" * 80)
    
    # Verify base directory exists
    if not Path(base_dir).exists():
        print(f"❌ Error: Base directory does not exist: {base_dir}")
        sys.exit(1)
    
    # Verify LineageGraph.json exists
    lineage_file = Path(base_dir) / "LineageGraph.json"
    if not lineage_file.exists():
        print(f"❌ Error: LineageGraph.json not found: {lineage_file}")
        sys.exit(1)
    
    # Create output directory if it doesn't exist
    output_path = Path(args.output_dir) if args.output_dir else Path("/mnt/home/dchhantyal/ceph/MouseNuceli/") / datetime.now().strftime("%Y%m%d_%H%M%S") 
    output_path.mkdir(parents=True, exist_ok=True)
    
    try:
        # Load the lineage forest
        print(f"📖 Loading lineage data from: {lineage_file}")
        forest = read_json_file(lineage_file)
        
        # Extract nuclei
        print(f"🔬 Starting nucleus extraction for {dataset_name}...")
        results = nucleus_extractor(
            forest=forest,
            timeframe=args.timeframe,
            base_dir=base_dir,
            output_dir=args.output_dir,
            max_samples=args.max_samples
        )
        
        print(f"\n✅ Successfully completed processing for {dataset_name}")
        print("🎯 EXTRACTION RESULTS SUMMARY:")
        
        # Print results summary if available
        if isinstance(results, dict):
            for classification, count in results.items():
                if isinstance(count, (int, float)):
                    print(f"   📊 {classification}: {count} samples")
        
        print(f"📂 Results saved to: {args.output_dir}")
        
    except Exception as e:
        print(f"❌ Error processing dataset {dataset_name}: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
