"""
Preprocessing CLI for Multi-Dataset Pipeline (PIE + JAAD + TITAN + MicroVision).
Usage:
  # 1. Ingest all 4 datasets into unified training corpus
  python preprocess_multidataset.py \
      --pie_dir data/raw/PIE \
      --jaad_dir data/raw/JAAD \
      --titan_dir data/raw/TITAN \
      --microvision_dir data/raw/MicroVision \
      --output_dir data/processed

  # 2. Evaluate MicroVision suitability
  python preprocess_multidataset.py --evaluate_microvision --microvision_dir data/raw/MicroVision
"""

import os
import argparse
from datasets.microvision_adapter import MicroVisionEvaluator
from datasets.multidataset_builder import MultiDatasetBuilder


def parse_args():
    parser = argparse.ArgumentParser(description="Multi-Dataset Preprocessor (PIE + JAAD + TITAN + MicroVision)")
    parser.add_argument("--pie_dir", type=str, default="data/raw/PIE", help="Path to raw PIE dataset")
    parser.add_argument("--jaad_dir", type=str, default="data/raw/JAAD", help="Path to raw JAAD dataset")
    parser.add_argument("--titan_dir", type=str, default="data/raw/TITAN", help="Path to raw TITAN dataset")
    parser.add_argument("--microvision_dir", type=str, default="data/raw/MicroVision", help="Path to MicroVision dataset")
    parser.add_argument("--output_dir", type=str, default="data/processed", help="Output directory for processed .npz sequences")
    parser.add_argument("--evaluate_microvision", action="store_true", help="Run MicroVision suitability assessment")
    parser.add_argument("--max_pie_samples", type=int, default=200, help="Max PIE sequences to generate")
    parser.add_argument("--max_jaad_samples", type=int, default=200, help="Max JAAD sequences to generate")
    parser.add_argument("--max_titan_samples", type=int, default=150, help="Max TITAN sequences to generate")
    parser.add_argument("--max_micro_samples", type=int, default=100, help="Max MicroVision sequences to generate")
    return parser.parse_args()


def main():
    args = parse_args()

    # 1. MicroVision Suitability Assessment
    if args.evaluate_microvision:
        print("\n" + "="*75)
        print("MICROVISION DATASET SUITABILITY EVALUATION")
        print("="*75)
        evaluator = MicroVisionEvaluator(args.microvision_dir)
        report = evaluator.evaluate_dataset_suitability()
        for k, v in report.items():
            if k == "recommendation":
                print(f"\n[Verdict]:\n  >>> {v}\n")
            else:
                print(f"  • {k:35s}: {v}")
        print("="*75 + "\n")

    # 2. Multi-Dataset Build across all 4 datasets
    builder = MultiDatasetBuilder(output_dir=args.output_dir)
    builder.build(
        pie_dir=args.pie_dir,
        jaad_dir=args.jaad_dir,
        titan_dir=args.titan_dir,
        microvision_dir=args.microvision_dir,
        max_pie_samples=args.max_pie_samples,
        max_jaad_samples=args.max_jaad_samples,
        max_titan_samples=args.max_titan_samples,
        max_micro_samples=args.max_micro_samples
    )


if __name__ == "__main__":
    main()
