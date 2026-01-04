#!/usr/bin/env python3
"""
Script to parse model information from benchmark logs and add model metadata columns.

This script takes the night_logs_2_5_combined.json file and adds three new columns:
- model_name: The base model name (e.g., "01-DeepSeek-R1-Distill-Qwen")
- model_size: The model size in millions of parameters (e.g., 1500)
- model_quant: The quantization method (e.g., "Q4_K_M")

Usage:
    uv run benchmarks/parse_model_info.py
"""

import json
import re
import sys
from pathlib import Path
from typing import Dict, Any, Tuple, Optional


def parse_model_name(model_filename: str) -> Tuple[Optional[str], Optional[int], Optional[str]]:
    """
    Parse model filename to extract model name, size, and quantization.

    Args:
        model_filename: The model filename (e.g., "01-DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf")

    Returns:
        Tuple of (model_name, model_size_mb, quantization)
    """
    # Remove .gguf extension if present
    model_name = model_filename.replace('.gguf', '')

    # Pattern to match model names with size and quantization
    # Examples:
    # 01-DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M
    # 02-Llama-3.2-3B-Instruct-Q8_0
    # 03-Mistral-7B-v0.1-Q4_K_M

    # Try to extract size (e.g., 1.5B, 7B, 3B)
    size_match = re.search(r'-(\d+(?:\.\d+)?)[BM]-', model_name)
    if not size_match:
        # Try alternative pattern without dash after size
        size_match = re.search(r'-(\d+(?:\.\d+)?)[BM](?:-|$)', model_name)

    model_size = None
    if size_match:
        size_str = size_match.group(1)
        size_float = float(size_str)
        # Convert to millions of parameters (B = billions, M = millions)
        if 'B' in model_name[size_match.start():size_match.end() + 1]:
            model_size = int(size_float * 1000)  # Convert billions to millions
        else:  # M for millions
            model_size = int(size_float)

    # Extract quantization using comprehensive patterns list
    quantization_patterns = [
        'Q4_K_M', 'Q4_K_S', 'Q4_0', 'Q4_1',
        'Q5_K_M', 'Q5_K_S', 'Q5_0', 'Q5_1',
        'Q6_K', 'Q8_0', 'F16', 'F32', 'BF16',
        'IQ4_NL', 'IQ4_XS'
    ]
    
    quantization = None
    for pattern in quantization_patterns:
        if model_name.endswith(f'-{pattern}'):
            quantization = pattern
            break

    # Extract base model name (everything before size and quantization)
    base_name = model_name
    if size_match:
        # Extract everything before the size pattern
        base_name = model_name[:size_match.start()].rstrip('-')
    elif quantization:
        # If no size but has quantization, extract everything before quantization
        quant_pos = model_name.rfind(f'-{quantization}')
        if quant_pos != -1:
            base_name = model_name[:quant_pos].rstrip('-')

    return base_name, model_size, quantization


def process_benchmark_data(input_file: Path, output_file: Path) -> None:
    """
    Process the benchmark JSON file and add model metadata columns.

    Args:
        input_file: Path to the input JSON file
        output_file: Path to the output JSON file
    """
    print(f"Reading data from {input_file}")

    with open(input_file, 'r') as f:
        data = json.load(f)

    print(f"Processing {len(data)} records...")

    # Process each record
    for record in data:
        model_filename = record.get('model', '')

        # Parse model information
        model_name, model_size, model_quant = parse_model_name(model_filename)

        # Add new columns
        record['model_name'] = model_name
        record['model_size'] = model_size
        record['model_quant'] = model_quant

    # Write processed data
    print(f"Writing processed data to {output_file}")

    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)

    # Print some statistics
    unique_models = set()
    unique_sizes = set()
    unique_quants = set()

    for record in data:
        if record.get('model_name'):
            unique_models.add(record['model_name'])
        if record.get('model_size'):
            unique_sizes.add(record['model_size'])
        if record.get('model_quant'):
            unique_quants.add(record['model_quant'])

    print(f"\nProcessing complete!")
    print(f"Found {len(unique_models)} unique model names: {sorted(unique_models)}")
    print(f"Found {len(unique_sizes)} unique model sizes: {sorted(unique_sizes)}")
    print(f"Found {len(unique_quants)} unique quantizations: {sorted(unique_quants)}")

    # Show a few examples
    print(f"\nFirst 3 examples:")
    for i, record in enumerate(data[:3]):
        print(f"  {i+1}. {record.get('model', 'N/A')} -> "
              f"name='{record.get('model_name', 'N/A')}', "
              f"size={record.get('model_size', 'N/A')}, "
              f"quant='{record.get('model_quant', 'N/A')}'")


def main():
    """Main function to run the script."""
    # Define file paths
    input_file = Path("benchmarks/parsed_logs/night_logs_6.json")
    output_file = Path("benchmarks/parsed_logs/night_logs_6_with_model_info.json")

    # Check if input file exists
    if not input_file.exists():
        print(f"Error: Input file {input_file} does not exist!")
        sys.exit(1)

    # Process the data
    try:
        process_benchmark_data(input_file, output_file)
        print(f"\nSuccess! Processed data saved to {output_file}")
    except Exception as e:
        print(f"Error processing data: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
