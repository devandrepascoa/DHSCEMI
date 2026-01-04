#!/usr/bin/env python3
"""
Throughput per Dollar Analysis
Analyzes benchmark data to calculate throughput per dollar for different device configurations.
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import numpy as np

def load_benchmark_data(file_path: str) -> List[Dict]:
    """Load benchmark data from JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)

def define_cost_assumptions() -> Dict[str, float]:
    """
    Define cost per hour assumptions for different device types.
    These are approximate costs based on cloud provider pricing.
    """
    return {
        # GPU costs ($/hour) - based on AWS/GCP GPU instance pricing
        'cuda_100': 2.50,  # Full GPU utilization (e.g., V100/A100 equivalent)
        'cuda_75': 1.88,   # 75% GPU utilization
        'cuda_50': 1.25,   # 50% GPU utilization
        'cuda_25': 0.63,   # 25% GPU utilization

        # CPU costs ($/hour) - based on cloud CPU instance pricing
        'cpu_8': 0.40,     # 8-core CPU instance
        'cpu_4': 0.20,     # 4-core CPU instance
        'cpu_2': 0.10,     # 2-core CPU instance
        'cpu_1': 0.05,     # 1-core CPU instance
    }

def categorize_device(variant: str, cpu_cores: int, gpu_percentage: int) -> str:
    """Categorize device configuration for cost calculation."""
    if variant == 'cuda':
        return f'cuda_{gpu_percentage}'
    else:
        return f'cpu_{cpu_cores}'

def calculate_throughput_per_dollar(data: List[Dict], cost_assumptions: Dict[str, float]) -> pd.DataFrame:
    """Calculate throughput per dollar for each device configuration."""

    results = []

    for entry in data:
        # Extract key metrics
        variant = entry['variant']
        model = entry['model']
        cpu_cores = entry.get('cpu_cores')
        gpu_percentage = entry.get('gpu_percentage')

        # Get throughput metrics
        throughput_mean = entry['throughput_mean']
        token_gen_throughput = entry['token_generation_throughput_mean']
        prompt_processing_throughput = entry['prompt_processing_throughput_mean']

        # Categorize device for cost lookup
        device_category = categorize_device(variant, cpu_cores, gpu_percentage)
        cost_per_hour = cost_assumptions.get(device_category, 0)

        # Calculate throughput per dollar per hour
        if cost_per_hour > 0:
            throughput_per_dollar = throughput_mean / cost_per_hour
            token_gen_per_dollar = token_gen_throughput / cost_per_hour
            prompt_proc_per_dollar = prompt_processing_throughput / cost_per_hour
        else:
            throughput_per_dollar = 0
            token_gen_per_dollar = 0
            prompt_proc_per_dollar = 0

        results.append({
            'variant': variant,
            'model': model.split('/')[-1] if '/' in model else model,  # Clean model name
            'cpu_cores': cpu_cores,
            'gpu_percentage': gpu_percentage,
            'device_category': device_category,
            'cost_per_hour': cost_per_hour,
            'throughput_mean': throughput_mean,
            'token_generation_throughput': token_gen_throughput,
            'prompt_processing_throughput': prompt_processing_throughput,
            'throughput_per_dollar': throughput_per_dollar,
            'token_gen_per_dollar': token_gen_per_dollar,
            'prompt_proc_per_dollar': prompt_proc_per_dollar,
            'concurrent_requests': entry['concurrent_requests'],
            'token_size': entry['token_size']
        })

    return pd.DataFrame(results)

def create_visualizations(df: pd.DataFrame):
    """Create visualizations for throughput per dollar analysis."""

    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Throughput per Dollar Analysis', fontsize=16, fontweight='bold')

    # 1. Overall Throughput per Dollar by Device Category
    ax1 = axes[0, 0]
    device_summary = df.groupby('device_category')['throughput_per_dollar'].mean().sort_values(ascending=False)
    device_summary.plot(kind='bar', ax=ax1, color='skyblue')
    ax1.set_title('Average Throughput per Dollar by Device')
    ax1.set_xlabel('Device Configuration')
    ax1.set_ylabel('Tokens/sec per $')
    ax1.tick_params(axis='x', rotation=45)

    # 2. Token Generation Throughput per Dollar
    ax2 = axes[0, 1]
    token_gen_summary = df.groupby('device_category')['token_gen_per_dollar'].mean().sort_values(ascending=False)
    token_gen_summary.plot(kind='bar', ax=ax2, color='lightcoral')
    ax2.set_title('Token Generation Throughput per Dollar')
    ax2.set_xlabel('Device Configuration')
    ax2.set_ylabel('Tokens/sec per $')
    ax2.tick_params(axis='x', rotation=45)

    # 3. Cost vs Performance Scatter Plot
    ax3 = axes[1, 0]
    scatter = ax3.scatter(df['cost_per_hour'], df['throughput_mean'],
                         c=df['variant'].map({'cuda': 'red', 'cpu': 'blue'}),
                         alpha=0.7, s=60)
    ax3.set_xlabel('Cost per Hour ($)')
    ax3.set_ylabel('Throughput (tokens/sec)')
    ax3.set_title('Cost vs Performance')

    # Add legend for scatter plot
    cuda_points = ax3.scatter([], [], c='red', alpha=0.7, s=60, label='CUDA')
    cpu_points = ax3.scatter([], [], c='blue', alpha=0.7, s=60, label='CPU')
    ax3.legend()

    # 4. Model Comparison
    ax4 = axes[1, 1]
    model_comparison = df.groupby(['model', 'variant'])['throughput_per_dollar'].mean().unstack()
    model_comparison.plot(kind='bar', ax=ax4, width=0.8)
    ax4.set_title('Throughput per Dollar by Model and Variant')
    ax4.set_xlabel('Model')
    ax4.set_ylabel('Tokens/sec per $')
    ax4.tick_params(axis='x', rotation=45)
    ax4.legend(title='Variant')

    plt.tight_layout()
    plt.show()

    return fig

def print_summary_table(df: pd.DataFrame):
    """Print a summary table of the results."""

    print("\n" + "="*80)
    print("THROUGHPUT PER DOLLAR ANALYSIS SUMMARY")
    print("="*80)

    # Cost assumptions
    print("\nCost Assumptions ($/hour):")
    cost_assumptions = define_cost_assumptions()
    for device, cost in sorted(cost_assumptions.items()):
        print(f"  {device}: ${cost:.2f}")

    # Top performers overall
    print(f"\nTop 10 Configurations by Overall Throughput per Dollar:")
    print("-" * 60)
    top_configs = df.nlargest(10, 'throughput_per_dollar')[
        ['device_category', 'model', 'throughput_per_dollar', 'throughput_mean', 'cost_per_hour']
    ]

    for idx, row in top_configs.iterrows():
        print(f"{row['device_category']:12} | {row['model'][:25]:25} | "
              f"{row['throughput_per_dollar']:8.1f} tok/sec/$ | "
              f"{row['throughput_mean']:8.1f} tok/sec | ${row['cost_per_hour']:5.2f}/hr")

    # Summary by device category
    print(f"\nSummary by Device Category:")
    print("-" * 60)
    summary = df.groupby('device_category').agg({
        'throughput_per_dollar': ['mean', 'std', 'count'],
        'throughput_mean': 'mean',
        'cost_per_hour': 'first'
    }).round(2)

    summary.columns = ['Avg_TPD', 'Std_TPD', 'Count', 'Avg_Throughput', 'Cost_Per_Hour']
    summary = summary.sort_values('Avg_TPD', ascending=False)

    print(summary.to_string())

    # Best value propositions
    print(f"\nBest Value Propositions:")
    print("-" * 40)

    cuda_best = df[df['variant'] == 'cuda'].nlargest(1, 'throughput_per_dollar').iloc[0]
    cpu_best = df[df['variant'] == 'cpu'].nlargest(1, 'throughput_per_dollar').iloc[0]

    print(f"Best CUDA: {cuda_best['device_category']} - {cuda_best['throughput_per_dollar']:.1f} tokens/sec/$")
    print(f"Best CPU:  {cpu_best['device_category']} - {cpu_best['throughput_per_dollar']:.1f} tokens/sec/$")

    print("\n" + "="*80)

def main():
    """Main analysis function."""

    # Load data
    print("Loading benchmark data...")
    data_file = "/Users/apascoa/IdeaProjects/thesis_proxy/benchmarks/parsed_logs/night_logs_6.json"
    data = load_benchmark_data(data_file)
    print(f"Loaded {len(data)} benchmark entries")

    # Define cost assumptions
    cost_assumptions = define_cost_assumptions()

    # Calculate throughput per dollar
    print("Calculating throughput per dollar...")
    df = calculate_throughput_per_dollar(data, cost_assumptions)

    # Print summary
    print_summary_table(df)

    # Create visualizations
    print("\nCreating visualizations...")
    fig = create_visualizations(df)

    # Save results
    output_file = "/Users/apascoa/IdeaProjects/thesis_proxy/benchmarks/visualizations/throughput_per_dollar_results.csv"
    df.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")

    return df

if __name__ == "__main__":
    df = main()
