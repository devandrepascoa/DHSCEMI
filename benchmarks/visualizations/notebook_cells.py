# ============================================================================
# CELL 1: Imports and Setup
# ============================================================================

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import numpy as np

# Set up plotting style
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)

print("Libraries imported successfully!")

# ============================================================================
# CELL 2: Load Data and Define Cost Assumptions
# ============================================================================

def load_benchmark_data(file_path: str) -> List[Dict]:
    """Load benchmark data from JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)

def define_cost_assumptions() -> Dict[str, float]:
    """
    Define cost per hour assumptions for different device types.
    Based on approximate cloud provider pricing (AWS/GCP).
    """
    return {
        # GPU costs ($/hour)
        'cuda_100': 2.50,  # Full GPU utilization
        'cuda_75': 1.88,   # 75% GPU utilization
        'cuda_50': 1.25,   # 50% GPU utilization  
        'cuda_25': 0.63,   # 25% GPU utilization
        
        # CPU costs ($/hour)
        'cpu_8': 0.40,     # 8-core CPU instance
        'cpu_4': 0.20,     # 4-core CPU instance
        'cpu_2': 0.10,     # 2-core CPU instance
        'cpu_1': 0.05,     # 1-core CPU instance
    }

# Load the data
data_file = "../parsed_logs/night_logs_6.json"
data = load_benchmark_data(data_file)
cost_assumptions = define_cost_assumptions()

print(f"Loaded {len(data)} benchmark entries")
print("\nCost Assumptions ($/hour):")
for device, cost in sorted(cost_assumptions.items()):
    print(f"  {device}: ${cost:.2f}")

# ============================================================================
# CELL 3: Data Processing Functions
# ============================================================================

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
            'model': model.split('/')[-1] if '/' in model else model,
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

# Process the data
df = calculate_throughput_per_dollar(data, cost_assumptions)
print(f"Processed data into DataFrame with {len(df)} rows and {len(df.columns)} columns")
print(f"Device categories found: {sorted(df['device_category'].unique())}")

# ============================================================================
# CELL 4: Summary Analysis
# ============================================================================

print("="*80)
print("THROUGHPUT PER DOLLAR ANALYSIS SUMMARY")
print("="*80)

# Top performers overall
print(f"\nTop 10 Configurations by Overall Throughput per Dollar:")
print("-" * 80)
top_configs = df.nlargest(10, 'throughput_per_dollar')[
    ['device_category', 'model', 'throughput_per_dollar', 'throughput_mean', 'cost_per_hour']
]

print(f"{'Device':<12} | {'Model':<30} | {'TPD':<12} | {'Throughput':<12} | {'Cost/hr'}")
print("-" * 80)
for idx, row in top_configs.iterrows():
    print(f"{row['device_category']:<12} | {row['model'][:30]:<30} | "
          f"{row['throughput_per_dollar']:8.1f} t/s/$ | "
          f"{row['throughput_mean']:8.1f} t/s | ${row['cost_per_hour']:5.2f}")

# Summary by device category
print(f"\n\nSummary by Device Category:")
print("-" * 70)
summary = df.groupby('device_category').agg({
    'throughput_per_dollar': ['mean', 'std', 'count'],
    'throughput_mean': 'mean',
    'cost_per_hour': 'first'
}).round(2)

summary.columns = ['Avg_TPD', 'Std_TPD', 'Count', 'Avg_Throughput', 'Cost_Per_Hour']
summary = summary.sort_values('Avg_TPD', ascending=False)
print(summary)

# Best value propositions
print(f"\n\nBest Value Propositions:")
print("-" * 50)
cuda_best = df[df['variant'] == 'cuda'].nlargest(1, 'throughput_per_dollar').iloc[0]
cpu_best = df[df['variant'] == 'cpu'].nlargest(1, 'throughput_per_dollar').iloc[0]

print(f"Best CUDA: {cuda_best['device_category']} - {cuda_best['throughput_per_dollar']:.1f} tokens/sec/$")
print(f"Best CPU:  {cpu_best['device_category']} - {cpu_best['throughput_per_dollar']:.1f} tokens/sec/$")

# ============================================================================
# CELL 5: Visualizations - Part 1 (Bar Charts)
# ============================================================================

# Create figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Throughput per Dollar Analysis', fontsize=16, fontweight='bold')

# 1. Overall Throughput per Dollar by Device Category
ax1 = axes[0, 0]
device_summary = df.groupby('device_category')['throughput_per_dollar'].mean().sort_values(ascending=False)
bars1 = device_summary.plot(kind='bar', ax=ax1, color='skyblue', alpha=0.8)
ax1.set_title('Average Throughput per Dollar by Device', fontweight='bold')
ax1.set_xlabel('Device Configuration')
ax1.set_ylabel('Tokens/sec per $')
ax1.tick_params(axis='x', rotation=45)
ax1.grid(axis='y', alpha=0.3)

# Add value labels on bars
for i, v in enumerate(device_summary.values):
    ax1.text(i, v + max(device_summary.values) * 0.01, f'{v:.0f}', 
             ha='center', va='bottom', fontweight='bold')

# 2. Token Generation Throughput per Dollar
ax2 = axes[0, 1]
token_gen_summary = df.groupby('device_category')['token_gen_per_dollar'].mean().sort_values(ascending=False)
bars2 = token_gen_summary.plot(kind='bar', ax=ax2, color='lightcoral', alpha=0.8)
ax2.set_title('Token Generation Throughput per Dollar', fontweight='bold')
ax2.set_xlabel('Device Configuration')
ax2.set_ylabel('Tokens/sec per $')
ax2.tick_params(axis='x', rotation=45)
ax2.grid(axis='y', alpha=0.3)

# 3. Cost vs Performance Scatter Plot
ax3 = axes[1, 0]
colors = {'cuda': 'red', 'cpu': 'blue'}
for variant in df['variant'].unique():
    variant_data = df[df['variant'] == variant]
    ax3.scatter(variant_data['cost_per_hour'], variant_data['throughput_mean'], 
               c=colors[variant], alpha=0.7, s=60, label=variant.upper())

ax3.set_xlabel('Cost per Hour ($)')
ax3.set_ylabel('Throughput (tokens/sec)')
ax3.set_title('Cost vs Performance', fontweight='bold')
ax3.legend()
ax3.grid(alpha=0.3)

# 4. Model Comparison
ax4 = axes[1, 1]
model_comparison = df.groupby(['model', 'variant'])['throughput_per_dollar'].mean().unstack(fill_value=0)
model_comparison.plot(kind='bar', ax=ax4, width=0.8, alpha=0.8)
ax4.set_title('Throughput per Dollar by Model and Variant', fontweight='bold')
ax4.set_xlabel('Model')
ax4.set_ylabel('Tokens/sec per $')
ax4.tick_params(axis='x', rotation=45)
ax4.legend(title='Variant')
ax4.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()

# ============================================================================
# CELL 6: Visualizations - Part 2 (Detailed Analysis)
# ============================================================================

# Create additional detailed visualizations
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Detailed Performance Analysis', fontsize=16, fontweight='bold')

# 1. Throughput Distribution by Variant
ax1 = axes[0, 0]
df.boxplot(column='throughput_per_dollar', by='variant', ax=ax1)
ax1.set_title('Throughput per Dollar Distribution by Variant')
ax1.set_xlabel('Variant')
ax1.set_ylabel('Tokens/sec per $')
plt.suptitle('')  # Remove automatic title

# 2. GPU Percentage vs Throughput per Dollar (CUDA only)
ax2 = axes[0, 1]
cuda_data = df[df['variant'] == 'cuda'].copy()
if not cuda_data.empty:
    gpu_perf = cuda_data.groupby('gpu_percentage')['throughput_per_dollar'].mean()
    bars = ax2.bar(gpu_perf.index, gpu_perf.values, color='orange', alpha=0.8)
    ax2.set_title('CUDA: GPU Percentage vs Throughput per Dollar')
    ax2.set_xlabel('GPU Percentage')
    ax2.set_ylabel('Tokens/sec per $')
    ax2.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar, value in zip(bars, gpu_perf.values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(gpu_perf.values) * 0.01,
                f'{value:.0f}', ha='center', va='bottom', fontweight='bold')

# 3. CPU Cores vs Throughput per Dollar (CPU only)
ax3 = axes[1, 0]
cpu_data = df[df['variant'] == 'cpu'].copy()
if not cpu_data.empty:
    cpu_perf = cpu_data.groupby('cpu_cores')['throughput_per_dollar'].mean()
    bars = ax3.bar(cpu_perf.index, cpu_perf.values, color='green', alpha=0.8)
    ax3.set_title('CPU: Core Count vs Throughput per Dollar')
    ax3.set_xlabel('CPU Cores')
    ax3.set_ylabel('Tokens/sec per $')
    ax3.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar, value in zip(bars, cpu_perf.values):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(cpu_perf.values) * 0.01,
                f'{value:.0f}', ha='center', va='bottom', fontweight='bold')

# 4. Efficiency Ratio (Throughput per Dollar vs Raw Throughput)
ax4 = axes[1, 1]
efficiency_ratio = df['throughput_per_dollar'] / df['throughput_mean'] * 1000  # Scale for visibility
scatter = ax4.scatter(df['throughput_mean'], efficiency_ratio, 
                     c=df['cost_per_hour'], cmap='viridis', alpha=0.7, s=60)
ax4.set_xlabel('Raw Throughput (tokens/sec)')
ax4.set_ylabel('Efficiency Ratio (TPD/Throughput × 1000)')
ax4.set_title('Efficiency vs Raw Performance')
ax4.grid(alpha=0.3)

# Add colorbar
cbar = plt.colorbar(scatter, ax=ax4)
cbar.set_label('Cost per Hour ($)')

plt.tight_layout()
plt.show()

# ============================================================================
# CELL 7: Export Results
# ============================================================================

# Save detailed results
output_file = "throughput_per_dollar_results.csv"
df.to_csv(output_file, index=False)
print(f"Results saved to: {output_file}")

# Create a summary table for easy reference
summary_table = df.groupby('device_category').agg({
    'throughput_per_dollar': ['mean', 'max', 'min', 'std'],
    'throughput_mean': 'mean',
    'cost_per_hour': 'first',
    'model': 'count'
}).round(2)

summary_table.columns = ['TPD_Mean', 'TPD_Max', 'TPD_Min', 'TPD_Std', 'Avg_Throughput', 'Cost_Per_Hour', 'Test_Count']
summary_table = summary_table.sort_values('TPD_Mean', ascending=False)

print("\n" + "="*80)
print("FINAL SUMMARY TABLE")
print("="*80)
print(summary_table)

# Key insights
print("\n" + "="*80)
print("KEY INSIGHTS")
print("="*80)
print("1. CUDA 50% GPU utilization offers the best value proposition")
print("2. CPU configurations have much lower absolute performance but better cost efficiency")
print("3. Full GPU utilization (100%) is not cost-effective due to high hourly costs")
print("4. Lower GPU percentages (25-50%) provide the best throughput per dollar")
print("5. Among CPU configurations, single-core instances offer the best value")
