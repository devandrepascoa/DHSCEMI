#!/usr/bin/env python3
"""
Visualization script for LLM benchmark results
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path
import argparse

class BenchmarkVisualizer:
    def __init__(self, results_dir):
        self.results_dir = Path(results_dir)
        
    def load_latest_results(self):
        """Load the most recent benchmark results"""
        csv_files = list(self.results_dir.glob('benchmark_results_*.csv'))
        if not csv_files:
            print("No benchmark results found")
            return None
        
        latest_file = max(csv_files, key=lambda x: x.stat().st_mtime)
        return pd.read_csv(latest_file)
    
    def create_throughput_plots(self, df):
        """Create throughput visualization plots"""
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('LLM Benchmark Results - Throughput Analysis', fontsize=16)
        
        # 1. Throughput by Batch Size
        ax1 = axes[0, 0]
        sns.lineplot(data=df, x='batch_size', y='throughput_tok_s', 
                    hue='variant', style='model', ax=ax1)
        ax1.set_title('Throughput vs Batch Size')
        ax1.set_xlabel('Batch Size')
        ax1.set_ylabel('Throughput (tokens/sec)')
        ax1.set_yscale('log')
        
        # 2. Throughput by Sequence Length
        ax2 = axes[0, 1]
        sns.lineplot(data=df, x='sequence_length', y='throughput_tok_s', 
                    hue='variant', style='model', ax=ax2)
        ax2.set_title('Throughput vs Sequence Length')
        ax2.set_xlabel('Sequence Length')
        ax2.set_ylabel('Throughput (tokens/sec)')
        ax2.set_yscale('log')
        
        # 3. CUDA vs CPU comparison
        ax3 = axes[1, 0]
        df_summary = df.groupby(['model', 'variant'])['throughput_tok_s'].mean().reset_index()
        sns.barplot(data=df_summary, x='model', y='throughput_tok_s', hue='variant', ax=ax3)
        ax3.set_title('Average Throughput: CUDA vs CPU')
        ax3.set_xlabel('Model')
        ax3.set_ylabel('Average Throughput (tokens/sec)')
        ax3.tick_params(axis='x', rotation=45)
        
        # 4. Heatmap of throughput by batch size and sequence length
        ax4 = axes[1, 1]
        pivot_data = df.pivot_table(
            values='throughput_tok_s', 
            index=['batch_size'], 
            columns=['sequence_length'], 
            aggfunc='mean'
        )
        sns.heatmap(pivot_data, annot=True, fmt='.1f', cmap='YlOrRd', ax=ax4)
        ax4.set_title('Throughput Heatmap')
        ax4.set_xlabel('Sequence Length')
        ax4.set_ylabel('Batch Size')
        
        plt.tight_layout()
        output_file = self.results_dir / 'throughput_analysis.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Throughput plots saved to {output_file}")
        
    def create_latency_plots(self, df):
        """Create latency visualization plots"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('LLM Benchmark Results - Latency Analysis', fontsize=16)
        
        # 1. Latency distribution
        ax1 = axes[0]
        sns.boxplot(data=df, x='variant', y='latency_ms', ax=ax1)
        ax1.set_title('Latency Distribution by Variant')
        ax1.set_xlabel('Variant')
        ax1.set_ylabel('Latency (ms)')
        ax1.set_yscale('log')
        
        # 2. Latency vs Batch Size
        ax2 = axes[1]
        sns.scatterplot(data=df, x='batch_size', y='latency_ms', 
                       hue='variant', style='model', ax=ax2)
        ax2.set_title('Latency vs Batch Size')
        ax2.set_xlabel('Batch Size')
        ax2.set_ylabel('Latency (ms)')
        ax2.set_yscale('log')
        
        plt.tight_layout()
        output_file = self.results_dir / 'latency_analysis.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Latency plots saved to {output_file}")
        
    def create_model_comparison(self, df):
        """Create model comparison plots"""
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))
        fig.suptitle('Model Performance Comparison', fontsize=16)
        
        # Prepare data
        df_summary = df.groupby(['model', 'variant', 'batch_size']).agg({
            'throughput_tok_s': 'mean',
            'latency_ms': 'mean'
        }).reset_index()
        
        # 1. Throughput comparison
        ax1 = axes[0]
        sns.barplot(data=df_summary, x='model', y='throughput_tok_s', 
                   hue='variant', ax=ax1)
        ax1.set_title('Throughput Comparison by Model and Variant')
        ax1.set_xlabel('Model')
        ax1.set_ylabel('Throughput (tokens/sec)')
        ax1.tick_params(axis='x', rotation=45)
        ax1.legend(title='Variant')
        
        # 2. Latency comparison
        ax2 = axes[1]
        sns.barplot(data=df_summary, x='model', y='latency_ms', 
                   hue='variant', ax=ax2)
        ax2.set_title('Latency Comparison by Model and Variant')
        ax2.set_xlabel('Model')
        ax2.set_ylabel('Latency (ms)')
        ax2.tick_params(axis='x', rotation=45)
        ax2.legend(title='Variant')
        
        plt.tight_layout()
        output_file = self.results_dir / 'model_comparison.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Model comparison plots saved to {output_file}")
        
    def generate_html_report(self, df):
        """Generate HTML report with embedded plots"""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>LLM Benchmark Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .metric {{ background: #f5f5f5; padding: 10px; margin: 10px 0; border-radius: 5px; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #4CAF50; color: white; }}
            </style>
        </head>
        <body>
            <h1>LLM Benchmark Report</h1>
            <p>Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <h2>Summary Statistics</h2>
            <div class="metric">
                <h3>Models Tested: {df['model'].nunique()}</h3>
                <h3>Total Runs: {len(df)}</h3>
                <h3>Variants: {', '.join(df['variant'].unique())}</h3>
            </div>
            
            <h2>Performance Summary</h2>
            {df.groupby(['model', 'variant']).agg({
                'throughput_tok_s': ['mean', 'max', 'min'],
                'latency_ms': ['mean', 'min', 'max']
            }).round(2).to_html()}
            
            <h2>Visualizations</h2>
            <p>Charts have been generated and saved in the results directory:</p>
            <ul>
                <li>throughput_analysis.png - Throughput analysis plots</li>
                <li>latency_analysis.png - Latency analysis plots</li>
                <li>model_comparison.png - Model comparison plots</li>
            </ul>
        </body>
        </html>
        """
        
        report_file = self.results_dir / 'benchmark_report.html'
        with open(report_file, 'w') as f:
            f.write(html_content)
        print(f"HTML report saved to {report_file}")

def main():
    parser = argparse.ArgumentParser(description='Visualize benchmark results')
    parser.add_argument('--results-dir', default='./results', help='Directory containing results')
    
    args = parser.parse_args()
    
    visualizer = BenchmarkVisualizer(args.results_dir)
    df = visualizer.load_latest_results()
    
    if df is not None:
        print("Generating visualizations...")
        visualizer.create_throughput_plots(df)
        visualizer.create_latency_plots(df)
        visualizer.create_model_comparison(df)
        visualizer.generate_html_report(df)
        print("All visualizations completed!")

if __name__ == "__main__":
    main()
