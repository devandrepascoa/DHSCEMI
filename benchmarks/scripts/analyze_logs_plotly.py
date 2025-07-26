#!/usr/bin/env python3
"""
Benchmark Log Analyzer with Plotly and Polars

This script reads all benchmark log files and generates interactive plots using Plotly
and processes data efficiently using Polars instead of pandas/seaborn.
"""

import os
import json
import re
import glob
import polars as pl
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime
import numpy as np
from typing import Dict, List, Tuple, Optional

class BenchmarkLogAnalyzerPlotly:
    def __init__(self, log_directory: str = "/home/andrepascoa/projects/llmtests/benchmarks"):
        self.log_directory = log_directory
        self.data = []

    def extract_quantization(self, model_type: str) -> str:
        """Extract quantization level from model type string."""
        if not model_type:
            return "unknown"

        # Common quantization patterns
        patterns = [
            r'Q4_K_M',
            r'Q4_K_S',
            r'Q4_0',
            r'Q4_1',
            r'Q5_K_M',
            r'Q5_K_S',
            r'Q5_0',
            r'Q5_1',
            r'Q6_K',
            r'Q8_0',
            r'F16',
            r'F32',
            r'IQ4_NL',
            r'IQ4_XS'
        ]

        for pattern in patterns:
            if pattern in model_type.upper():
                return pattern

        # Extract any Q* pattern
        match = re.search(r'Q\d+_[A-Z0-9]+', model_type.upper())
        if match:
            return match.group()

        return "unknown"

    def extract_model_size(self, model_type: str) -> str:
        """Extract model size from model type string."""
        if not model_type:
            return "unknown"

        # Common size patterns
        patterns = [
            r'(\d+\.?\d*[BM])',
            r'(\d+\.?\d*B)',
            r'(\d+\.?\d*M)',
        ]

        for pattern in patterns:
            match = re.search(pattern, model_type, re.IGNORECASE)
            if match:
                return match.group(1).upper()

        # Try to extract from model filename patterns
        size_match = re.search(r'(\d+\.?\d*)[BM]', model_type)
        if size_match:
            return size_match.group(0).upper()

        return "unknown"

    def parse_log_file(self, filepath: str) -> List[Dict]:
        """Parse a single log file and extract benchmark data."""
        results = []

        try:
            with open(filepath, 'r') as f:
                content = f.read()

            # Find all JSON lines in the log
            lines = content.split('\n')

            for line in lines:
                line = line.strip()
                if line.startswith('{') and line.endswith('}'):
                    try:
                        data = json.loads(line)

                        # Only process valid benchmark results
                        if 'avg_ts' in data and 'model_type' in data:
                            result = {
                                'timestamp': data.get('test_time', ''),
                                'model_filename': data.get('model_filename', ''),
                                'model_type': data.get('model_type', ''),
                                'model_size_bytes': data.get('model_size', 0),
                                'n_params': data.get('model_n_params', 0),
                                'device_type': 'GPU' if data.get('backends') == 'CUDA' else 'CPU',
                                'batch_size': data.get('n_batch', 1),
                                'throughput': data.get('avg_ts', 0),
                                'stddev_throughput': data.get('stddev_ts', 0),
                                'quantization': self.extract_quantization(data.get('model_type', '')),
                                'model_size': self.extract_model_size(data.get('model_type', '')),
                                'log_file': os.path.basename(filepath)
                            }
                            results.append(result)

                    except json.JSONDecodeError:
                        continue

        except Exception as e:
            print(f"Error parsing {filepath}: {e}")

        return results

    def load_all_logs(self) -> None:
        """Load and parse all log files in the directory."""
        log_pattern = os.path.join(self.log_directory, "log_*.log")
        log_files = glob.glob(log_pattern)

        print(f"Found {len(log_files)} log files")

        for log_file in log_files:
            print(f"Processing {log_file}...")
            results = self.parse_log_file(log_file)
            self.data.extend(results)
            print(f"  Extracted {len(results)} benchmark results")

        print(f"Total benchmark results: {len(self.data)}")

    def create_polars_dataframe(self) -> pl.DataFrame:
        """Create a Polars DataFrame from the collected data."""
        if not self.data:
            print("No data found!")
            return pl.DataFrame()

        # Create Polars DataFrame
        df = pl.DataFrame(self.data)

        # Handle missing data and type conversion
        df = df.filter(
            pl.col('throughput').is_not_null() &
            pl.col('model_type').is_not_null() &
            pl.col('device_type').is_not_null()
        )

        # Convert numeric columns
        df = df.with_columns([
            pl.col('throughput').cast(pl.Float64),
            pl.col('model_size_bytes').cast(pl.Int64),
            pl.col('n_params').cast(pl.Int64),
            pl.col('batch_size').cast(pl.Int64),
            pl.col('stddev_throughput').cast(pl.Float64)
        ])

        # Add model size category
        df = df.with_columns([
            pl.when(pl.col('model_size').str.contains(r'(?i)32B|70B|65B'))
              .then(pl.lit('Large'))
              .when(pl.col('model_size').str.contains(r'(?i)7B|13B'))
              .then(pl.lit('Medium'))
              .when(pl.col('model_size').str.contains(r'(?i)1\.5B|3B|6B'))
              .then(pl.lit('Small'))
              .otherwise(pl.lit('Unknown'))
              .alias('model_size_category')
        ])

        return df

    def plot_throughput_by_quantization(self, df: pl.DataFrame) -> None:
        """Create interactive plot of throughput by quantization level."""
        if df.is_empty():
            return

        # Group by quantization and device type
        grouped = df.group_by(['quantization', 'device_type']).agg([
            pl.col('throughput').mean().alias('mean_throughput'),
            pl.col('throughput').std().alias('std_throughput'),
            pl.count().alias('count')
        ]).sort('quantization', 'device_type')

        # Create interactive bar plot
        fig = px.box(
            df.to_pandas(),
            x='quantization',
            y='throughput',
            color='device_type',
            title='Throughput by Quantization Level and Device Type',
            labels={'throughput': 'Throughput (tokens/second)', 'quantization': 'Quantization Level'},
            category_orders={'quantization': sorted(df['quantization'].unique())}
        )

        fig.update_layout(
            xaxis_title='Quantization Level',
            yaxis_title='Throughput (tokens/second)',
            boxmode='group'
        )

        fig.write_html('throughput_by_quantization.html')
        print("Saved: throughput_by_quantization.html")

    def plot_throughput_by_model_size(self, df: pl.DataFrame) -> None:
        """Create interactive scatter plot of throughput vs model size."""
        if df.is_empty():
            return

        # Create interactive scatter plot
        fig = px.scatter(
            df.to_pandas(),
            x='n_params',
            y='throughput',
            color='device_type',
            size='batch_size',
            hover_data=['model_type', 'quantization', 'batch_size'],
            log_x=True,
            log_y=True,
            title='Throughput vs Model Parameters'
        )

        fig.update_layout(
            xaxis_title='Model Parameters',
            yaxis_title='Throughput (tokens/second)'
        )

        fig.write_html('throughput_by_model_size.html')
        print("Saved: throughput_by_model_size.html")

    def plot_throughput_by_device(self, df: pl.DataFrame) -> None:
        """Create interactive device comparison plot."""
        if df.is_empty():
            return

        fig = px.violin(
            df.to_pandas(),
            x='device_type',
            y='throughput',
            box=True,
            points='all',
            title='Throughput Distribution by Device Type',
            color='device_type'
        )

        fig.update_layout(
            xaxis_title='Device Type',
            yaxis_title='Throughput (tokens/second)'
        )

        fig.write_html('throughput_by_device.html')
        print("Saved: throughput_by_device.html")

    def plot_throughput_by_batch_size(self, df: pl.DataFrame) -> None:
        """Create interactive plot of throughput vs batch size."""
        if df.is_empty():
            return

        fig = px.scatter(
            df.to_pandas(),
            x='batch_size',
            y='throughput',
            color='device_type',
            size='n_params',
            hover_data=['model_type', 'quantization'],
            log_x=True,
            log_y=True,
            title='Throughput vs Batch Size'
        )

        fig.update_layout(
            xaxis_title='Batch Size',
            yaxis_title='Throughput (tokens/second)'
        )

        fig.write_html('throughput_by_batch_size.html')
        print("Saved: throughput_by_batch_size.html")

    def plot_comprehensive_dashboard(self, df: pl.DataFrame) -> None:
        """Create interactive comprehensive dashboard."""
        if df.is_empty():
            return

        # Create subplot figure
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Throughput by Quantization', 'Throughput by Model Size',
                          'Device Comparison', 'Batch Size Impact'),
            specs=[[{"type": "box"}, {"type": "scatter"}],
                   [{"type": "violin"}, {"type": "scatter"}]]
        )

        # 1. Throughput by Quantization
        for device in df['device_type'].unique():
            device_data = df.filter(pl.col('device_type') == device)
            fig.add_trace(
                go.Box(
                    x=device_data['quantization'].to_list(),
                    y=device_data['throughput'].to_list(),
                    name=f"{device}",
                    boxpoints='outliers'
                ),
                row=1, col=1
            )

        # 2. Throughput by Model Size
        for device in df['device_type'].unique():
            device_data = df.filter(pl.col('device_type') == device)
            fig.add_trace(
                go.Scatter(
                    x=device_data['n_params'].to_list(),
                    y=device_data['throughput'].to_list(),
                    mode='markers',
                    name=f"{device}",
                    marker=dict(size=8)
                ),
                row=1, col=2
            )

        # 3. Device Comparison
        for device in df['device_type'].unique():
            device_data = df.filter(pl.col('device_type') == device)
            fig.add_trace(
                go.Violin(
                    x=device_data['device_type'].to_list(),
                    y=device_data['throughput'].to_list(),
                    name=f"{device}",
                    box=dict(visible=True),
                    meanline=dict(visible=True)
                ),
                row=2, col=1
            )

        # 4. Batch Size Impact
        for device in df['device_type'].unique():
            device_data = df.filter(pl.col('device_type') == device)
            fig.add_trace(
                go.Scatter(
                    x=device_data['batch_size'].to_list(),
                    y=device_data['throughput'].to_list(),
                    mode='markers',
                    name=f"{device}",
                    marker=dict(size=8)
                ),
                row=2, col=2
            )

        # Update layout
        fig.update_layout(
            title_text="Benchmark Results Dashboard",
            height=800,
            showlegend=True
        )

        # Update axes
        fig.update_xaxes(type="log", row=1, col=2)
        fig.update_yaxes(type="log", row=1, col=2)
        fig.update_xaxes(type="log", row=2, col=2)
        fig.update_yaxes(type="log", row=2, col=2)

        fig.write_html('benchmark_dashboard.html')
        print("Saved: benchmark_dashboard.html")

    def generate_summary_report(self, df: pl.DataFrame) -> None:
        """Generate a summary report of the analysis."""
        if df.is_empty():
            print("No data to generate report!")
            return

        # Calculate statistics using Polars
        device_stats = df.group_by('device_type').agg([
            pl.col('throughput').mean().alias('mean_throughput'),
            pl.col('throughput').std().alias('std_throughput'),
            pl.col('throughput').min().alias('min_throughput'),
            pl.col('throughput').max().alias('max_throughput')
        ])

        # Top performers
        top_performers = df.sort('throughput', descending=True).limit(5).select([
            'model_type', 'device_type', 'quantization', 'throughput'
        ])

        report = []
        report.append("=" * 50)
        report.append("BENCHMARK ANALYSIS SUMMARY (Plotly + Polars)")
        report.append("=" * 50)
        report.append(f"Total benchmark runs: {df.height}")
        report.append(f"Unique models tested: {df['model_type'].n_unique()}")
        report.append(f"Unique quantizations: {df['quantization'].n_unique()}")
        report.append(f"Device types: {', '.join(df['device_type'].unique())}")
        report.append("")

        # Performance summary
        report.append("PERFORMANCE SUMMARY")
        report.append("-" * 30)
        for row in device_stats.iter_rows(named=True):
            report.append(f"{row['device_type']}: {row['mean_throughput']:.2f} ± {row['std_throughput']:.2f} tokens/sec")
        report.append("")

        # Top performers
        report.append("TOP PERFORMERS")
        report.append("-" * 30)
        for row in top_performers.iter_rows(named=True):
            report.append(f"{row['model_type']} ({row['device_type']}, {row['quantization']})")
            report.append(f"  Throughput: {row['throughput']:.2f} tokens/sec")
        report.append("")

        # Save report
        with open('benchmark_summary_plotly.txt', 'w') as f:
            f.write('\n'.join(report))

        print("\n".join(report))

    def run_analysis(self) -> None:
        """Run the complete analysis pipeline with Plotly and Polars."""
        print("Starting benchmark log analysis with Plotly + Polars...")

        # Load all logs
        self.load_all_logs()

        # Create Polars DataFrame
        df = self.create_polars_dataframe()

        if df.is_empty():
            print("No data found in logs!")
            return

        print(f"Data loaded: {df.height} benchmark results")

        # Save raw data
        df.write_csv('benchmark_data_polars.csv')
        print("Raw data saved to benchmark_data_polars.csv")

        # Generate interactive plots
        print("Generating interactive plots...")
        self.plot_throughput_by_quantization(df)
        self.plot_throughput_by_model_size(df)
        self.plot_throughput_by_device(df)
        self.plot_throughput_by_batch_size(df)
        self.plot_comprehensive_dashboard(df)
        print("All interactive plots saved as HTML files!")

        # Generate summary
        self.generate_summary_report(df)

if __name__ == "__main__":
    analyzer = BenchmarkLogAnalyzerPlotly()
    analyzer.run_analysis()
