#!/usr/bin/env python3

import os
import json
import re
import glob
import argparse
import polars as pl
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime
import numpy as np
from typing import Dict, List, Tuple, Optional

class BenchmarkLogAnalyzer:
    def __init__(self, log_file: str = None, json_file: str = None):
        self.log_file = log_file
        self.json_file = json_file
        self.data = []
        self.raw_data = []

    def load_json_result_file(self, filepath: str) -> List[Dict]:
        """Load data from a pre-parsed JSON result file."""
        results = []
        
        try:
            with open(filepath, 'r') as f:
                json_data = json.load(f)
            
            print(f"Loaded {len(json_data)} entries from JSON file")
            
            for data in json_data:
                # Extract relevant metrics from the JSON structure
                gpu_percentage = data.get('gpu_percentage', 0)
                cpu_cores = data.get('cpu_cores', 0)
                
                # Determine device type
                if gpu_percentage and gpu_percentage > 0:
                    device_type = f"GPU_{gpu_percentage}%_CPU_{cpu_cores}"
                else:
                    device_type = f"CPU_{cpu_cores}_cores"
                
                # Extract model information
                model_filename = data.get('model', '')
                model_type = model_filename.replace('.gguf', '') if model_filename else ''
                
                # Calculate prompt length (approximate from prompt text)
                prompt = data.get('prompt', '')
                try:
                    import tiktoken
                    tokenizer = tiktoken.get_encoding("cl100k_base")
                    prompt_length = len(tokenizer.encode(prompt))
                except:
                    # Fallback to character-based approximation
                    prompt_length = len(prompt) // 4
                
                # Extract quantization and model size
                quantization = self.extract_quantization(model_type)
                model_size = self.extract_model_size(model_type)
                
                # Calculate n_params from model size
                n_params = 0
                if model_size != "unknown":
                    try:
                        size_str = model_size.replace('B', '').replace('M', '')
                        size_val = float(size_str)
                        if 'B' in model_size.upper():
                            n_params = int(size_val * 1e9)
                        elif 'M' in model_size.upper():
                            n_params = int(size_val * 1e6)
                    except:
                        pass
                
                result = {
                    'endpoint': data.get('endpoint', ''),
                    'model_filename': model_filename,
                    'model_type': model_type,
                    'device_type': device_type,
                    'quantization': quantization,
                    'model_size': model_size,
                    'n_params': n_params,
                    'throughput': data.get('throughput_mean', 0),
                    'stddev_throughput': data.get('throughput_stddev', 0),
                    'batch_size': data.get('concurrent_requests', 1),
                    'gpu_percentage': gpu_percentage or 0,
                    'cpu_cores': cpu_cores or 0,
                    'prompt_length': prompt_length,
                    'variant': data.get('variant', ''),
                    'token_size': data.get('token_size', 0),
                    'prompt_processing_throughput': data.get('prompt_processing_throughput_mean', 0),
                    'token_generation_throughput': data.get('token_generation_throughput_mean', 0),
                    'time_to_first_token': data.get('time_to_first_token_mean', 0),
                    'total_time': data.get('total_time_mean', 0),
                    'avg_time_per_request': data.get('avg_time_per_request_mean', 0)
                }
                
                results.append(result)
                self.raw_data.append(data)
                
        except Exception as e:
            print(f"Error loading JSON file {filepath}: {e}")
        
        return results

    def parse_log_file(self, filepath: str) -> List[Dict]:
        results = []

        try:
            with open(filepath, 'r') as f:
                content = f.read()

            lines = content.split('\n')
            print(len(lines))

            for line in lines:
                line = line.strip()

                if "stats_result" in line:
                    try:
                        json_str = line[len('stats_result: '): ]
                        import ast
                        data = ast.literal_eval(json_str)
                        data = json.loads(json.dumps(data))

                        gpu_percentage = data.get('gpu_percentage', 0)
                        cpu_cores = data.get('cpu_cores', 0)

                        if gpu_percentage is None:
                            gpu_percentage = 0
                        if cpu_cores is None:
                            cpu_cores = 0

                        if gpu_percentage > 0:
                            if gpu_percentage >= 80:
                                device_type = f'GPU_High_{gpu_percentage}%'
                            elif gpu_percentage >= 50:
                                device_type = f'GPU_Med_{gpu_percentage}%'
                            else:
                                device_type = f'GPU_Low_{gpu_percentage}%'
                        else:
                            if cpu_cores > 0:
                                device_type = f'CPU_{cpu_cores}_cores'
                            else:
                                device_type = 'CPU_unknown'

                        result = {
                            'timestamp': data.get('test_time', ''),
                            'model_filename': data.get('model', ''),
                            'model_type': data.get('model', ''),
                            'model_size_bytes': 0,
                            'n_params': 0,
                            'device_type': device_type,
                            'gpu_percentage': gpu_percentage,
                            'cpu_cores': cpu_cores,
                            'batch_size': data.get('concurrent_requests', 1),
                            'throughput': data.get('throughput_mean', 0),
                            'stddev_throughput': data.get('throughput_stddev', 0),
                            'quantization': self.extract_quantization(data.get('model', '')),
                            'model_size': self.extract_model_size(data.get('model', '')),
                            'prompt_length': len(data.get('prompt', '')),
                            'log_file': os.path.basename(filepath)
                        }

                        model_name = data.get('model', '')
                        if model_name:
                            param_match = re.search(r'(\d+\.?\d*)[BM]', model_name)
                            if param_match:
                                size_str = param_match.group(1)
                                if 'B' in model_name.upper():
                                    result['n_params'] = int(float(size_str) * 1e9)
                                elif 'M' in model_name.upper():
                                    result['n_params'] = int(float(size_str) * 1e6)

                        results.append(result)
                        self.raw_data.append(data)

                    except json.JSONDecodeError:
                        print("Error decoding JSON data in log file")
                        continue

        except Exception as e:
            print(f"Error parsing {filepath}: {e}")

        return results

    def load_all_logs(self) -> None:
        if self.json_file:
            print(f"Processing JSON file: {self.json_file}...")
            results = self.load_json_result_file(self.json_file)
        elif self.log_file:
            print(f"Processing raw log file: {self.log_file}...")
            results = self.parse_log_file(self.log_file)
        else:
            print("No input file specified!")
            return
            
        self.data.extend(results)
        print(f"Extracted {len(results)} benchmark results")

    def extract_quantization(self, model_type: str) -> str:
        if not model_type:
            return "unknown"

        patterns = [
            r'Q4_K_M', r'Q4_K_S', r'Q4_0', r'Q4_1',
            r'Q5_K_M', r'Q5_K_S', r'Q5_0', r'Q5_1',
            r'Q6_K', r'Q8_0', r'F16', r'F32', r'BF16',
            r'IQ4_NL', r'IQ4_XS'
        ]

        for pattern in patterns:
            if pattern in model_type.upper():
                return pattern

        match = re.search(r'Q\d+_[A-Z0-9]+', model_type.upper())
        if match:
            return match.group()

        return "unknown"

    def extract_model_size(self, model_type: str) -> str:
        if not model_type:
            return "unknown"

        patterns = [r'(\d+\.?\d*[BM])', r'(\d+\.?\d*B)', r'(\d+\.?\d*M)']

        for pattern in patterns:
            match = re.search(pattern, model_type, re.IGNORECASE)
            if match:
                return match.group(1).upper()

        size_match = re.search(r'(\d+\.?\d*)[BM]', model_type)
        if size_match:
            return size_match.group(0).upper()

        return "unknown"

    def create_polars_dataframe(self) -> pl.DataFrame:
        if not self.data:
            print("No data found!")
            return pl.DataFrame()

        df = pl.DataFrame(self.data)

        df = df.filter(
            pl.col('throughput').is_not_null() &
            pl.col('model_type').is_not_null() &
            pl.col('device_type').is_not_null()
        )

        df = df.with_columns([
            pl.col('throughput').cast(pl.Float64),
            pl.col('stddev_throughput').cast(pl.Float64),
            pl.col('batch_size').cast(pl.Int64),
            pl.col('n_params').cast(pl.Int64),
            pl.col('gpu_percentage').cast(pl.Int64),
            pl.col('cpu_cores').cast(pl.Int64),
            pl.col('prompt_length').cast(pl.Int64)
        ])

        return df

    def plot_throughput_by_quantization(self, df: pl.DataFrame, output_dir: str) -> None:
        if df.is_empty():
            return

        grouped = df.group_by('quantization').agg([
            pl.col('throughput').mean().alias('avg_throughput'),
            pl.col('throughput').count().alias('count')
        ]).sort('avg_throughput', descending=True)

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=grouped['quantization'].to_list(),
            y=grouped['avg_throughput'].to_list(),
            mode='lines+markers',
            name='Average Throughput',
            line=dict(width=3),
            marker=dict(size=10)
        ))

        fig.update_layout(
            title='Average Throughput by Quantization Level',
            xaxis_title='Quantization Level',
            yaxis_title='Average Throughput (tokens/second)',
            height=600
        )

        fig.write_html(os.path.join(output_dir, 'throughput_by_quantization_line.html'))
        print(f"Saved: throughput_by_quantization_line.html")

    def plot_throughput_by_model_size(self, df: pl.DataFrame, output_dir: str) -> None:
        if df.is_empty():
            return

        def extract_model_family(model_name: str) -> str:
            base_name = re.sub(r'^\d+-', '', model_name)
            base_name = re.sub(r'\.gguf.*$', '', base_name, flags=re.IGNORECASE)

            if 'deepseek-r1' in base_name.lower():
                return 'deepseek-r1'
            elif 'qwen3' in base_name.lower():
                return 'qwen3'
            else:
                parts = base_name.split('-')
                if len(parts) >= 2:
                    return '-'.join(parts[:2]).lower()
                return base_name.lower()

        df = df.with_columns([
            pl.col('model_type').map_elements(extract_model_family, return_dtype=pl.Utf8).alias('model_family')
        ])

        quantization_colors = {
            'Q4_K_M': '#FF6B6B',
            'Q4_K_S': '#FF8E8E',
            'Q4_0': '#FFB3B3',
            'Q4_1': '#FFD6D6',
            'Q5_K_M': '#4ECDC4',
            'Q5_K_S': '#7FDDDD',
            'Q5_0': '#A6E8E8',
            'Q5_1': '#CCF2F2',
            'Q6_K': '#45B7D1',
            'Q8_0': '#96CEB4',
            'F16': '#FECA57',
            'F32': '#FF9FF3',
            'BF16': '#FFA726',
            'IQ4_NL': '#A55EEA',
            'IQ4_XS': '#C44569',
            'unknown': '#95A5A6'
        }

        model_families = df['model_family'].unique().sort().to_list()

        print(f"Creating throughput vs model size plots for {len(model_families)} model families...")

        for model_family in model_families:
            family_data = df.filter(pl.col('model_family') == model_family)

            if family_data.is_empty():
                continue

            fig = go.Figure()

            quantizations = family_data['quantization'].unique().sort().to_list()

            for quantization in quantizations:
                quant_data = family_data.filter(pl.col('quantization') == quantization)

                if quant_data.is_empty():
                    continue

                grouped = quant_data.group_by(['model_size', 'n_params']).agg([
                    pl.col('throughput').mean().alias('avg_throughput'),
                    pl.col('throughput').count().alias('count')
                ]).sort('n_params')

                if grouped.is_empty():
                    continue

                color = quantization_colors.get(quantization, '#95A5A6')

                fig.add_trace(go.Scatter(
                    x=grouped['n_params'].to_list(),
                    y=grouped['avg_throughput'].to_list(),
                    mode='lines+markers',
                    name=quantization,
                    line=dict(width=3, color=color),
                    marker=dict(size=10, color=color, opacity=0.8),
                    text=[f"Model Family: {model_family}<br>Quantization: {quantization}<br>Model Size: {size}<br>Parameters: {params:,}<br>Avg Throughput: {throughput:.2f}<br>Count: {count}"
                          for size, params, throughput, count in zip(
                              grouped['model_size'].to_list(),
                              grouped['n_params'].to_list(),
                              grouped['avg_throughput'].to_list(),
                              grouped['count'].to_list()
                          )],
                    hovertemplate='<b>%{text}</b><extra></extra>'
                ))

            fig.update_layout(
                title=f'Throughput vs Model Size - {model_family.upper()}',
                xaxis_title='Model Parameters',
                yaxis_title='Average Throughput (tokens/second)',
                xaxis=dict(tickformat='.0s'),
                height=600,
                legend=dict(
                    title="Quantization Level",
                    orientation="v",
                    yanchor="top",
                    y=1,
                    xanchor="left",
                    x=1.02
                )
            )

            safe_family_name = re.sub(r'[^\w\-_\.]', '_', model_family)
            filename = f'throughput_vs_model_size_{safe_family_name}_line.html'
            fig.write_html(os.path.join(output_dir, filename))
            print(f"Saved: {filename}")

        fig_summary = go.Figure()

        for model_family in model_families:
            family_data = df.filter(pl.col('model_family') == model_family)

            if family_data.is_empty():
                continue

            grouped = family_data.group_by(['model_size', 'n_params']).agg([
                pl.col('throughput').mean().alias('avg_throughput'),
                pl.col('quantization').first().alias('dominant_quantization')
            ]).sort('n_params')

            if grouped.is_empty():
                continue

            family_colors = {
                'deepseek-r1': '#FF6B6B',
                'qwen3': '#4ECDC4'
            }
            color = family_colors.get(model_family, '#95A5A6')

            fig_summary.add_trace(go.Scatter(
                x=grouped['n_params'].to_list(),
                y=grouped['avg_throughput'].to_list(),
                mode='lines+markers',
                name=model_family,
                line=dict(width=3, color=color),
                marker=dict(size=8, color=color, opacity=0.8)
            ))

        fig_summary.update_layout(
            title='Throughput vs Model Size - All Model Families Summary',
            xaxis_title='Model Parameters',
            yaxis_title='Average Throughput (tokens/second)',
            xaxis=dict(tickformat='.0s'),
            height=600,
            legend=dict(
                title="Model Family",
                orientation="v",
                yanchor="top",
                y=1,
                xanchor="left",
                x=1.02
            )
        )

        fig_summary.write_html(os.path.join(output_dir, 'throughput_vs_model_size_summary_line.html'))
        print("Saved: throughput_vs_model_size_summary_line.html")

    def plot_throughput_by_device(self, df: pl.DataFrame, output_dir: str) -> None:
        if df.is_empty():
            return

        def extract_model_family(model_name: str) -> str:
            base_name = re.sub(r'^\d+-', '', model_name)
            base_name = re.sub(r'\.gguf.*$', '', base_name, flags=re.IGNORECASE)

            if 'deepseek-r1' in base_name.lower():
                return 'deepseek-r1'
            elif 'qwen3' in base_name.lower():
                return 'qwen3'
            else:
                parts = base_name.split('-')
                if len(parts) >= 2:
                    return '-'.join(parts[:2]).lower()
                return base_name.lower()

        df = df.with_columns([
            pl.col('model_type').map_elements(extract_model_family, return_dtype=pl.Utf8).alias('model_family')
        ])

        df = df.with_columns([
            pl.concat_str([pl.col('model_family'), pl.lit('-'), pl.col('quantization')]).alias('model_quant_combo')
        ])

        def get_device_sort_key(device_type: str) -> tuple:
            device_lower = device_type.lower()

            if 'cpu' in device_lower and 'gpu' not in device_lower:
                if '2_cores' in device_lower:
                    return (0, 2)
                elif '4_cores' in device_lower:
                    return (0, 4)
                elif '8_cores' in device_lower:
                    return (0, 8)
                elif '16_cores' in device_lower:
                    return (0, 16)
                else:
                    return (0, 999)

            elif 'gpu' in device_lower:
                if '25%' in device_lower or 'low_25' in device_lower:
                    return (1, 25)
                elif '50%' in device_lower or 'medium_50' in device_lower:
                    return (1, 50)
                elif '75%' in device_lower or 'high_75' in device_lower:
                    return (1, 75)
                elif '100%' in device_lower or 'high_100' in device_lower:
                    return (1, 100)
                else:
                    return (1, 999)

            else:
                return (2, 999)

        model_size_colors = {
            '1.5B': '#FF6B6B',
            '1.7B': '#FF8E8E',
            '7B': '#4ECDC4',
            '8B': '#45B7D1',
            '14B': '#96CEB4',
            '32B': '#FECA57',
            '1B': '#FF9FF3',
            '3B': '#A55EEA',
            '13B': '#C44569',
            'unknown': '#95A5A6'
        }

        model_quant_combos = df['model_quant_combo'].unique().sort().to_list()

        print(f"Creating throughput vs device plots for {len(model_quant_combos)} model + quantization combinations...")

        for combo in model_quant_combos:
            combo_data = df.filter(pl.col('model_quant_combo') == combo)

            if combo_data.is_empty():
                continue

            fig = go.Figure()

            model_sizes = combo_data['model_size'].unique().sort().to_list()

            for model_size in model_sizes:
                size_data = combo_data.filter(pl.col('model_size') == model_size)

                if size_data.is_empty():
                    continue

                grouped = size_data.group_by('device_type').agg([
                    pl.col('throughput').mean().alias('avg_throughput'),
                    pl.col('throughput').count().alias('count')
                ])

                if grouped.is_empty():
                    continue

                grouped_pd = grouped.to_pandas()

                grouped_pd['sort_key'] = grouped_pd['device_type'].apply(get_device_sort_key)
                grouped_pd = grouped_pd.sort_values('sort_key')

                color = model_size_colors.get(model_size, '#95A5A6')

                fig.add_trace(go.Scatter(
                    x=grouped_pd['device_type'].tolist(),
                    y=grouped_pd['avg_throughput'].tolist(),
                    mode='lines+markers',
                    name=model_size,
                    line=dict(width=3, color=color),
                    marker=dict(size=10, color=color, opacity=0.8),
                    text=[f"Model Size: {model_size}<br>Device: {device}<br>Avg Throughput: {throughput:.2f}<br>Count: {count}"
                          for device, throughput, count in zip(
                            grouped_pd['device_type'].tolist(),
                            grouped_pd['avg_throughput'].tolist(),
                            grouped_pd['count'].tolist()
                        )],
                    hovertemplate='<b>%{text}</b><extra></extra>'
                ))

            fig.update_layout(
                title=f'Throughput vs Device Type - {combo.upper()}',
                xaxis_title='Device Type',
                yaxis_title='Average Throughput (tokens/second)',
                xaxis=dict(tickangle=45),
                height=600,
                legend=dict(
                    title="Model Size",
                    orientation="v",
                    yanchor="top",
                    y=1,
                    xanchor="left",
                    x=1.02
                )
            )

            safe_combo_name = re.sub(r'[^\w\-_\.]', '_', combo)
            filename = f'throughput_vs_device_{safe_combo_name}_line.html'
            fig.write_html(os.path.join(output_dir, filename))
            print(f"Saved: {filename}")

        fig_summary = go.Figure()

        for combo in model_quant_combos:
            combo_data = df.filter(pl.col('model_quant_combo') == combo)

            if combo_data.is_empty():
                continue

            grouped = combo_data.group_by('device_type').agg([
                pl.col('throughput').mean().alias('avg_throughput'),
                pl.col('model_size').first().alias('model_size')
            ])

            if grouped.is_empty():
                continue

            grouped_pd = grouped.to_pandas()
            grouped_pd['sort_key'] = grouped_pd['device_type'].apply(get_device_sort_key)
            grouped_pd = grouped_pd.sort_values('sort_key')

            model_size = grouped_pd['model_size'].iloc[0]
            color = model_size_colors.get(model_size, '#95A5A6')

            fig_summary.add_trace(go.Scatter(
                x=grouped_pd['device_type'].tolist(),
                y=grouped_pd['avg_throughput'].tolist(),
                mode='lines+markers',
                name=combo,
                line=dict(width=2, color=color),
                marker=dict(size=6, color=color, opacity=0.8)
            ))

        fig_summary.update_layout(
            title='Throughput vs Device Type - All Model + Quantization Combinations',
            xaxis_title='Device Type',
            yaxis_title='Average Throughput (tokens/second)',
            xaxis=dict(tickangle=45),
            height=600,
            legend=dict(
                title="Model + Quantization",
                orientation="v",
                yanchor="top",
                y=1,
                xanchor="left",
                x=1.02
            )
        )

        fig_summary.write_html(os.path.join(output_dir, 'throughput_vs_device_summary_line.html'))
        print("Saved: throughput_vs_device_summary_line.html")

    def plot_throughput_by_batch_size(self, df: pl.DataFrame, output_dir: str) -> None:
        if df.is_empty():
            return

        df = df.with_columns([
            pl.when(pl.col('gpu_percentage') > 0)
            .then(pl.concat_str([pl.lit('GPU_'), pl.col('gpu_percentage').cast(pl.Utf8), pl.lit('%_CPU_'), pl.col('cpu_cores').cast(pl.Utf8)]))
            .otherwise(pl.concat_str([pl.lit('CPU_'), pl.col('cpu_cores').cast(pl.Utf8), pl.lit('_cores')]))
            .alias('device_config_label')
        ])

        device_config_colors = {
            'CPU_16_cores': '#FF6B6B',
            'GPU_100%_CPU_0': '#4ECDC4',
            'GPU_50%_CPU_8': '#45B7D1',
            'GPU_75%_CPU_4': '#96CEB4',
            'CPU_8_cores': '#FECA57',
            'CPU_4_cores': '#FF9FF3',
            'GPU_25%_CPU_12': '#A55EEA',
        }

        models = df['model_type'].unique().sort().to_list()

        print(f"Creating throughput vs batch size plots for {len(models)} models...")

        for model in models:
            model_data = df.filter(pl.col('model_type') == model)

            if model_data.is_empty():
                continue

            fig = go.Figure()

            device_configs = model_data['device_config_label'].unique().sort().to_list()

            for device_config in device_configs:
                device_data = model_data.filter(pl.col('device_config_label') == device_config)

                if device_data.is_empty():
                    continue

                grouped = device_data.group_by('batch_size').agg([
                    pl.col('throughput').mean().alias('avg_throughput'),
                    pl.col('throughput').count().alias('count'),
                    pl.col('quantization').first().alias('quantization'),
                    pl.col('model_size').first().alias('model_size')
                ]).sort('batch_size')

                if grouped.is_empty():
                    continue

                color = device_config_colors.get(device_config, '#95A5A6')

                fig.add_trace(go.Scatter(
                    x=grouped['batch_size'].to_list(),
                    y=grouped['avg_throughput'].to_list(),
                    mode='lines+markers',
                    name=device_config,
                    line=dict(width=3, color=color),
                    marker=dict(size=10, color=color, opacity=0.8),
                    text=[f"Device: {device_config}<br>Batch Size: {batch}<br>Avg Throughput: {throughput:.2f}<br>Quantization: {quant}<br>Model Size: {size}<br>Count: {count}"
                          for batch, throughput, quant, size, count in zip(
                            grouped['batch_size'].to_list(),
                            grouped['avg_throughput'].to_list(),
                            grouped['quantization'].to_list(),
                            grouped['model_size'].to_list(),
                            grouped['count'].to_list()
                        )],
                    hovertemplate='<b>%{text}</b><extra></extra>'
                ))

            model_info = model_data.select(['model_size', 'quantization']).unique().to_pandas().iloc[0]
            model_title = f"{model_info['model_size']} - {model_info['quantization']}"

            fig.update_layout(
                title=f'Throughput vs Batch Size - {model_title}<br>Model: {model}',
                xaxis_title='Batch Size',
                yaxis_title='Average Throughput (tokens/second)',
                height=600,
                legend=dict(
                    title="Device Configuration",
                    orientation="v",
                    yanchor="top",
                    y=1,
                    xanchor="left",
                    x=1.02
                )
            )

            safe_model_name = re.sub(r'[^\w\-_\.]', '_', model)
            filename = f'throughput_vs_batch_size_{safe_model_name}_line.html'
            fig.write_html(os.path.join(output_dir, filename))
            print(f"Saved: {filename}")

        fig_summary = go.Figure()

        for model in models:
            model_data = df.filter(pl.col('model_type') == model)

            if model_data.is_empty():
                continue

            grouped = model_data.group_by('batch_size').agg([
                pl.col('throughput').mean().alias('avg_throughput'),
                pl.col('model_size').first().alias('model_size'),
                pl.col('quantization').first().alias('quantization')
            ]).sort('batch_size')

            if grouped.is_empty():
                continue

            model_info = model_data.select(['model_size']).unique().to_pandas().iloc[0]
            model_size = model_info['model_size']

            size_colors = {
                '1.5B': '#FF6B6B', '1.7B': '#FF8E8E', '7B': '#4ECDC4',
                '8B': '#45B7D1', '14B': '#96CEB4', '32B': '#FECA57'
            }
            color = size_colors.get(model_size, '#95A5A6')

            fig_summary.add_trace(go.Scatter(
                x=grouped['batch_size'].to_list(),
                y=grouped['avg_throughput'].to_list(),
                mode='lines+markers',
                name=f"{model_size} - {grouped['quantization'].to_list()[0]}",
                line=dict(width=3, color=color),
                marker=dict(size=8, color=color, opacity=0.8)
            ))

        fig_summary.update_layout(
            title='Throughput vs Batch Size - All Models Summary',
            xaxis_title='Batch Size',
            yaxis_title='Average Throughput (tokens/second)',
            height=600,
            legend=dict(
                title="Model (Size - Quantization)",
                orientation="v",
                yanchor="top",
                y=1,
                xanchor="left",
                x=1.02
            )
        )

        fig_summary.write_html(os.path.join(output_dir, 'throughput_vs_batch_size_summary_line.html'))
        print("Saved: throughput_vs_batch_size_summary_line.html")

    def plot_throughput_vs_batch_size_by_prompt_length(self, df: pl.DataFrame, output_dir: str) -> None:
        if df.is_empty():
            return

        models = df['model_type'].unique().sort().to_list()

        prompt_lengths = df['prompt_length'].unique().sort().to_list()
        colors = px.colors.qualitative.Set3[:len(prompt_lengths)]
        prompt_color_map = {length: colors[i % len(colors)] for i, length in enumerate(prompt_lengths)}

        for model in models:
            model_data = df.filter(pl.col('model_type') == model)

            if model_data.is_empty():
                continue

            fig = go.Figure()

            for prompt_length in prompt_lengths:
                prompt_data = model_data.filter(pl.col('prompt_length') == prompt_length)

                if prompt_data.is_empty():
                    continue

                grouped = prompt_data.group_by('batch_size').agg([
                    pl.col('throughput').mean().alias('avg_throughput'),
                    pl.col('throughput').std().alias('std_throughput'),
                    pl.col('throughput').count().alias('count')
                ]).sort('batch_size')

                if grouped.is_empty():
                    continue

                grouped_pd = grouped.to_pandas()

                if prompt_length == 0:
                    prompt_label = "Empty prompt"
                elif prompt_length < 100:
                    prompt_label = f"Short prompt ({prompt_length} chars)"
                elif prompt_length < 1000:
                    prompt_label = f"Medium prompt ({prompt_length} chars)"
                else:
                    prompt_label = f"Long prompt ({prompt_length} chars)"

                color = prompt_color_map[prompt_length]

                fig.add_trace(go.Scatter(
                    x=grouped_pd['batch_size'],
                    y=grouped_pd['avg_throughput'],
                    mode='lines+markers',
                    name=prompt_label,
                    line=dict(width=3, color=color),
                    marker=dict(size=8, color=color, opacity=0.8),
                    error_y=dict(
                        type='data',
                        array=grouped_pd['std_throughput'].fillna(0),
                        visible=True,
                        color=color,
                        thickness=1.5,
                        width=3
                    )
                ))

            model_display_name = model.replace('.gguf', '').replace('_', ' ')
            fig.update_layout(
                title=f'Throughput vs Batch Size by Prompt Length - {model_display_name}',
                xaxis_title='Batch Size (Concurrent Requests)',
                yaxis_title='Average Throughput (tokens/second)',
                height=600,
                legend=dict(
                    title="Content Prompt Length",
                    orientation="v",
                    yanchor="top",
                    y=1,
                    xanchor="left",
                    x=1.02
                ),
                xaxis=dict(
                    type='linear',
                    tickmode='linear',
                    tick0=1,
                    dtick=1
                ),
                yaxis=dict(
                    type='linear'
                )
            )

            safe_model_name = re.sub(r'[^\w\-_\.]', '_', model)
            filename = f'throughput_vs_batch_size_by_prompt_length_{safe_model_name}_line.html'
            fig.write_html(os.path.join(output_dir, filename))
            print(f"Saved: {filename}")

        fig_summary = go.Figure()

        line_styles = ['solid', 'dash', 'dot', 'dashdot']

        for i, model in enumerate(models):
            model_data = df.filter(pl.col('model_type') == model)

            if model_data.is_empty():
                continue

            line_style = line_styles[i % len(line_styles)]
            model_display_name = model.replace('.gguf', '').replace('_', ' ')

            for prompt_length in prompt_lengths:
                prompt_data = model_data.filter(pl.col('prompt_length') == prompt_length)

                if prompt_data.is_empty():
                    continue

                grouped = prompt_data.group_by('batch_size').agg([
                    pl.col('throughput').mean().alias('avg_throughput')
                ]).sort('batch_size')

                if grouped.is_empty():
                    continue

                grouped_pd = grouped.to_pandas()

                if prompt_length == 0:
                    prompt_label = "Empty"
                elif prompt_length < 100:
                    prompt_label = "Short"
                elif prompt_length < 1000:
                    prompt_label = "Medium"
                else:
                    prompt_label = "Long"

                color = prompt_color_map[prompt_length]

                fig_summary.add_trace(go.Scatter(
                    x=grouped_pd['batch_size'],
                    y=grouped_pd['avg_throughput'],
                    mode='lines+markers',
                    name=f"{model_display_name} - {prompt_label} prompt",
                    line=dict(width=2, color=color, dash=line_style),
                    marker=dict(size=6, color=color, opacity=0.8)
                ))

        fig_summary.update_layout(
            title='Throughput vs Batch Size by Prompt Length - All Models Summary',
            xaxis_title='Batch Size (Concurrent Requests)',
            yaxis_title='Average Throughput (tokens/second)',
            height=700,
            legend=dict(
                title="Model - Prompt Length",
                orientation="v",
                yanchor="top",
                y=1,
                xanchor="left",
                x=1.02
            ),
            xaxis=dict(
                type='linear',
                tickmode='linear',
                tick0=1,
                dtick=1
            ),
            yaxis=dict(
                type='linear'
            )
        )

        fig_summary.write_html(os.path.join(output_dir, 'throughput_vs_batch_size_by_prompt_length_summary_line.html'))
        print("Saved: throughput_vs_batch_size_by_prompt_length_summary_line.html")

    def run_analysis(self, output_dir: str) -> None:
        print("Starting benchmark log analysis with line plots...")

        self.load_all_logs()

        df = self.create_polars_dataframe()

        if df.is_empty():
            print("No data found in logs!")
            return

        print(f"Data loaded: {df.height} benchmark results")

        model_names = df['model_filename'].unique().sort().to_list()
        print(f"\nFound {len(model_names)} unique models:")
        for i, model in enumerate(model_names, 1):
            print(f"  {i}. {model}")
        print()

        df.write_csv(os.path.join(output_dir, 'benchmark_data_line.csv'))
        print("Raw data saved to benchmark_data_line.csv")

        json_data = df.to_dicts()
        json_output_path = os.path.join(output_dir, 'benchmark_data_line.json')
        with open(json_output_path, 'w') as f:
            json.dump(json_data, f, indent=2)
        print("Raw data saved to benchmark_data_line.json")

        raw_data_output_path = os.path.join(output_dir, 'raw_stats_result_data.json')
        with open(raw_data_output_path, 'w') as f:
            json.dump(self.raw_data, f, indent=2)
        print("Raw stats_result JSON data saved to raw_stats_result_data.json")

        print("Generating line plots...")
        self.plot_throughput_by_quantization(df, output_dir)
        self.plot_throughput_by_model_size(df, output_dir)
        self.plot_throughput_by_device(df, output_dir)
        self.plot_throughput_by_batch_size(df, output_dir)
        self.plot_throughput_vs_batch_size_by_prompt_length(df, output_dir)
        print("All line plots saved!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze benchmark log files and generate line plots')
    parser.add_argument('--log_file', '-l', 
                       help='Path to the raw log file to analyze')
    parser.add_argument('--json_file', '-j',
                       help='Path to the parsed JSON result file to analyze')
    parser.add_argument('--output-dir', '-o', default='./benchmarks',
                       help='Output directory for generated HTML files (default: ./benchmarks)')

    args = parser.parse_args()
    
    if not args.log_file and not args.json_file:
        parser.error("Either --log_file or --json_file must be specified")
    
    if args.log_file and args.json_file:
        parser.error("Only one of --log_file or --json_file can be specified")

    analyzer = BenchmarkLogAnalyzer(log_file=args.log_file, json_file=args.json_file)
    analyzer.run_analysis(args.output_dir)
