#!/usr/bin/env python3

import os
import json
import argparse
from typing import List, Dict

class LogParser:
    def __init__(self):
        pass

    def extract_raw_stats_results(self, filepath: str) -> List[Dict]:
        raw_results = []

        try:
            with open(filepath, 'r') as f:
                content = f.read()

            lines = content.split('\n')
            print(f"Processing {len(lines)} lines from {filepath}")

            for line in lines:
                line = line.strip()

                if "stats_result" in line:
                    try:
                        json_str = line[len('stats_result: '):]

                        import ast
                        data = ast.literal_eval(json_str)

                        raw_results.append(data)

                    except (ValueError, SyntaxError) as e:
                        print(f"Error parsing stats_result line: {e}")
                        continue

        except Exception as e:
            print(f"Error reading {filepath}: {e}")

        return raw_results

    def parse_and_save(self, input_file: str, output_file: str = None) -> None:
        print(f"Extracting raw stats_result data from: {input_file}")
        raw_results = self.extract_raw_stats_results(input_file)

        if not raw_results:
            print("No stats_result data found in the log file!")
            return

        print(f"Extracted {len(raw_results)} raw stats_result entries")

        if not output_file:
            base_name = os.path.splitext(os.path.basename(input_file))[0]
            output_file = f"{base_name}_raw_stats_result_data.json"

        with open(output_file, 'w') as f:
            json.dump(raw_results, f, indent=2)

        print(f"Raw stats_result data saved to: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract raw stats_result data from benchmark log files')
    parser.add_argument('--input', '-i', required=True,
                       help='Path to the raw log file to parse')
    parser.add_argument('--output', '-o',
                       help='Path to save the raw JSON file (default: <input>_raw_stats_result_data.json)')

    args = parser.parse_args()

    log_parser = LogParser()
    log_parser.parse_and_save(args.input, args.output)
