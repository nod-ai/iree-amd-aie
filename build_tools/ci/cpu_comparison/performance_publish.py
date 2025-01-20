#!/usr/bin/env python3

# Copyright 2025 The IREE Authors

import json
import sys


def append_history(results_json_path: str, results_history_path: str):
    # Get the results for the current run.
    with open(results_json_path, "r") as f:
        results = json.load(f)

    # Append the results to the history.
    results_history = []
    max_history = 100
    with open(results_history_path, "r+") as f:
        results_history = json.load(f)
        results_history.append(results)
        # Keep only the most recent results.
        if len(results_history) > max_history:
            results_history = results_history[-max_history:]
        f.seek(0)
        # Write the updated history back to the file.
        json.dump(results_history, f, indent=2)


def generate_html(results_history_path: str, results_html_path: str):
    results_history = []
    with open(results_history_path, "r") as f:
        results_history = json.load(f)

    # Reformat the data for the graph.
    graph_data = {}
    time_unit = "us"
    for entry in results_history:
        commit_hash = entry["commit_hash"]
        # Truncate commit hash to first 7 characters
        truncated_commit_hash = str(commit_hash)[:7]
        for test in entry["tests"]:
            test_name = test["name"]
            duration = test["time_mean"]
            assert test["time_mean_unit"] == time_unit
            if test_name not in graph_data:
                graph_data[test_name] = {"commit_hashes": [], "durations": []}
            graph_data[test_name]["commit_hashes"].append(truncated_commit_hash)
            graph_data[test_name]["durations"].append(duration)

    # Start building the HTML content
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Performance History</title>
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <style>
            body { font-family: Arial, sans-serif; line-height: 1.6; padding: 20px; }
            .chart-container { width: 80%; margin: 30px auto; }
            canvas { width: 100%; height: auto; }
        </style>
    </head>
    <body>
        <h1>Performance History</h1>
        <p>Overview of test results for the most recent commits:</p>
    """

    # Add a graph for each test
    for test_name, data in graph_data.items():
        html_content += f"""
        <div class="chart-container">
            <h2>Performance for {test_name}</h2>
            <canvas id="chart-{test_name.replace(' ', '-')}"></canvas>
        </div>
        <script>
            const ctx_{test_name.replace(' ', '_')} = document.getElementById('chart-{test_name.replace(' ', '-')}')
            const chart_{test_name.replace(' ', '_')} = new Chart(ctx_{test_name.replace(' ', '_')}, {{
                type: 'line',
                data: {{
                    labels: {data["commit_hashes"]},  // Truncated commit hashes as X-axis labels
                    datasets: [{{
                        label: 'Duration ({time_unit})',
                        data: {data["durations"]},   // Test durations as Y-axis
                        borderColor: 'rgba(75, 192, 192, 1)',
                        backgroundColor: 'rgba(75, 192, 192, 0.2)',
                        borderWidth: 2
                    }}]
                }},
                options: {{
                    responsive: true,
                    scales: {{
                        x: {{
                            title: {{
                                display: true,
                                text: 'Commit Hash'
                            }}
                        }},
                        y: {{
                            title: {{
                                display: true,
                                text: 'Duration ({time_unit})'
                            }}
                        }}
                    }}
                }}
            }});
        </script>
        """

    # Close the HTML content
    html_content += """
    </body>
    </html>
    """

    # Save the HTML file
    with open(results_html_path, "w") as f:
        f.write(html_content)


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print(
            "Usage: python3 performance_publish.py <path_to_results_json> <path_to_results_history> <path_to_results_html>\n"
            "This script reads the performance results from the specified JSON file, appends them to the history file, and generates an HTML visualization.\n"
        )
        sys.exit(1)

    results_json_path = sys.argv[1]
    results_history_path = sys.argv[2]
    results_html_path = sys.argv[3]

    append_history(results_json_path, results_history_path)
    generate_html(results_history_path, results_html_path)
