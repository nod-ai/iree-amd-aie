#!/usr/bin/env python3

# Copyright 2025 The IREE Authors

import json
import sys
import functools


def append_history(results_json_path: str, results_history_path: str):
    # Get the results for the current run.
    with open(results_json_path, "r") as f:
        results = json.load(f)

    # Append the results to the history.
    results_history = []
    max_history = 100
    with open(results_history_path, "r") as f:
        results_history = json.load(f)
        results_history.append(results)
        # Keep only the most recent results.
        if len(results_history) > max_history:
            results_history = results_history[-max_history:]

    # Write the updated history back to the file.
    with open(results_history_path, "w") as f:
        json.dump(results_history, f, indent=2)


def get_canonical_name(name):
    """
    Test names might change with commits, even though the test is unchanged.
    In this case, we canonicalize names to the original names.
    """
    # replace callrepl_0_outline with outline_empty:
    name = name.replace("callrepl_0_outline", "outline_empty")
    name = name.replace("ctrlpkt_benchmark", "benchmark")
    name = name.replace("chess_benchmark", "benchmark")
    name = name.replace("matmul4d_16_128_8", "matmul4d_512_4096_512")
    return name


def generate_html(results_history: list):
    graph_data = {}
    for entry in results_history:
        for test in entry["tests"]:
            name = get_canonical_name(test["name"])
            if name not in graph_data:
                graph_data[name] = {
                    "commit_hashes": [],
                    "durations": [],
                    "n_cores": None,
                    "total_ops": None,
                }
            if "n_cols" in test and "n_rows" in test:
                n_cores = int(test["n_rows"]) * int(test["n_cols"])
                graph_data[name]["n_cores"] = n_cores
            if "total_ops" in test:
                graph_data[name]["total_ops"] = int(test["total_ops"])

    time_unit = "us"
    for entry in results_history:
        commit_hash = str(entry["commit_hash"])[:7]
        local_tests = dict.fromkeys(graph_data.keys(), None)
        for test in entry["tests"]:
            local_tests[get_canonical_name(test["name"])] = test
        for test_name, test in local_tests.items():
            graph_data[test_name]["commit_hashes"].append(commit_hash)
            # So that the time/commit horizontal/x axis is consistent
            # across tests, even if a test is missing for a commit,
            # we add duration=0 if a test did not run.
            if not test:
                graph_data[test_name]["durations"].append(0)
            else:
                duration = test["time_mean"]
                assert test["time_mean_unit"] == time_unit
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
            <h2>Performance for {test_name}</h2>"""
        if data["total_ops"] is not None:
            html_content += f"""
            Total ops: {data["total_ops"]}"""
        if data["n_cores"] is not None:
            html_content += f"""
            Number of cores: {data["n_cores"]}"""
        html_content += f"""
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
                            beginAtZero: true,  // Ensures the Y-axis starts at 0
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

    return html_content


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

    # Generate and save the HTML file
    results_history = []
    with open(results_history_path, "r") as f:
        results_history = json.load(f)
    html_content = generate_html(results_history)
    with open(results_html_path, "w") as f:
        f.write(html_content)
