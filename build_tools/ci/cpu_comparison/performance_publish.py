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


def get_total_ops(name):
    """
    where name is a string,
    1) split it on "_"
    2) return the product of all values in the list that are integers.
    """
    fragments = name.split("_")
    if len(fragments) == 0:
        return None
    if fragments[0] != "matmul":
        return None
    if "empty" in fragments:
        return 0
    total_ops = 1
    for fragment in fragments:
        if fragment.isdigit():
            total_ops *= int(fragment)
    return total_ops


def get_canonical_name(name):
    """
    Test names might change with commits, even though the test is unchanged.
    In this case, we canonicalize names to the original names.
    """
    # replace callrepl_0_outline with outline_empty:
    name = name.replace("callrepl_0_outline", "outline_empty")
    name = name.replace("chess_benchmark", "benchmark")
    name = name.replace("matmul4d_16_128_8", "matmul4d_512_4096_512")
    return name


def generate_html(results_history_path: str, results_html_path: str):
    results_history = []
    with open(results_history_path, "r") as f:
        results_history = json.load(f)

    graph_data = {}
    for entry in results_history:
        for test in entry["tests"]:
            name = get_canonical_name(test["name"])
            graph_data[name] = {
                "commit_hashes": [],
                "durations": [],
                "ops": get_total_ops(name),
                "note": None,
            }

            if "callrepl_100" in name:
                graph_data[name]["note"] = "This test is run on a single AIE core."

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
        if data["ops"] is not None:
            html_content += f"""
            Total ops: {data["ops"]}"""
        if data["note"] is not None:
            html_content += f"""
            Note: {data["note"]}"""
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
    generate_html(results_history_path, results_html_path)
