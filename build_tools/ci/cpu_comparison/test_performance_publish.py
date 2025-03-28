from .performance_publish import *
import lit


def test_common_x_axis():
    """
    Two commits. Two tests. One test appears on both commits, the other only on one.
    The html generated should have plots for both tests which is the union of all
    commits.
    """

    results_history = [
        {
            "commit_hash": "3a6aa4af21f807331a065edf80c5e0a7677f5764",
            "cpu": "AMD Ryzen 7 8845HS w/ Radeon 780M Graphics",
            "tests": [
                {
                    "name": "matmul_512_512_4096_bf16_f32_O2_npu1_4col_benchmark",
                    "time_mean": "2660",
                    "time_mean_unit": "us",
                },
                {
                    "name": "matmul_512_512_4096_bf16_f32_O2_npu1_4col_outline_benchmark",
                    "time_mean": "6529",
                    "time_mean_unit": "us",
                },
            ],
        },
        {
            "commit_hash": "a6669378498cd131c7f233dabdb65fda1004449c",
            "cpu": "AMD Ryzen 7 8845HS w/ Radeon 780M Graphics",
            "tests": [
                {
                    "name": "matmul_512_512_4096_bf16_f32_O2_npu1_4col_benchmark",
                    "time_mean": "2680",
                    "time_mean_unit": "us",
                },
            ],
        },
    ]
    html = generate_html(results_history)
    assert html.count("labels: ['3a6aa4a', 'a666937']") == 2


def test_ops_and_note():
    results_history = [
        {
            "commit_hash": "20082027ded7c14c33e447748e2695f14d746ac4",
            "cpu": "AMD Ryzen 7 8845HS w/ Radeon 780M Graphics",
            "tests": [
                {
                    "name": "matmul_512_512_512_bf16_f32_O2_npu1_4col_callrepl_100_outline_benchmark",
                    "time_mean": "272526",
                    "time_mean_unit": "us",
                }
            ],
        }
    ]

    html = generate_html(results_history)
    expected_successive_lines = [
        "<h2>Performance for matmul_512_512_512_bf16_f32_O2_npu1_4col_callrepl_100_outline_benchmark</h2>",
        "Total ops: 13421772800",
    ]
    for line in expected_successive_lines:
        assert line in html
