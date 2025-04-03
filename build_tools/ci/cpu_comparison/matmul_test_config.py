# Copyright 2025 The IREE Authors

# # NPU1_4COL matmul test(s):
npu1_4col_matmul_tests = [
    # 1x1 core tests.
    {
        "M": 128,
        "N": 128,
        "K": 128,
        "input_type": "i8",
        "acc_type": "i32",
        "name_suffix": "OneCore_npu1_4col",
        "additional_labels": ["OneCore"],
        "aie_compilation_flags": [
            "--iree-amdaie-num-rows=1",
            "--iree-amdaie-num-cols=1",
        ],
    },
    # 2x2 core tests.
    {
        "M": 32,
        "N": 32,
        "K": 32,
        "input_type": "bf16",
        "acc_type": "f32",
        "name_suffix": "2rows_2cols",
        "aie_compilation_flags": [
            "--iree-amdaie-num-rows=2",
            "--iree-amdaie-num-cols=2",
        ],
    },
    # 4x2 core tests.
    {
        "M": 32,
        "N": 32,
        "K": 32,
        "input_type": "bf16",
        "acc_type": "f32",
        "name_suffix": "4rows_2cols",
        "aie_compilation_flags": [
            "--iree-amdaie-num-rows=4",
            "--iree-amdaie-num-cols=2",
        ],
    },
    # 4x4 core tests.
    {
        "M": 32,
        "N": 32,
        "K": 32,
        "input_type": "i32",
        "acc_type": "i32",
        "name_suffix": "infinite_loop_npu1_4col",
        "aie_compilation_flags": [
            "--iree-amdaie-enable-infinite-loop-around-core-block=true"
        ],
    },
]
# NPU4 matmul test(s):
npu4_matmul_tests = [
    # 1x1 core tests.
    {
        "M": 32,
        "N": 32,
        "K": 128,
        "input_type": "i32",
        "acc_type": "i32",
        "tile_pipeline": "pack-peel-4-level-tiling",
        "name_suffix": "OneCore_npu4",
        "additional_labels": ["OneCore"],
        "aie_compilation_flags": [
            "--iree-amdaie-num-rows=1",
            "--iree-amdaie-num-cols=1",
        ],
    },
    {
        "M": 32,
        "N": 32,
        "K": 256,
        "input_type": "i32",
        "acc_type": "i32",
        "tile_pipeline": "pack-peel-4-level-tiling",
        "name_suffix": "OneCore_npu4",
        "additional_labels": ["OneCore"],
        "aie_compilation_flags": [
            "--iree-amdaie-num-rows=1",
            "--iree-amdaie-num-cols=1",
        ],
    },
    {
        "M": 64,
        "N": 128,
        "K": 128,
        "input_type": "i32",
        "acc_type": "i32",
        "tile_pipeline": "pack-peel-4-level-tiling",
        "name_suffix": "OneCore_npu4",
        "additional_labels": ["OneCore"],
        "aie_compilation_flags": [
            "--iree-amdaie-num-rows=1",
            "--iree-amdaie-num-cols=1",
        ],
    },
    {
        "M": 128,
        "N": 128,
        "K": 128,
        "input_type": "i8",
        "acc_type": "i32",
        "name_suffix": "OneCore_npu4",
        "additional_labels": ["OneCore"],
        "aie_compilation_flags": [
            "--iree-amdaie-num-rows=1",
            "--iree-amdaie-num-cols=1",
        ],
    },
    # 4x2 core tests.
    {
        "M": 32,
        "N": 128,
        "K": 128,
        "input_type": "i32",
        "acc_type": "i32",
        "tile_pipeline": "pack-peel-4-level-tiling",
        "aie_compilation_flags": [
            "--iree-amdaie-num-rows=4",
            "--iree-amdaie-num-cols=2",
        ],
    },
    {
        "M": 256,
        "N": 32,
        "K": 32,
        "input_type": "i32",
        "acc_type": "i32",
        "tile_pipeline": "pack-peel-4-level-tiling",
        "aie_compilation_flags": [
            "--iree-amdaie-num-rows=4",
            "--iree-amdaie-num-cols=2",
        ],
    },
    # 4x8 core tests.
    {"M": 32, "N": 32, "K": 32, "input_type": "i32", "acc_type": "i32"},
    {
        "M": 32,
        "N": 32,
        "K": 32,
        "input_type": "i32",
        "acc_type": "i32",
        "use_chess": True,
    },
    {
        "M": 32,
        "N": 32,
        "K": 32,
        "input_type": "i32",
        "acc_type": "i32",
        "name_suffix": "infinite_loop_npu4",
        "aie_compilation_flags": [
            "--iree-amdaie-enable-infinite-loop-around-core-block=true"
        ],
    },
    {
        "M": 64,
        "N": 64,
        "K": 64,
        "input_type": "bf16",
        "acc_type": "f32",
        "use_ukernel": True,
        "use_chess_for_ukernel": False,
    },
    {
        "M": 64,
        "N": 64,
        "K": 64,
        "input_type": "bf16",
        "acc_type": "f32",
        "use_ukernel": True,
        "use_chess_for_ukernel": False,
        "tile_pipeline": "pack-peel-4-level-tiling",
        "name_suffix": "4rows_8cols_npu4",
    },
    {
        "M": 512,
        "N": 512,
        "K": 512,
        "input_type": "bf16",
        "acc_type": "f32",
        "use_ukernel": True,
        "use_chess_for_ukernel": False,
        "tile_pipeline": "pack-peel-4-level-tiling",
        "name_suffix": "4rows_8cols_npu4",
    },
    {
        "M": 512,
        "N": 512,
        "K": 512,
        "input_type": "i8",
        "acc_type": "i32",
        "use_ukernel": True,
        "use_chess_for_ukernel": False,
        "tile_pipeline": "pack-peel-4-level-tiling",
        "additional_labels": ["I8UKernel"],
    },
    {
        "M": 512,
        "N": 512,
        "K": 256,
        "input_type": "i32",
        "acc_type": "i32",
        "tile_pipeline": "pack-peel-4-level-tiling",
        "name_suffix": "4rows_8cols_npu4_pack_peel_4_level_tiling",
    },
    {
        "M": 1024,
        "N": 1024,
        "K": 1024,
        "input_type": "i32",
        "acc_type": "i32",
        "name_suffix": "4rows_8cols_npu4",
    },
]
matmul_tests_for_each_device = {
    "npu1_4col": npu1_4col_matmul_tests,
    "npu4": npu4_matmul_tests,
}
