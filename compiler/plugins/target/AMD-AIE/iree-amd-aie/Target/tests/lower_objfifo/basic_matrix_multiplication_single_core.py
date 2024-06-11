from aie.dialects import arith, linalg
from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.dialects.scf import *
from aie.extras.context import mlir_mod_ctx


def my_matmul(M, K, N):
    m = 32
    k = 32
    n = 32
    r = 4
    s = 8
    t = 4
    word_size_in = 4
    word_size_out = 4

    A_sz_in_i32s = M * K * word_size_in // 4
    B_sz_in_i32s = K * N * word_size_in // 4
    C_sz_in_bytes = M * N * word_size_out
    C_sz_in_i32s = C_sz_in_bytes // 4

    M_div_m = M // m
    K_div_k = K // k
    N_div_n = N // n
    tiles = M_div_m * N_div_n

    # Matrix A: MxK, submatrices a: mxk
    k_in_i32s = k * word_size_in // 4
    K_in_i32s = K * word_size_in // 4

    # Matrix B: KxN, submatrices b: kxn
    n_in_i32s = n * word_size_in // 4
    N_in_i32s = N * word_size_in // 4
    k_x_N_in_i32s = k * N * word_size_in // 4

    # Output Matrix C: MxN
    n_in_i32s_out = n * word_size_out // 4
    N_in_i32s_out = N * word_size_out // 4
    m_x_N_in_i32s_out = m * N * word_size_out // 4

    @device(AIEDevice.npu1_1col)
    def device_body():
        memref_a_ty = T.memref(m, k, T.f32())
        memref_b_ty = T.memref(k, n, T.f32())
        memref_c_ty = T.memref(m, n, T.f32())

        # Tile declarations
        shim_tile = tile(0, 0)
        mem_tile = tile(0, 1)
        compute_tile2_col, compute_tile2_row = 0, 2
        compute_tile2 = tile(compute_tile2_col, compute_tile2_row)

        # AIE-array data movement with object fifos
        # Input A
        inA = object_fifo("inA", shim_tile, mem_tile, 2, memref_a_ty)
        memA = object_fifo(
            "memA",
            mem_tile,
            compute_tile2,
            2,
            memref_a_ty,
            [
                (m // r, r * k),
                (k // s, s),
                (r, k),
                (s, 1),
            ],
        )
        object_fifo_link(inA, memA)

        # Input B
        inB = object_fifo("inB", shim_tile, mem_tile, 2, memref_b_ty)
        memB = object_fifo(
            "memB",
            mem_tile,
            compute_tile2,
            2,
            memref_b_ty,
            [
                (k // s, s * n),
                (n // t, t),
                (s, n),
                (t, 1),
            ],
        )
        object_fifo_link(inB, memB)

        # Output C
        memC = object_fifo("memC", compute_tile2, mem_tile, 2, memref_c_ty)
        outC = object_fifo(
            "outC",
            mem_tile,
            shim_tile,
            2,
            memref_c_ty,
            [
                (m // r, r * n),
                (r, t),
                (n // t, r * t),
                (t, 1),
            ],
        )
        object_fifo_link(memC, outC)

        # Compute tile 2
        @core(compute_tile2)
        def core_body():
            for _ in for_(0xFFFFFFFF):
                for _ in for_(tiles):
                    elem_out = memC.acquire(ObjectFifoPort.Produce, 1)
                    cf0 = arith.constant(T.f32(), 0.0)
                    linalg.fill(cf0, outs=[elem_out])
                    for _ in for_(K_div_k):
                        elem_in_a = memA.acquire(ObjectFifoPort.Consume, 1)
                        elem_in_b = memB.acquire(ObjectFifoPort.Consume, 1)
                        linalg.matmul(elem_in_a, elem_in_b, outs=[elem_out])
                        memA.release(ObjectFifoPort.Consume, 1)
                        memB.release(ObjectFifoPort.Consume, 1)
                        yield_([])

                    memC.release(ObjectFifoPort.Produce, 1)
                    yield_([])
                yield_([])

        # To/from AIE-array data movement

        @FuncOp.from_py_func(
            T.memref(A_sz_in_i32s, T.i32()),
            T.memref(B_sz_in_i32s, T.i32()),
            T.memref(C_sz_in_i32s, T.i32()),
        )
        def sequence(A, B, C):
            # only do 5 tile rows at a time before synchronizing, so we can reuse BDs
            rows_per_block = 5
            for tile_row_block in range(
                (M_div_m + rows_per_block - 1) // rows_per_block
            ):
                C_row_offset_in_i32s = (
                    tile_row_block * rows_per_block * m * N * word_size_out // 4
                )
                num_tile_rows = min(
                    [rows_per_block, M_div_m - tile_row_block * rows_per_block]
                )
                npu_dma_memcpy_nd(
                    metadata="outC",
                    bd_id=0,
                    mem=C,
                    offsets=[0, 0, 0, C_row_offset_in_i32s],
                    sizes=[num_tile_rows, N_div_n, m, n_in_i32s_out],
                    strides=[m_x_N_in_i32s_out, n_in_i32s_out, N_in_i32s_out],
                )
                for tile_row in range(num_tile_rows):
                    A_row_offset_in_i32s = (
                        ((tile_row_block * rows_per_block) + tile_row)
                        * m
                        * K
                        * word_size_in
                        // 4
                    )
                    npu_dma_memcpy_nd(
                        metadata="inA",
                        bd_id=2 * tile_row + 1,
                        mem=A,
                        offsets=[0, 0, 0, A_row_offset_in_i32s],
                        sizes=[N_div_n, K_div_k, m, k_in_i32s],
                        strides=[0, k_in_i32s, K_in_i32s],
                    )
                    npu_dma_memcpy_nd(
                        metadata="inB",
                        bd_id=2 * tile_row + 2,
                        mem=B,
                        sizes=[N_div_n, K_div_k, k, n_in_i32s],
                        strides=[n_in_i32s, k_x_N_in_i32s, N_in_i32s],
                    )

                npu_sync(column=0, row=0, direction=0, channel=0)


def emit_module(M=64, K=64, N=64):
    with mlir_mod_ctx() as ctx:
        my_matmul(M, K, N)
        return str(ctx.module)
