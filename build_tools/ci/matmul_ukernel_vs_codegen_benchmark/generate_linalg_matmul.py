def generate(M, N, K):
    """
    Create MLIR linalg matmul function with given dimensions M, N, K.
    Initialize the output with '0.0'
    bfloat16 operands, float32 output.
    """

    mlirStr = """
!A = tensor<%dx%dxbf16>
!B = tensor<%dx%dxbf16>
!C = tensor<%dx%dxf32>""" % (
        M,
        K,
        K,
        N,
        M,
        N,
    )

    mlirStr += """

// C = 0 + A @ B (The '@' symbol denotes matrix multiplication)
func.func @matmul(%A : !A, %B : !B) -> !C {

  // Initialize output tensor with '0'
  %init_acc = tensor.empty() : !C
  %c0_acc_type = arith.constant 0.0 : f32

  %acc = linalg.fill ins(%c0_acc_type : f32)
                     outs(%init_acc : !C) -> !C

  %C = linalg.matmul ins(%A, %B: !A, !B) outs(%acc: !C) -> !C

  return %C: !C
}
"""
    return mlirStr


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 4:
        print(f"Usage: {sys.argv[0]} M=int N=int K=int")
        sys.exit(1)

    M = -1
    N = -1
    K = -1

    for arg in sys.argv[1:]:
        if arg.startswith("M="):
            M = int(arg[2:])
        elif arg.startswith("N="):
            N = int(arg[2:])
        elif arg.startswith("K="):
            K = int(arg[2:])
        else:
            print(f"Invalid argument: {arg}")
            sys.exit(1)

    if M < 0 or N < 0 or K < 0:
        print(f"Invalid arguments: M={M} N={N} K={K}")
        sys.exit(1)

    print(generate(M, N, K))
