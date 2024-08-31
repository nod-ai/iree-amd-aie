#!/bin/bash

LLVM_AIE_DIR=/home/mlevental/dev_projects/iree-amd-aie/llvm-aie
IREE_INSTALL_DIR=/home/mlevental/dev_projects/iree-amd-aie/cmake-build-debug
AIETOOLS=/home/mlevental/dev_projects/iree-amd-aie/Vitis/2024.2/aietools
export PATH=$AIETOOLS/bin/unwrapped/lnx64.o:$AIETOOLS/bin/unwrapped/lnx64.o/aie_ml:$AIETOOLS/tps/lnx64/target_aie_ml/bin/LNa64bin:$PATH
export XILINXD_LICENSE_FILE=$HOME/.Xilinx/aie.lic
export LD_LIBRARY_PATH=/home/mlevental/dev_projects/iree-amd-aie/Vitis/2024.2/aietools/lib/lnx64.o:$LD_LIBRARY_PATH
export RDI_DATADIR=/home/mlevental/dev_projects/iree-amd-aie/Vitis/2024.2/aietools/data
export FLEXLM_DIAGNOSTICS=3
export CHESSDIR=$AIETOOLS/tps/lnx64/target_aie_ml/chessdir

$IREE_INSTALL_DIR/tools/iree-opt $PWD/i16_max_reduce.mlir --convert-vector-to-aievec -lower-affine -canonicalize -cse --convert-aievec-to-llvm --convert-scf-to-cf --iree-convert-to-llvm | $IREE_INSTALL_DIR/tools/iree-aie-translate --mlir-to-llvmir -o kernel.ll
$AIETOOLS/bin/unwrapped/lnx64.o/xchesscc -j1 -pme -P $AIETOOLS/data/aie_ml/lib -f -C Release_LLVM +w $PWD -D__AIENGINE__ -D__AIE_ARCH__=20 -D__AIEARCH__=20 -I $AIETOOLS/include -d kernel.ll

#$LLVM_AIE_DIR/bin/clang --target=aie2-none-unknown-elf -c kernel.ll -o kernel.o
#$LLVM_AIE_DIR/bin/clang --target=aie2-none-unknown-elf -Wl,--gc-sections -Wl,--orphan-handling=warn -Wl,-T,$PWD/ldfile kernel.o -o test.exe -v

xca_udm_dbg +C -T -P $AIETOOLS/data/aie_ml/lib -t "$PWD/profiling.tcl $PWD/a.out"
#xca_udm_dbg +C -T -P $AIETOOLS/data/aie_ml/lib -t "$PWD/sim.tcl"
#xca_udm_dbg -P $AIETOOLS/data/aie_ml/lib -p $PWD/test.exe
