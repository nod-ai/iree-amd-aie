// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_AMD_AIE_TRANSFORMS_PASSDETAIL_H_
#define IREE_AMD_AIE_TRANSFORMS_PASSDETAIL_H_

#include "iree-amd-aie/Transforms/KernelDispatch.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Operation.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassOptions.h"

namespace mlir::iree_compiler::AMDAIE {

#define GEN_PASS_DECL
#define GEN_PASS_DEF_AMDAIEACCESSTOACQUIRERELEASE
#define GEN_PASS_DEF_AMDAIEACQUIRERELEASETOUSELOCK
#define GEN_PASS_DEF_AMDAIEASSIGNCONNECTIONTYPES
#define GEN_PASS_DEF_AMDAIEASSIGNCHANNELS
#define GEN_PASS_DEF_AMDAIEASSIGNLOGICALOBJECTFIFODEPTH
#define GEN_PASS_DEF_AMDAIEASSIGNNPUDMABDIDS
#define GEN_PASS_DEF_AMDAIEASSIGNPACKETIDS
#define GEN_PASS_DEF_AMDAIEASSIGNTILES
#define GEN_PASS_DEF_AMDAIEBRIDGETOAIR
#define GEN_PASS_DEF_AMDAIEBUFFERIZETOALLOCATION
#define GEN_PASS_DEF_AMDAIECANONICALIZEDOUBLYSTRIDEDOP
#define GEN_PASS_DEF_AMDAIECANONICALIZENPUDMACPYND
#define GEN_PASS_DEF_AMDAIECLEANUP
#define GEN_PASS_DEF_AMDAIECOMBINESTRIDEDOPS
#define GEN_PASS_DEF_AMDAIECONNECTIONTOFLOW
#define GEN_PASS_DEF_AMDAIECONTROLCODEFORALLTOFOR
#define GEN_PASS_DEF_AMDAIECONTROLCODELOOPUNROLL
#define GEN_PASS_DEF_AMDAIECONTROLCODETOHALFDMACPYND
#define GEN_PASS_DEF_AMDAIECONTROLCODELOWERING
#define GEN_PASS_DEF_AMDAIECONTROLCODETOTRANSACTION
#define GEN_PASS_DEF_AMDAIECONVERTCOREFORALLTOFOR
#define GEN_PASS_DEF_AMDAIECREATEAIEWORKGROUP
#define GEN_PASS_DEF_AMDAIECREATELOGICALOBJECTFIFOLINK
#define GEN_PASS_DEF_AMDAIECREATEREFERENCETOALLOCATION
#define GEN_PASS_DEF_AMDAIEDECOMPOSELINALGEXTPACKUNPACKTOAIR
#define GEN_PASS_DEF_AMDAIEDISTRIBUTECORESANDOBJECTFIFOS
#define GEN_PASS_DEF_AMDAIEDISTRIBUTEL1ALLOCATIONS
#define GEN_PASS_DEF_AMDAIEDMACOMPOSITION
#define GEN_PASS_DEF_AMDAIEDMACSE
#define GEN_PASS_DEF_AMDAIEDMALOOPSUBSUMPTION
#define GEN_PASS_DEF_AMDAIEDMATOCIRCULARDMA
#define GEN_PASS_DEF_AMDAIEFLATTENLOGICALOBJECTFIFO
#define GEN_PASS_DEF_AMDAIELINALGFUNCTIONOUTLINING
#define GEN_PASS_DEF_AMDAIEFUSECONSUMERINTOLOOP
#define GEN_PASS_DEF_AMDAIEFUSEFILLINTOFORALL
#define GEN_PASS_DEF_AMDAIEFUSEPACKINTOLOOP
#define GEN_PASS_DEF_AMDAIEHOISTFORLOOPAFFINEAPPLY
#define GEN_PASS_DEF_AMDAIEHOISTLOGICALOBJFIFO
#define GEN_PASS_DEF_AMDAIEINSERTAIEWORKGROUP
#define GEN_PASS_DEF_AMDAIEINSERTCORES
#define GEN_PASS_DEF_AMDAIEINSERTDMABDCHAIN
#define GEN_PASS_DEF_AMDAIEINSERTINFINITELOOPAROUNDCOREBLOCK
#define GEN_PASS_DEF_AMDAIEINSERTLOOPSFORVECTORIZATION
#define GEN_PASS_DEF_AMDAIELINKEXECUTABLES
#define GEN_PASS_DEF_AMDAIELOADALIGNMENTRESET
#define GEN_PASS_DEF_AMDAIELOCALIZELOGICALOBJECTFIFO
#define GEN_PASS_DEF_AMDAIELOWEREXECUTABLETARGET
#define GEN_PASS_DEF_AMDAIELOWERINGSTRATEGY
#define GEN_PASS_DEF_AMDAIELOWERFUNCARGS
#define GEN_PASS_DEF_AMDAIELOWERTOAIE
#define GEN_PASS_DEF_AMDAIELOWERTOUKERNELS
#define GEN_PASS_DEF_AMDAIELOWERWORKGROUPCOUNT
#define GEN_PASS_DEF_AMDAIEMAPFORALLTOCORES
#define GEN_PASS_DEF_AMDAIENONEACCESSTOTEMPORARYBUFFER
#define GEN_PASS_DEF_AMDAIENORMALIZELOOPBOUNDS
#define GEN_PASS_DEF_AMDAIENPUDMATOHALFDMACPYND
#define GEN_PASS_DEF_AMDAIEOBJFIFOBUFFERIZATION
#define GEN_PASS_DEF_AMDAIEPACKANDTRANSPOSE
#define GEN_PASS_DEF_AMDAIECONVERTTODMA
#define GEN_PASS_DEF_AMDAIEPAD
#define GEN_PASS_DEF_AMDAIEPEELFORLOOP
#define GEN_PASS_DEF_AMDAIEPROPAGATEDATALAYOUT
#define GEN_PASS_DEF_AMDAIEREMOVEMEMORYSPACE
#define GEN_PASS_DEF_AMDAIESINKINTOCORE
#define GEN_PASS_DEF_AMDAIESPLITLOGICALOBJFIFOS
#define GEN_PASS_DEF_AMDAIESPLITLOGICALOBJFIFOSFORCONNECTIONREUSE
#define GEN_PASS_DEF_AMDAIETEMPORARYALLOCBUFFERIZATION
#define GEN_PASS_DEF_AMDAIETILE
#define GEN_PASS_DEF_AMDAIETILEANDFUSE
#define GEN_PASS_DEF_AMDAIEVECTORIZATION
#include "iree-amd-aie/Transforms/Passes.h.inc"

}  // namespace mlir::iree_compiler::AMDAIE

#endif  // IREE_AMD_AIE_TRANSFORMS_PASSDETAIL_H_
