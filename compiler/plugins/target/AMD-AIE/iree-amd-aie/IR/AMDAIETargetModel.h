// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// In absence of a complete hardware model interface, this file contains some
// constants to describe hardware-related parameters used in transformations.
// This is meant to be temporary.
//
//===----------------------------------------------------------------------===//

#ifndef IREE_COMPILER_AMDAIE_TARGET_MODEL_H_
#define IREE_COMPILER_AMDAIE_TARGET_MODEL_H_

namespace mlir::iree_compiler::AMDAIE {

//===----------------------------------------------------------------------===//
//
// DMA iteration dimensions
//
// DMAs support multi-dimensional addressing through buffer descriptors in two
// ways:
// 1. Intra-iteration access pattern. Specified via 'strides' ('steps' in buffer
// descriptor lingo), 'sizes' ('wraps' in buffer descriptro lingo) and
// 'padding'. When a DMA executes a buffer descriptor, it will access the data
// (read/write) as specified by the intra-iteration access pattern.
// 2. Inter-iteration access pattern. Specified via an iteration 'stride',
// 'size' and 'current_iteration' ('stride' is the same as 'stepsize' and 'size'
// is the same as 'wrap' in buffer descriptor lingo). Here, 'current_iteration'
// keeps track of the current execution iteration of the buffer descriptor and
// is incremented after buffer descriptor execution. the 'stride' is the offset
// to be used for each execution of the buffer descriptor, relative to the
// previous one. When 'iteration_current' is equal to 'size', the
// 'iteration_current' is reset to zero.
//
// Although all DMAs use the same buffer descriptor format to describe the
// execution configuration, the intra-iteration and inter-dimensions are
// typically used for different purposes on different DMAs. See for example the
// usage of these constants inside the DMA loop subsumption pass.
//
//===----------------------------------------------------------------------===//

/// Shim DMAs support 3 intra-iteration dimensions + 1 inter-iteration
/// dimension.
static const int64_t kAMDAIEShimDmaNbIntraDims = 3;
static const int64_t kAMDAIEShimDmaNbInterDims = 1;

/// MemTile DMAs support 4 intra-iteration dimensions + 1 inter-iteration
/// dimension.
static const int64_t kAMDAIEMemTileDmaNbIntraDims = 4;
static const int64_t kAMDAIEMemTileDmaNbInterDims = 1;

/// Core DMAs support 3 intra-iteration dimensions + 1 inter-iteration
/// dimension.
static const int64_t kAMDAIECoreDmaNbIntraDims = 3;
static const int64_t kAMDAIECoreDmaNbInterDims = 1;

}  // namespace mlir::iree_compiler::AMDAIE

#endif  // IREE_COMPILER_AMDAIE_TARGET_MODEL_H_
