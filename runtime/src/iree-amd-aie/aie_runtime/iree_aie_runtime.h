// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions. See
// https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: # Apache-2.0 WITH LLVM-exception

#ifndef IREE_AIE_RUNTIME_H
#define IREE_AIE_RUNTIME_H

#ifdef _WIN32
#ifndef IREE_AIE_RUNTIME_EXPORT
#ifdef iree_aie_runtime_EXPORTS
// We are building this library
#define IREE_AIE_RUNTIME_EXPORT __declspec(dllexport)
#else
// We are using this library
#define IREE_AIE_RUNTIME_EXPORT __declspec(dllimport)
#endif  // iree_aie_runtime_EXPORTS
#endif  // IREE_AIE_RUNTIME_EXPORT
#else
// Non-windows: use visibility attributes.
#define IREE_AIE_RUNTIME_EXPORT __attribute__((visibility("default")))
#endif  // _WIN32

extern "C" {
#include "xaiengine.h"

enum byte_ordering { Little_Endian, Big_Endian };
void startCDOFileStream(const char* cdoFileName);
void endCurrentCDOFileStream();
void FileHeader();
void EnAXIdebug();
void setEndianness(bool endianness);
void configureHeader();
}

#endif  // IREE_AIE_RUNTIME_H
