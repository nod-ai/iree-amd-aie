// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions. See
// https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: # Apache-2.0 WITH LLVM-exception

#ifndef IREE_AIE_RUNTIME_H
#define IREE_AIE_RUNTIME_H

#include <optional>
#include <ostream>
#include <sstream>

#include "llvm/ADT/Twine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FormattedStream.h"

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
void startCDOFileStream(const char *cdoFileName);
void endCurrentCDOFileStream();
void FileHeader();
void EnAXIdebug();
void setEndianness(bool endianness);
void configureHeader();
void insertNoOpCommand(unsigned int numPadBytes);
}

#define OSTREAM_OP(O_TYPE, TYPE) O_TYPE &operator<<(O_TYPE &os, const TYPE &s);

namespace mlir::iree_compiler::AMDAIE {
#define TO_STRING(TYPE) std::string to_string(const TYPE &t);

#define TO_STRINGS(_) \
  _(AieRC)            \
  _(XAie_LocType)     \
  _(XAie_Lock)        \
  _(XAie_Packet)

TO_STRINGS(TO_STRING)
#undef TO_STRING
#undef TO_STRINGS
}  // namespace mlir::iree_compiler::AMDAIE

#define BOTH_OSTREAM_OP(OSTREAM_OP_, TYPE) \
  OSTREAM_OP_(std::ostream, TYPE)          \
  OSTREAM_OP_(llvm::raw_ostream, TYPE)

#define BOTH_OSTREAM_OPS_FORALL_TYPES(OSTREAM_OP_, _) \
  _(OSTREAM_OP_, AieRC)                               \
  _(OSTREAM_OP_, StrmSwPortType)                      \
  _(OSTREAM_OP_, XAie_LocType)                        \
  _(OSTREAM_OP_, XAie_Lock)                           \
  _(OSTREAM_OP_, XAie_Packet)

BOTH_OSTREAM_OPS_FORALL_TYPES(OSTREAM_OP, BOTH_OSTREAM_OP)
#undef OSTREAM_OP

// https://stackoverflow.com/a/32230306
template <typename H1>
llvm::raw_ostream &showArgs(llvm::raw_ostream &out, const char *label,
                            H1 &&value) {
  if constexpr (std::is_pointer<H1>::value)
    return out << label << "=" << "ptr";
  else
    return out << label << "=" << std::forward<H1>(value);
}

template <typename H1, typename... T>
llvm::raw_ostream &showArgs(llvm::raw_ostream &out, const char *label,
                            H1 &&value, T &&...rest) {
  const char *pcomma = strchr(label, ',');
  if constexpr (std::is_pointer<H1>::value)
    return showArgs(out.write(label, pcomma - label) << "=ptr,", pcomma + 1,
                    std::forward<T>(rest)...);
  else
    return showArgs(out.write(label, pcomma - label)
                        << "=" << std::forward<H1>(value) << ',',
                    pcomma + 1, std::forward<T>(rest)...);
}

#define SHOW_ARGS(os, ...) showArgs(os, #__VA_ARGS__, __VA_ARGS__)

// So that we can use the pattern if(auto r = TRY_XAIE_API...) { // r is nonzero
// }
static_assert(XAIE_OK == 0);

#define TRY_XAIE_API_FATAL_ERROR(API, ...)                                 \
  do {                                                                     \
    LLVM_DEBUG(llvm::dbgs() << "XAIE API: " << #API << " with args: ");    \
    LLVM_DEBUG(SHOW_ARGS(llvm::dbgs(), __VA_ARGS__));                      \
    LLVM_DEBUG(llvm::dbgs() << "\n");                                      \
    if (auto r = API(__VA_ARGS__))                                         \
      llvm::report_fatal_error(llvm::Twine(#API " failed with ") +         \
                               mlir::iree_compiler::AMDAIE::to_string(r)); \
  } while (0)

#define TRY_XAIE_API_EMIT_ERROR(OP, API, ...)                           \
  do {                                                                  \
    LLVM_DEBUG(llvm::dbgs() << "XAIE API: " << #API << " with args: "); \
    LLVM_DEBUG(SHOW_ARGS(llvm::dbgs(), __VA_ARGS__));                   \
    LLVM_DEBUG(llvm::dbgs() << "\n");                                   \
    if (auto r = API(__VA_ARGS__))                                      \
      return OP.emitOpError() << #API " failed with " << r;             \
  } while (0)

#define TRY_XAIE_API_LOGICAL_RESULT(API, ...)                           \
  do {                                                                  \
    LLVM_DEBUG(llvm::dbgs() << "XAIE API: " << #API << " with args: "); \
    LLVM_DEBUG(SHOW_ARGS(llvm::dbgs(), __VA_ARGS__));                   \
    LLVM_DEBUG(llvm::dbgs() << "\n");                                   \
    if (auto r = API(__VA_ARGS__)) {                                    \
      llvm::errs() << #API " failed with " << r;                        \
      return failure();                                                 \
    }                                                                   \
  } while (0)

#endif  // IREE_AIE_RUNTIME_H
