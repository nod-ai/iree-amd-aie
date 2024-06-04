//
// Created by mlevental on 6/3/24.
//

#ifndef IREE_IREE_AMD_AIE_RUNTIME_H
#define IREE_IREE_AMD_AIE_RUNTIME_H

#ifdef _WIN32
#ifndef IREE_AMD_AIE_RUNTIME_EXPORT
#ifdef iree_amd_aie_runtime_EXPORTS
// We are building this library
#define IREE_AMD_AIE_RUNTIME_EXPORT __declspec(dllexport)
#else
// We are using this library
#define IREE_AMD_AIE_RUNTIME_EXPORT __declspec(dllimport)
#endif  // iree_amd_aie_runtime_EXPORTS
#endif  // IREE_AMD_AIE_RUNTIME_EXPORT
#else
// Non-windows: use visibility attributes.
#define IREE_AMD_AIE_RUNTIME_EXPORT __attribute__((visibility("default")))
#endif  // _WIN32

class iree_amd_aie_runtime {};

#endif  // IREE_IREE_AMD_AIE_RUNTIME_H
