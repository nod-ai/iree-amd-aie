// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2024, Advanced Micro Devices, Inc. All rights reserved.

#ifndef PCIE_DEVICE_LINUX_XDNA_H
#define PCIE_DEVICE_LINUX_XDNA_H

#include <map>
#include <mutex>

#include "shared.h"
#include "shim_debug.h"

namespace shim_xdna {

#define XRT_NULL_HANDLE NULL

// cuidx_type - encode cuidx and domain
//
// @domain_index: index within domain
// @domain:       domain identifier
// @index:        combined encoded index
//
// The domain_index is used in command cumask in exec_buf
// The combined index is used in context creation in open_context
struct cuidx_type {
  union {
    std::uint32_t index;
    struct {
      std::uint16_t domain_index;  // [15-0]
      std::uint16_t domain;        // [31-16]
    };
  };

  // Ensure consistent use of domain and index types
  using domain_type = uint16_t;
  using domain_index_type = uint16_t;
};

struct hw_ctx;
struct pdev;
struct bo;

struct device {
  device(const pdev& pdev, void* shim_handle);

  ~device();

  using qos_type = std::map<std::string, uint32_t>;
  enum class access_mode : uint8_t { exclusive = 0, shared = 1 };
  std::unique_ptr<bo> alloc_bo(void* userptr, uint32_t ctx_id, size_t size,
                               uint64_t flags);

  // std::unique_ptr<hw_ctx> create_hw_context(const device& dev,
  //                                           const qos_type& qos) const;

  std::unique_ptr<bo> import_bo(shared_handle::export_handle ehdl) const;

  const pdev& get_pdev() const;

  std::unique_ptr<bo> alloc_bo(size_t size, uint64_t flags);

  std::unique_ptr<bo> import_bo(pid_t, shared_handle::export_handle);

  std::unique_ptr<hw_ctx> create_hw_context(const qos_type& qos,
                                            access_mode mode) const;

  std::vector<char> read_aie_mem(uint16_t col, uint16_t row, uint32_t offset,
                                 uint32_t size);

  size_t write_aie_mem(uint16_t col, uint16_t row, uint32_t offset,
                       const std::vector<char>& buf);

  uint32_t read_aie_reg(uint16_t col, uint16_t row, uint32_t reg_addr);

  bool write_aie_reg(uint16_t col, uint16_t row, uint32_t reg_addr,
                     uint32_t reg_val);

  const pdev& m_pdev;  // The pcidev that this device object is derived from
  std::map<uint32_t, bo*> m_bo_map;
  void* m_handle = XRT_NULL_HANDLE;

  mutable std::mutex m_mutex;
};

}  // namespace shim_xdna

#endif
