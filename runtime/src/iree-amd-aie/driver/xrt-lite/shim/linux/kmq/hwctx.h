// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2024, Advanced Micro Devices, Inc. All rights reserved.

#ifndef _HWCTX_XDNA_H_
#define _HWCTX_XDNA_H_

#include <map>

#include "amdxdna_accel.h"
#include "device.h"
#include "shim_debug.h"

namespace shim_xdna {

struct hw_q;
struct bo;
struct device;

struct cu_info {
  std::string m_name;
  size_t m_func;
  std::vector<uint8_t> m_pdi;
};

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

struct hw_ctx {
  using qos_t = std::map<std::string, uint32_t>;
  enum class access_mode : uint8_t { exclusive = 0, shared = 1 };
  device &m_device;
  uint32_t m_handle = AMDXDNA_INVALID_CTX_HANDLE;
  amdxdna_qos_info m_qos = {};
  std::vector<cu_info> m_cu_info;
  std::unique_ptr<hw_q> m_q;
  uint32_t m_ops_per_cycle;
  uint32_t m_num_cols;
  uint32_t m_doorbell;
  std::unique_ptr<bo> m_log_bo;
  void *m_log_buf;
  std::vector<std::unique_ptr<bo>> m_pdi_bos;

  hw_ctx(device &dev, const qos_t &qos, std::unique_ptr<hw_q> q,
         const xrt::xclbin &xclbin);
  hw_ctx(device &dev, const xrt::xclbin &xclbin, const qos_t &qos);
  ~hw_ctx();

  // TODO
  std::unique_ptr<bo> alloc_bo(void *userptr, size_t size, uint64_t flags);
  std::unique_ptr<bo> alloc_bo(size_t size, uint64_t flags);
  std::unique_ptr<bo> import_bo(pid_t, int);
  cuidx_type open_cu_context(const std::string &cuname);
  void create_ctx_on_device();
  void init_log_buf();
  void fini_log_buf() const;
  void delete_ctx_on_device();

  hw_q *get_hw_queue() const;
};

std::unique_ptr<hw_ctx> create_hw_context(
    device &dev, const xrt::xclbin &xclbin,
    const std::map<std::string, uint32_t> &qos);

}  // namespace shim_xdna

#endif  // _HWCTX_XDNA_H_
