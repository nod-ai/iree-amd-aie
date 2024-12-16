// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2024, Advanced Micro Devices, Inc. All rights reserved.

#ifndef _HWCTX_XDNA_H_
#define _HWCTX_XDNA_H_

#include <map>

#include "device.h"

namespace shim_xdna {

struct hw_q;
struct bo;
struct device;

enum cert_log_flag { debug_buffer = 0, trace_buffer };

struct cert_log_metadata {
#define CERT_MAGIC_NO 0x43455254  // "CERT"
  uint32_t magic_no;
  uint8_t major;
  uint8_t minor;
  uint8_t cert_log_flag;
  uint8_t num_cols;       // how many valid cols, up to 8 for now
  uint64_t col_paddr[8];  // device accessible address array for each valid col
  uint32_t col_size[8];   // bo size for each valid col
};

struct cu_info {
  std::string m_name;
  size_t m_func;
  std::vector<uint8_t> m_pdi;
};

struct cuidx_t {
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
  enum class access_mode : uint8_t { exclusive = 0, shared = 1 };
  device &m_device;
  uint32_t m_handle = AMDXDNA_INVALID_CTX_HANDLE;
  amdxdna_qos_info m_qos = {};
  std::vector<cu_info> m_cu_info;
  cert_log_metadata m_metadata;
  std::unique_ptr<hw_q> m_q;
  uint32_t m_ops_per_cycle;
  uint32_t m_num_rows;
  uint32_t m_num_cols;
  uint32_t m_doorbell;
  uint32_t m_syncobj;
  std::unique_ptr<bo> m_log_bo;
  void *m_log_buf;
  std::vector<std::unique_ptr<bo>> m_pdi_bos;

  hw_ctx(device &dev, const std::map<std::string, uint32_t> &qos,
         std::unique_ptr<hw_q> q, const std::vector<uint8_t> &pdi,
         const std::string &cu_name, uint32_t n_rows, uint32_t n_cols);
  hw_ctx(device &dev, const std::vector<uint8_t> &pdi,
         const std::string &cu_name, uint32_t n_rows, uint32_t n_cols,
         const std::map<std::string, uint32_t> &qos = {});
  ~hw_ctx();
  // no copying
  hw_ctx(const hw_ctx &) = delete;
  hw_ctx &operator=(const hw_ctx &) = delete;

  std::unique_ptr<bo> alloc_bo(size_t size, shim_xcl_bo_flags flags);
  std::unique_ptr<bo> import_bo(pid_t, int);

  cuidx_t open_cu_context(const std::string &cuname);
  void create_ctx_on_device();
  void init_log_buf();
  void fini_log_buf() const;
  void delete_ctx_on_device() const;
  void delete_syncobj() const;

  hw_q *get_hw_queue() const;

  void set_metadata(int num_cols, size_t size, uint64_t bo_paddr, uint8_t flag);
};

}  // namespace shim_xdna

#endif  // _HWCTX_XDNA_H_
