// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2024, Advanced Micro Devices, Inc. All rights reserved.

#ifndef _BO_XDNA_H_
#define _BO_XDNA_H_

#include <string>

#include "amdxdna_accel.h"
#include "device.h"
#include "ert.h"
#include "hwctx.h"

namespace shim_xdna {

#define MAX_EXEC_BO_SIZE 4096

enum xclBOSyncDirection {
  XCL_BO_SYNC_BO_TO_DEVICE = 0,
  XCL_BO_SYNC_BO_FROM_DEVICE,
};

// direction - direction of sync operation
enum class direction {
  host2device = XCL_BO_SYNC_BO_TO_DEVICE,
  device2host = XCL_BO_SYNC_BO_FROM_DEVICE,
};

// properties - buffer details
struct properties {
  shim_xcl_bo_flags flags;  // flags of bo
  uint64_t size;            // size of bo
  uint64_t paddr;           // physical address
  uint64_t kmhdl;           // kernel mode handle
};

struct drm_bo {
  bo &m_parent;
  uint32_t m_handle = AMDXDNA_INVALID_BO_HANDLE;
  off_t m_map_offset = AMDXDNA_INVALID_ADDR;
  uint64_t m_xdna_addr = AMDXDNA_INVALID_ADDR;
  uint64_t m_vaddr = AMDXDNA_INVALID_ADDR;

  drm_bo(bo &parent, const amdxdna_drm_get_bo_info &bo_info);
  ~drm_bo();
};

struct bo {
  const pdev &m_pdev;
  void *m_parent = nullptr;
  void *m_aligned = nullptr;
  size_t m_parent_size = 0;
  size_t m_aligned_size = 0;
  shim_xcl_bo_flags m_flags{};
  amdxdna_bo_type m_type = AMDXDNA_BO_INVALID;
  std::unique_ptr<drm_bo> m_drm_bo;
  const shared_handle m_import;
  // Only for AMDXDNA_BO_CMD type
  std::map<size_t, uint32_t> m_args_map;
  mutable std::mutex m_args_map_lock;

  // Command ID in the queue after command submission.
  // Only valid for cmd BO.
  uint64_t m_cmd_id = -1;

  // Used when exclusively assigned to a HW context. By default, BO is shared
  // among all HW contexts.
  uint32_t m_owner_ctx_id = AMDXDNA_INVALID_CTX_HANDLE;

  bo(const pdev &p, uint32_t ctx_id, size_t size, shim_xcl_bo_flags flags,
     amdxdna_bo_type type);
  bo(const pdev &p, uint32_t ctx_id, size_t size, shim_xcl_bo_flags flags);
  bo(const pdev &p, uint32_t ctx_id, size_t size, uint32_t flags);
  bo(const pdev &p, int ehdl);
  // Support BO creation from internal
  bo(const pdev &p, size_t size, amdxdna_bo_type type);
  ~bo();

  void *map() const;
  void unmap(void *addr);
  void sync(direction, size_t size, size_t offset);
  void sync(direction);
  properties get_properties() const;
  size_t size();

  std::unique_ptr<shared_handle> share() const;
  // For cmd BO only
  void set_cmd_id(uint64_t id);
  // For cmd BO only
  uint64_t get_cmd_id() const;
  uint32_t get_drm_bo_handle() const;
  amdxdna_bo_type get_type() const;
  // DRM BO managed by driver.
  void bind_at(size_t pos, const bo &bh, size_t offset, size_t size);
  std::string describe() const;
  // Import DRM BO from m_import shared object
  void import_bo();
  // Free DRM BO in driver
  void free_bo();
  void mmap_bo(size_t align = 0);
  void munmap_bo();
  uint64_t get_paddr() const;
  std::string type_to_name() const;
  void attach_to_ctx();
  void detach_from_ctx();
  // Obtain array of arg BO handles, returns real number of handles
  uint32_t get_arg_bo_handles(uint32_t *handles, size_t num) const;
};

struct exec_buf {
  std::unique_ptr<bo> m_exec_buf_bo;
  ert_start_kernel_cmd *m_cmd_pkt;
  size_t m_cmd_size;
  uint32_t m_op;
  uint32_t m_arg_cnt;
  uint32_t m_reg_idx;
  std::vector<std::pair<std::string, uint64_t> > m_patching_args;

  exec_buf(const pdev &p, uint32_t op);

  static void set_cu_idx(bo &bo_execbuf, cuidx_t cu_idx);
  void set_cu_idx(cuidx_t cu_idx);
  bo* get_exec_buf_bo();

  void add_ctrl_bo(bo &bo_ctrl);
  void add_arg_32(uint32_t val);
  void add_arg_64(uint64_t val);
  void add_arg_bo(bo &bo_arg, std::string arg_name = "");
  void dump();
  void inc_pkt_count(uint32_t n);
};

}  // namespace shim_xdna

#endif
