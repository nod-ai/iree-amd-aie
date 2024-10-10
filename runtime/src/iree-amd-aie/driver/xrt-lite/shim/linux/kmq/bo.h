// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2024, Advanced Micro Devices, Inc. All rights reserved.

#ifndef _BO_XDNA_H_
#define _BO_XDNA_H_

#include <string>

#include "amdxdna_accel.h"
#include "device.h"
#include "hwctx.h"

namespace shim_xdna {

#define XRT_BO_USE_NORMAL 0
#define XRT_BO_USE_DEBUG 1

// map_type - determines how a buffer is mapped
enum class map_type { read, write };

enum xclBOSyncDirection {
  XCL_BO_SYNC_BO_TO_DEVICE = 0,
  XCL_BO_SYNC_BO_FROM_DEVICE,
  XCL_BO_SYNC_BO_GMIO_TO_AIE,
  XCL_BO_SYNC_BO_AIE_TO_GMIO,
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
  bo(const pdev &p, int ehdl);
  // Support BO creation from internal
  bo(const pdev &p, size_t size, amdxdna_bo_type type);
  ~bo();

  void *map(map_type) const;
  void unmap(void *addr);
  void sync(direction, size_t size, size_t offset);
  properties get_properties() const;
  std::unique_ptr<shared_handle> share() const;
  // For cmd BO only
  void set_cmd_id(uint64_t id);
  // For cmd BO only
  uint64_t get_cmd_id() const;
  uint32_t get_drm_bo_handle() const;
  amdxdna_bo_type get_type() const;
  // DRM BO managed by driver.
  void bind_at(size_t pos, const bo *bh, size_t offset, size_t size);
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

}  // namespace shim_xdna

#endif
