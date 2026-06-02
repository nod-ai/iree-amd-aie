// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2024, Advanced Micro Devices, Inc. All rights reserved.

#ifndef PCIE_DEVICE_LINUX_XDNA_H
#define PCIE_DEVICE_LINUX_XDNA_H

#include <sys/types.h>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "fence.h"
#include "xrt_mem.h"

namespace shim_xdna {
struct pdev;
struct bo;
struct hw_ctx;

enum class power_mode : uint8_t {
  default_mode = 0,
  low,
  medium,
  high,
  turbo,
};

struct pdev {
  mutable std::mutex m_lock;
  mutable int m_dev_fd = -1;
  mutable std::unique_ptr<bo> m_dev_heap_bo;
  int m_init_errno = 0;

  pdev();
  explicit pdev(const std::filesystem::path &device_path);
  ~pdev();

  int init_errno() const;
  int open_device(const std::filesystem::path &device_path);
  // Returns 0 on success or the failing errno for recoverable ioctl failures.
  int try_ioctl(unsigned long cmd, void *arg) const;
  int try_mmap(void *addr, size_t len, int prot, int flags, off_t offset,
               void **out_ptr) const;
};

struct device {
  enum class access_mode : uint8_t { exclusive = 0, shared = 1 };
  pdev m_pdev;
  uint32_t n_rows;
  uint32_t n_cols;
  int m_init_errno = 0;

  device(uint32_t n_rows, uint32_t n_cols);
  device(uint32_t n_rows, uint32_t n_cols,
         const std::filesystem::path &device_path);
  device(uint32_t n_rows, uint32_t n_cols, power_mode mode);
  device(uint32_t n_rows, uint32_t n_cols,
         const std::filesystem::path &device_path, power_mode mode);
  ~device();
  static int create(uint32_t n_rows, uint32_t n_cols,
                    const std::filesystem::path &device_path,
                    std::unique_ptr<device> *out_device);
  int init_errno() const;

  const pdev &get_pdev() const;

  int alloc_bo(uint32_t ctx_id, size_t size, shim_xcl_bo_flags flags,
               std::unique_ptr<bo> *out_bo);
  int alloc_bo(size_t size, uint32_t flags, std::unique_ptr<bo> *out_bo);
  int alloc_bo(size_t size, shim_xcl_bo_flags flags,
               std::unique_ptr<bo> *out_bo);

  int create_hw_context(const std::vector<uint8_t> &pdi,
                        const std::string &cu_name,
                        const std::map<std::string, uint32_t> &qos,
                        std::unique_ptr<hw_ctx> *out_context);
  int create_hw_context(const std::vector<uint8_t> &pdi,
                        const std::string &cu_name,
                        std::unique_ptr<hw_ctx> *out_context);

  int read_aie_mem(uint16_t col, uint16_t row, uint32_t offset, uint32_t size,
                   std::vector<char> *out_buf);
  int write_aie_mem(uint16_t col, uint16_t row, uint32_t offset,
                    const std::vector<char> &buf, size_t *out_size);
  int read_aie_reg(uint16_t col, uint16_t row, uint32_t reg_addr,
                   uint32_t *out_reg_val);
  int write_aie_reg_checked(uint16_t col, uint16_t row, uint32_t reg_addr,
                            uint32_t reg_val);

  int get_power_mode(power_mode *out_mode) const;
  // Returns 0 on success or the errno from DRM_IOCTL_AMDXDNA_SET_STATE.
  int set_power_mode(power_mode mode) const;

  int create_fence(fence_handle::access_mode,
                   std::unique_ptr<fence_handle> *out_fence);
  int import_fence(pid_t, int, std::unique_ptr<fence_handle> *out_fence);
};

std::filesystem::path find_default_accel_device_path();
// Resolves 0 rows/cols to hardware defaults without constructing a device.
// Returns 0 on success or an errno-style value on open/ioctl/metadata failure.
int resolve_core_grid_size(const std::filesystem::path &device_path,
                           uint32_t requested_rows, uint32_t requested_cols,
                           uint32_t *out_rows, uint32_t *out_cols);
std::string read_sysfs(const std::string &filename);
std::string stringify_power_mode(power_mode mode);

}  // namespace shim_xdna

#endif
