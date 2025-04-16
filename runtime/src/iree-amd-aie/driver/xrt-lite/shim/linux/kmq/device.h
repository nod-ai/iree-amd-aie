// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2024, Advanced Micro Devices, Inc. All rights reserved.

#ifndef PCIE_DEVICE_LINUX_XDNA_H
#define PCIE_DEVICE_LINUX_XDNA_H

#include <filesystem>
#include <map>

#include "amdxdna_accel.h"
#include "fence.h"
#include "xrt_mem.h"

namespace shim_xdna {
struct pdev;
struct bo;

struct pdev {
  mutable std::mutex m_lock;
  mutable int m_dev_fd = -1;
  mutable std::unique_ptr<bo> m_dev_heap_bo;

  pdev();
  ~pdev();

  void ioctl(unsigned long cmd, void *arg) const;
  void *mmap(void *addr, size_t len, int prot, int flags, off_t offset) const;
};

struct device {
  enum class access_mode : uint8_t { exclusive = 0, shared = 1 };
  pdev m_pdev;
  uint32_t n_rows;
  uint32_t n_cols;

  device(uint32_t n_rows, uint32_t n_cols);
  device(uint32_t n_rows, uint32_t n_cols, amdxdna_power_mode_type power_mode);
  ~device();

  std::unique_ptr<bo> import_bo(int ehdl) const;
  const pdev &get_pdev() const;

  std::unique_ptr<bo> alloc_bo(uint32_t ctx_id, size_t size,
                               shim_xcl_bo_flags flags);
  std::unique_ptr<bo> alloc_bo(size_t size, uint32_t flags);
  std::unique_ptr<bo> alloc_bo(size_t size, shim_xcl_bo_flags flags);
  std::unique_ptr<bo> import_bo(pid_t, int);

  std::unique_ptr<hw_ctx> create_hw_context(
      const std::vector<uint8_t> &pdi, const std::string &cu_name,
      const std::map<std::string, uint32_t> &qos);
  std::unique_ptr<hw_ctx> create_hw_context(const std::vector<uint8_t> &pdi,
                                            const std::string &cu_name);

  std::vector<char> read_aie_mem(uint16_t col, uint16_t row, uint32_t offset,
                                 uint32_t size);
  size_t write_aie_mem(uint16_t col, uint16_t row, uint32_t offset,
                       const std::vector<char> &buf);
  uint32_t read_aie_reg(uint16_t col, uint16_t row, uint32_t reg_addr);
  void write_aie_reg(uint16_t col, uint16_t row, uint32_t reg_addr,
                     uint32_t reg_val);

  // TODO(max): hide amdxdna_accel enums so they don't leak
  amdxdna_power_mode_type get_power_mode() const;
  void set_power_mode(amdxdna_power_mode_type mode) const;

  std::unique_ptr<fence_handle> create_fence(fence_handle::access_mode);
  std::unique_ptr<fence_handle> import_fence(pid_t, int);
};

std::string read_sysfs(const std::string &filename);
std::filesystem::path find_npu_device();
std::string stringify_amdxdna_power_mode_type(
    amdxdna_power_mode_type power_mode);

}  // namespace shim_xdna

#endif
