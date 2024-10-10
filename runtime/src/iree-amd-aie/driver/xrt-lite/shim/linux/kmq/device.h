// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2024, Advanced Micro Devices, Inc. All rights reserved.

#ifndef PCIE_DEVICE_LINUX_XDNA_H
#define PCIE_DEVICE_LINUX_XDNA_H

#include <filesystem>
#include <map>

#include "experimental/xrt_xclbin.h"
#include "fence.h"
#include "shim_debug.h"

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

  xrt::xclbin m_xclbin;
  mutable std::mutex m_mutex;
  pdev m_pdev;

  device();
  ~device();

  xrt::xclbin get_xclbin(const xrt::uuid &xclbin_id) const;

  std::unique_ptr<bo> import_bo(int ehdl) const;
  const pdev &get_pdev() const;

  std::unique_ptr<bo> alloc_bo(void *userptr, uint32_t ctx_id, size_t size,
                               uint64_t flags);

  std::unique_ptr<bo> alloc_bo(size_t size, uint64_t flags);
  std::unique_ptr<bo> alloc_bo(void *userptr, size_t size, uint64_t flags);
  std::unique_ptr<bo> import_bo(pid_t, int);
  std::unique_ptr<hw_ctx> create_hw_context(
      const xrt::uuid &xclbin_uuid, const std::map<std::string, uint32_t> &qos);
  std::vector<char> read_aie_mem(uint16_t col, uint16_t row, uint32_t offset,
                                 uint32_t size);
  size_t write_aie_mem(uint16_t col, uint16_t row, uint32_t offset,
                       const std::vector<char> &buf);
  uint32_t read_aie_reg(uint16_t col, uint16_t row, uint32_t reg_addr);
  void write_aie_reg(uint16_t col, uint16_t row, uint32_t reg_addr,
                     uint32_t reg_val);
  std::unique_ptr<fence_handle> create_fence(fence_handle::access_mode);
  std::unique_ptr<fence_handle> import_fence(pid_t, int);
  void record_xclbin(const xrt::xclbin &xclbin);
};

std::string read_sysfs(const std::string &filename);
std::filesystem::path find_npu_device();

}  // namespace shim_xdna

#endif
