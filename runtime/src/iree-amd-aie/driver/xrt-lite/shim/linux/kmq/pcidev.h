// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2024, Advanced Micro Devices, Inc. All rights reserved.

#ifndef PCIDEV_XDNA_H
#define PCIDEV_XDNA_H

#include <sys/mman.h>

#include <mutex>

#include "bo.h"
#include "device.h"
#include "pcidev.h"
#include "shim_debug.h"

namespace shim_xdna {

#define INVALID_ID 0xffff

struct drv;

struct pdev {
  pdev(std::shared_ptr<const drv> driver, std::string sysfs_name);
  ~pdev();

  void sysfs_get(const std::string& subdev, const std::string& entry,
                 std::string& err, std::vector<uint64_t>& iv) const;

  template <typename T>
  void sysfs_get(const std::string& subdev, const std::string& entry,
                 std::string& err, T& i, const T& default_val) {
    std::vector<uint64_t> iv;
    sysfs_get(subdev, entry, err, iv);
    if (!iv.empty())
      i = static_cast<T>(iv[0]);
    else
      i = static_cast<T>(default_val);  // default value
  }

  std::string get_subdev_path(const std::string& subdev, uint32_t idx) const;

  std::shared_ptr<device> create_device(void* handle) const;

  void ioctl(unsigned long cmd, void* arg) const;

  void* mmap(void* addr, size_t len, int prot, int flags, off_t offset) const;

  void munmap(void* addr, size_t len) const;

  int open(const std::string& subdev, uint32_t idx, int flag) const;
  int open(const std::string& subdev, int flag) const;

  void open() const;

  void close() const;

  void on_last_close() const;
  int map_usr_bar() const;

  // Virtual address of memory mapped BAR0, mapped on first use, once mapped,
  // never change.
  mutable char* m_user_bar_map = reinterpret_cast<char*>(MAP_FAILED);

  std::shared_ptr<const drv> m_driver;
  mutable int m_dev_fd = -1;
  mutable int m_dev_users = 0;
  mutable std::mutex m_lock;
  uint16_t m_domain = INVALID_ID;
  uint16_t m_bus = INVALID_ID;
  uint16_t m_dev = INVALID_ID;
  uint16_t m_func = INVALID_ID;
  uint32_t m_instance = INVALID_ID;
  std::string m_sysfs_name;  // dir name under /sys/bus/pci/devices
  int m_user_bar = 0;        // BAR mapped in by tools, default is BAR0
  size_t m_user_bar_size = 0;
  bool m_is_mgmt = false;
  bool m_is_ready = false;

  // Create on first device creation and removed right before device is closed
  mutable std::unique_ptr<bo> m_dev_heap_bo;
};

}  // namespace shim_xdna

#endif
