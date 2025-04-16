// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2024, Advanced Micro Devices, Inc. - All rights reserved

#include "device.h"

#include <fcntl.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <filesystem>
#include <fstream>
#include <iostream>

#include "bo.h"
#include "fence.h"
#include "hwctx.h"
#include "llvm/Support/ErrorHandling.h"
#include "shim_debug.h"
#include "xrt_mem.h"

namespace {

int64_t import_fd(pid_t pid, int ehdl) {
  if (pid == 0 || getpid() == pid) return ehdl;

#if defined(SYS_pidfd_open) && defined(SYS_pidfd_getfd)
  auto pidfd = syscall(SYS_pidfd_open, pid, 0);
  if (pidfd < 0) shim_xdna::shim_err(errno, "pidfd_open failed");

  int64_t fd = syscall(SYS_pidfd_getfd, pidfd, ehdl, 0);
  if (fd < 0) {
    if (errno == EPERM) {
      shim_xdna::shim_err(
          errno,
          "pidfd_getfd failed, check that ptrace access mode "
          "allows PTRACE_MODE_ATTACH_REALCREDS.  For more details please "
          "check /etc/sysctl.d/10-ptrace.conf");
    }

    shim_xdna::shim_err(errno, "pidfd_getfd failed");
  }
  return fd;
#else
  shim_xdna::shim_err(
      int(std::errc::not_supported),
      "Importing buffer object from different process requires XRT "
      " built and installed on a system with 'pidfd' kernel support");
#endif
}

std::string ioctl_cmd2name(unsigned long cmd) {
  switch (cmd) {
    case DRM_IOCTL_AMDXDNA_CREATE_HWCTX:
      return "DRM_IOCTL_AMDXDNA_CREATE_HWCTX";
    case DRM_IOCTL_AMDXDNA_DESTROY_HWCTX:
      return "DRM_IOCTL_AMDXDNA_DESTROY_HWCTX";
    case DRM_IOCTL_AMDXDNA_CONFIG_HWCTX:
      return "DRM_IOCTL_AMDXDNA_CONFIG_HWCTX";
    case DRM_IOCTL_AMDXDNA_CREATE_BO:
      return "DRM_IOCTL_AMDXDNA_CREATE_BO";
    case DRM_IOCTL_AMDXDNA_GET_BO_INFO:
      return "DRM_IOCTL_AMDXDNA_GET_BO_INFO";
    case DRM_IOCTL_AMDXDNA_SYNC_BO:
      return "DRM_IOCTL_AMDXDNA_SYNC_BO";
    case DRM_IOCTL_AMDXDNA_EXEC_CMD:
      return "DRM_IOCTL_AMDXDNA_EXEC_CMD";
    case DRM_IOCTL_AMDXDNA_WAIT_CMD:
      return "DRM_IOCTL_AMDXDNA_WAIT_CMD";
    case DRM_IOCTL_AMDXDNA_GET_INFO:
      return "DRM_IOCTL_AMDXDNA_GET_INFO";
    case DRM_IOCTL_AMDXDNA_SET_STATE:
      return "DRM_IOCTL_AMDXDNA_SET_STATE";
    case DRM_IOCTL_GEM_CLOSE:
      return "DRM_IOCTL_GEM_CLOSE";
    case DRM_IOCTL_PRIME_HANDLE_TO_FD:
      return "DRM_IOCTL_PRIME_HANDLE_TO_FD";
    case DRM_IOCTL_PRIME_FD_TO_HANDLE:
      return "DRM_IOCTL_PRIME_FD_TO_HANDLE";
    case DRM_IOCTL_SYNCOBJ_CREATE:
      return "DRM_IOCTL_SYNCOBJ_CREATE";
    case DRM_IOCTL_SYNCOBJ_QUERY:
      return "DRM_IOCTL_SYNCOBJ_QUERY";
    case DRM_IOCTL_SYNCOBJ_DESTROY:
      return "DRM_IOCTL_SYNCOBJ_DESTROY";
    case DRM_IOCTL_SYNCOBJ_HANDLE_TO_FD:
      return "DRM_IOCTL_SYNCOBJ_HANDLE_TO_FD";
    case DRM_IOCTL_SYNCOBJ_FD_TO_HANDLE:
      return "DRM_IOCTL_SYNCOBJ_FD_TO_HANDLE";
    case DRM_IOCTL_SYNCOBJ_TIMELINE_SIGNAL:
      return "DRM_IOCTL_SYNCOBJ_TIMELINE_SIGNAL";
    case DRM_IOCTL_SYNCOBJ_TIMELINE_WAIT:
      return "DRM_IOCTL_SYNCOBJ_TIMELINE_WAIT";
    default:
      return "UNKNOWN(" + std::to_string(cmd) + ")";
  }
  return "UNKNOWN(" + std::to_string(cmd) + ")";
}

// Device memory heap needs to be within one 64MB page. The maximum size is
// 64MB.
const size_t dev_mem_size = (64 << 20);
}  // namespace

namespace shim_xdna {

pdev::pdev() {
  const std::lock_guard<std::mutex> lock(m_lock);
  // TODO(max): hardcoded
  m_dev_fd = ::open("/dev/accel/accel0", O_RDWR);
  if (m_dev_fd < 0) shim_err(EINVAL, "Failed to open KMQ device");
  SHIM_DEBUG("Device opened, fd=%d", m_dev_fd);
  m_dev_heap_bo =
      std::make_unique<bo>(*this, dev_mem_size, AMDXDNA_BO_DEV_HEAP);
  SHIM_DEBUG("Created KMQ pcidev");
}

pdev::~pdev() {
  SHIM_DEBUG("Destroying KMQ pcidev");
  const std::lock_guard<std::mutex> lock(m_lock);
  m_dev_heap_bo.reset();
  ::close(m_dev_fd);
  SHIM_DEBUG("Device closed, fd=%d", m_dev_fd);
  SHIM_DEBUG("Destroyed KMQ pcidev");
}

void pdev::ioctl(unsigned long cmd, void *arg) const {
  if (::ioctl(m_dev_fd, cmd, arg) == -1) {
    shim_err(errno, "%s IOCTL failed", ioctl_cmd2name(cmd).c_str());
  }
}

void *pdev::mmap(void *addr, size_t len, int prot, int flags,
                 off_t offset) const {
  void *ret = ::mmap(addr, len, prot, flags, m_dev_fd, offset);
  if (ret == reinterpret_cast<void *>(-1))
    shim_err(errno,
             "mmap(addr=%p, len=%ld, prot=%d, flags=%d, offset=%ld) failed",
             addr, len, prot, flags, offset);
  return ret;
}

device::device(uint32_t n_rows, uint32_t n_cols)
    : n_rows(n_rows), n_cols(n_cols) {
  SHIM_DEBUG("Created KMQ device n_rows %d n_cols %d", n_rows, n_cols);
}

device::device(uint32_t n_rows, uint32_t n_cols,
               amdxdna_power_mode_type power_mode)
    : device(n_rows, n_cols) {
  set_power_mode(power_mode);
  SHIM_DEBUG("Created KMQ device n_rows %d n_cols %d with power_mode %s",
             n_rows, n_cols,
             stringify_amdxdna_power_mode_type(power_mode).c_str());
}

device::~device() { SHIM_DEBUG("Destroying KMQ device"); }

const pdev &device::get_pdev() const { return m_pdev; }

std::unique_ptr<hw_ctx> device::create_hw_context(
    const std::vector<uint8_t> &pdi, const std::string &cu_name,
    const std::map<std::string, uint32_t> &qos) {
  return std::make_unique<hw_ctx>(*this, pdi, cu_name, n_rows, n_cols, qos);
}

std::unique_ptr<hw_ctx> device::create_hw_context(
    const std::vector<uint8_t> &pdi, const std::string &cu_name) {
  return std::make_unique<hw_ctx>(*this, pdi, cu_name, n_rows, n_cols);
}

std::unique_ptr<bo> device::alloc_bo(uint32_t ctx_id, size_t size,
                                     shim_xcl_bo_flags flags) {
  return std::make_unique<bo>(this->m_pdev, ctx_id, size, flags);
}

std::unique_ptr<bo> device::alloc_bo(size_t size, shim_xcl_bo_flags flags) {
  return alloc_bo(AMDXDNA_INVALID_CTX_HANDLE, size, flags);
}

std::unique_ptr<bo> device::alloc_bo(size_t size, uint32_t flags) {
  return alloc_bo(AMDXDNA_INVALID_CTX_HANDLE, size,
                  shim_xcl_bo_flags{.flags = flags});
}

std::unique_ptr<bo> device::import_bo(pid_t pid, int ehdl) {
  return import_bo(import_fd(pid, ehdl));
}

std::unique_ptr<fence_handle> device::create_fence(fence_handle::access_mode) {
  return std::make_unique<fence_handle>(*this);
}

std::unique_ptr<fence_handle> device::import_fence(pid_t pid, int ehdl) {
  return std::make_unique<fence_handle>(*this, import_fd(pid, ehdl));
}

std::unique_ptr<bo> device::import_bo(int ehdl) const {
  return std::make_unique<bo>(this->m_pdev, ehdl);
}

std::vector<char> device::read_aie_mem(uint16_t col, uint16_t row,
                                       uint32_t offset, uint32_t size) {
  amdxdna_drm_aie_mem mem{};
  std::vector<char> store_buf(size);
  mem.col = col;
  mem.row = row;
  mem.addr = offset;
  mem.size = size;
  mem.buf_p = reinterpret_cast<uintptr_t>(store_buf.data());
  amdxdna_drm_get_info arg = {.param = DRM_AMDXDNA_READ_AIE_MEM,
                              .buffer_size = sizeof(mem),
                              .buffer = reinterpret_cast<uintptr_t>(&mem)};
  m_pdev.ioctl(DRM_IOCTL_AMDXDNA_GET_INFO, &arg);
  return store_buf;
}

uint32_t device::read_aie_reg(uint16_t col, uint16_t row, uint32_t reg_addr) {
  amdxdna_drm_aie_reg reg{};
  reg.col = col;
  reg.row = row;
  reg.addr = reg_addr;
  reg.val = 0;
  amdxdna_drm_get_info arg = {.param = DRM_AMDXDNA_READ_AIE_REG,
                              .buffer_size = sizeof(reg),
                              .buffer = reinterpret_cast<uintptr_t>(&reg)};
  m_pdev.ioctl(DRM_IOCTL_AMDXDNA_GET_INFO, &arg);
  return reg.val;
}

size_t device::write_aie_mem(uint16_t col, uint16_t row, uint32_t offset,
                             const std::vector<char> &buf) {
  amdxdna_drm_aie_mem mem{};
  uint32_t size = static_cast<uint32_t>(buf.size());
  mem.col = col;
  mem.row = row;
  mem.addr = offset;
  mem.size = size;
  mem.buf_p = reinterpret_cast<uintptr_t>(buf.data());
  amdxdna_drm_get_info arg = {.param = DRM_AMDXDNA_WRITE_AIE_MEM,
                              .buffer_size = sizeof(mem),
                              .buffer = reinterpret_cast<uintptr_t>(&mem)};
  m_pdev.ioctl(DRM_IOCTL_AMDXDNA_SET_STATE, &arg);
  return size;
}

void device::write_aie_reg(uint16_t col, uint16_t row, uint32_t reg_addr,
                           uint32_t reg_val) {
  amdxdna_drm_aie_reg reg{};
  reg.col = col;
  reg.row = row;
  reg.addr = reg_addr;
  reg.val = reg_val;
  amdxdna_drm_get_info arg = {.param = DRM_AMDXDNA_WRITE_AIE_REG,
                              .buffer_size = sizeof(reg),
                              .buffer = reinterpret_cast<uintptr_t>(&reg)};
  m_pdev.ioctl(DRM_IOCTL_AMDXDNA_SET_STATE, &arg);
}

amdxdna_power_mode_type device::get_power_mode() const {
  amdxdna_drm_get_power_mode state;
  amdxdna_drm_get_info arg = {.param = DRM_AMDXDNA_GET_POWER_MODE,
                              .buffer_size = sizeof(state),
                              .buffer = reinterpret_cast<uintptr_t>(&state)};

  m_pdev.ioctl(DRM_IOCTL_AMDXDNA_GET_INFO, &arg);
  return static_cast<amdxdna_power_mode_type>(state.power_mode);
}

void device::set_power_mode(amdxdna_power_mode_type mode) const {
  amdxdna_drm_set_power_mode state;
  state.power_mode = mode;
  amdxdna_drm_set_state arg = {.param = DRM_AMDXDNA_SET_POWER_MODE,
                               .buffer_size = sizeof(state),
                               .buffer = reinterpret_cast<uintptr_t>(&state)};
  if (::ioctl(m_pdev.m_dev_fd, DRM_IOCTL_AMDXDNA_SET_STATE, &arg) == -1) {
    shim_err(
        errno,
        "DRM_AMDXDNA_SET_POWER_MODE failed; probably you need sudo privileges");
  }
  SHIM_DEBUG("set power_mode to %s",
             stringify_amdxdna_power_mode_type(mode).c_str());
}

std::string read_sysfs(const std::string &filename) {
  std::ifstream file(filename);
  std::string line;
  if (file.is_open()) {
    std::getline(file, line);
    file.close();
  } else {
    std::cerr << "Error opening file: " << filename << std::endl;
    line = "";
  }
  return line;
}

std::filesystem::path find_npu_device() {
  const std::filesystem::path drvpath = "/sys/bus/pci/drivers/amdxdna";
  for (auto const &dir_entry : std::filesystem::directory_iterator{drvpath})
    if (dir_entry.is_symlink()) {
      std::cout << dir_entry.path() << '\n';
      auto actual_path = drvpath / std::filesystem::read_symlink(dir_entry);
      auto rel = std::filesystem::relative(actual_path, "/sys/devices");
      if (!rel.empty() && rel.native()[0] != '.') return absolute(actual_path);
    }
  shim_err(errno, "No npu device found");
}

std::string stringify_amdxdna_power_mode_type(
    amdxdna_power_mode_type power_mode) {
  switch (power_mode) {
    case POWER_MODE_DEFAULT:
      return {"DEFAULT"};
    case POWER_MODE_LOW:
      return {"LOW"};
    case POWER_MODE_MEDIUM:
      return {"MEDIUM"};
    case POWER_MODE_HIGH:
      return {"HIGH"};
    case POWER_MODE_TURBO:
      return {"TURBO"};
    default:
      llvm::report_fatal_error("unknown power mode");
  }
}

}  // namespace shim_xdna
