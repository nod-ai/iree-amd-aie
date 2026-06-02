// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2024, Advanced Micro Devices, Inc. - All rights reserved

#include "device.h"

#include <fcntl.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <system_error>
#include <vector>

#include "amdxdna_accel.h"
#include "bo.h"
#include "fence.h"
#include "hwctx.h"
#include "shim_debug.h"
#include "xrt_mem.h"

namespace {

int import_fd_checked(pid_t pid, int ehdl, int *out_fd) {
  *out_fd = -1;
  if (pid == 0 || getpid() == pid) {
    *out_fd = ehdl;
    return 0;
  }

#if defined(SYS_pidfd_open) && defined(SYS_pidfd_getfd)
  auto pidfd = syscall(SYS_pidfd_open, pid, 0);
  if (pidfd < 0) return errno;

  int64_t fd = syscall(SYS_pidfd_getfd, pidfd, ehdl, 0);
  int saved_errno = fd < 0 ? errno : 0;
  if (close(pidfd) && saved_errno == 0) saved_errno = errno;
  if (saved_errno != 0) return saved_errno;
  *out_fd = static_cast<int>(fd);
  return 0;
#else
  return int(std::errc::not_supported);
#endif
}

// Device memory heap needs to be within one 64MB page. The maximum size is
// 64MB.
const size_t dev_mem_size = (64 << 20);

std::filesystem::path try_find_npu_device() {
  const std::filesystem::path drvpath = "/sys/bus/pci/drivers/amdxdna";
  std::error_code ec;
  for (std::filesystem::directory_iterator it(drvpath, ec), end;
       !ec && it != end; it.increment(ec)) {
    auto const &dir_entry = *it;
    std::error_code read_ec;
    if (dir_entry.is_symlink(read_ec) && !read_ec) {
      auto actual_path =
          drvpath / std::filesystem::read_symlink(dir_entry, read_ec);
      if (read_ec) continue;
      auto rel =
          std::filesystem::relative(actual_path, "/sys/devices", read_ec);
      if (read_ec) continue;
      if (!rel.empty() && rel.native()[0] != '.') {
        auto abs_path = std::filesystem::absolute(actual_path, read_ec);
        if (!read_ec) return abs_path;
      }
    }
  }
  return {};
}

std::string read_first_line(const std::filesystem::path &path) {
  std::ifstream file(path);
  std::string line;
  if (file.is_open()) std::getline(file, line);
  return line;
}

int try_ioctl_fd(int fd, unsigned long cmd, void *arg) {
  if (::ioctl(fd, cmd, arg) == -1) return errno;
  return 0;
}

amdxdna_power_mode_type to_amdxdna_power_mode(shim_xdna::power_mode mode) {
  switch (mode) {
    case shim_xdna::power_mode::default_mode:
      return POWER_MODE_DEFAULT;
    case shim_xdna::power_mode::low:
      return POWER_MODE_LOW;
    case shim_xdna::power_mode::medium:
      return POWER_MODE_MEDIUM;
    case shim_xdna::power_mode::high:
      return POWER_MODE_HIGH;
    case shim_xdna::power_mode::turbo:
      return POWER_MODE_TURBO;
  }
  return POWER_MODE_DEFAULT;
}

int from_amdxdna_power_mode(uint8_t value, shim_xdna::power_mode *out_mode) {
  switch (static_cast<amdxdna_power_mode_type>(value)) {
    case POWER_MODE_DEFAULT:
      *out_mode = shim_xdna::power_mode::default_mode;
      return 0;
    case POWER_MODE_LOW:
      *out_mode = shim_xdna::power_mode::low;
      return 0;
    case POWER_MODE_MEDIUM:
      *out_mode = shim_xdna::power_mode::medium;
      return 0;
    case POWER_MODE_HIGH:
      *out_mode = shim_xdna::power_mode::high;
      return 0;
    case POWER_MODE_TURBO:
      *out_mode = shim_xdna::power_mode::turbo;
      return 0;
    default:
      *out_mode = shim_xdna::power_mode::default_mode;
      return EINVAL;
  }
}
}  // namespace

namespace shim_xdna {

std::filesystem::path find_default_accel_device_path() {
  const std::filesystem::path accel_dir = "/dev/accel";
  std::error_code ec;
  std::vector<std::filesystem::path> candidates;
  for (std::filesystem::directory_iterator it(accel_dir, ec), end;
       !ec && it != end; it.increment(ec)) {
    const auto filename = it->path().filename().native();
    if (filename.rfind("accel", 0) == 0) candidates.push_back(it->path());
  }
  if (!candidates.empty()) {
    std::sort(candidates.begin(), candidates.end());
    return candidates.front();
  }
  return accel_dir / "accel0";
}

pdev::pdev() : pdev(find_default_accel_device_path()) {}

pdev::pdev(const std::filesystem::path &device_path) {
  m_init_errno = open_device(device_path);
}

int pdev::open_device(const std::filesystem::path &device_path) {
  const std::lock_guard<std::mutex> lock(m_lock);
  m_dev_fd = ::open(device_path.c_str(), O_RDWR | O_CLOEXEC);
  if (m_dev_fd < 0) {
    return errno;
  }
  SHIM_DEBUG("Device opened, fd=%d", m_dev_fd);
  int err =
      bo::create(*this, dev_mem_size, AMDXDNA_BO_DEV_HEAP, &m_dev_heap_bo);
  if (err) {
    ::close(m_dev_fd);
    m_dev_fd = -1;
    return err;
  }
  SHIM_DEBUG("Created KMQ pcidev");
  return 0;
}

int pdev::init_errno() const { return m_init_errno; }

pdev::~pdev() {
  SHIM_DEBUG("Destroying KMQ pcidev");
  const std::lock_guard<std::mutex> lock(m_lock);
  m_dev_heap_bo.reset();
  if (m_dev_fd >= 0) ::close(m_dev_fd);
  SHIM_DEBUG("Device closed, fd=%d", m_dev_fd);
  SHIM_DEBUG("Destroyed KMQ pcidev");
}

int pdev::try_ioctl(unsigned long cmd, void *arg) const {
  if (::ioctl(m_dev_fd, cmd, arg) == -1) {
    return errno;
  }
  return 0;
}

int pdev::try_mmap(void *addr, size_t len, int prot, int flags, off_t offset,
                   void **out_ptr) const {
  *out_ptr = nullptr;
  void *ret = ::mmap(addr, len, prot, flags, m_dev_fd, offset);
  if (ret == MAP_FAILED) return errno;
  *out_ptr = ret;
  return 0;
}

namespace {

struct core_grid_size {
  uint32_t rows;
  uint32_t cols;
};

int query_aie_metadata(const pdev &pdev,
                       amdxdna_drm_query_aie_metadata *metadata) {
  amdxdna_drm_get_info arg = {.param = DRM_AMDXDNA_QUERY_AIE_METADATA,
                              .buffer_size = sizeof(*metadata),
                              .buffer = reinterpret_cast<uintptr_t>(metadata)};
  return pdev.try_ioctl(DRM_IOCTL_AMDXDNA_GET_INFO, &arg);
}

std::string normalize_vbnv_arch(std::string vbnv) {
  if (vbnv.rfind("NPU ", 0) == 0) return vbnv.substr(4);
  constexpr const char *kRyzenAiPrefix = "RyzenAI-npu";
  if (vbnv.rfind(kRyzenAiPrefix, 0) == 0) {
    const std::string gen = vbnv.substr(strlen(kRyzenAiPrefix));
    return gen == "1" ? "Phoenix" : "Strix";
  }
  return {};
}

std::string query_npu_arch() {
  const std::filesystem::path npu_device = try_find_npu_device();
  if (npu_device.empty()) return {};
  return normalize_vbnv_arch(read_first_line(npu_device / "vbnv"));
}

int resolve_core_grid_size(const pdev &pdev, uint32_t rows, uint32_t cols,
                           core_grid_size *out_grid) {
  if (rows != 0 && cols != 0) {
    *out_grid = {rows, cols};
    return 0;
  }

  amdxdna_drm_query_aie_metadata metadata{};
  int err = query_aie_metadata(pdev, &metadata);
  if (err) return err;
  if (rows == 0) rows = metadata.core.row_count;
  if (cols == 0) {
    cols = metadata.cols;
    if (query_npu_arch() == "Phoenix" && cols != 0) --cols;
  }
  if (rows == 0 || cols == 0) return EINVAL;
  *out_grid = {rows, cols};
  return 0;
}

}  // namespace

int resolve_core_grid_size(const std::filesystem::path &device_path,
                           uint32_t requested_rows, uint32_t requested_cols,
                           uint32_t *out_rows, uint32_t *out_cols) {
  if (requested_rows != 0 && requested_cols != 0) {
    *out_rows = requested_rows;
    *out_cols = requested_cols;
    return 0;
  }

  const std::filesystem::path resolved_device_path =
      device_path.empty() ? find_default_accel_device_path() : device_path;
  const int fd = ::open(resolved_device_path.c_str(), O_RDWR | O_CLOEXEC);
  if (fd < 0) return errno;

  amdxdna_drm_query_aie_metadata metadata{};
  amdxdna_drm_get_info arg = {.param = DRM_AMDXDNA_QUERY_AIE_METADATA,
                              .buffer_size = sizeof(metadata),
                              .buffer = reinterpret_cast<uintptr_t>(&metadata)};
  const int err = try_ioctl_fd(fd, DRM_IOCTL_AMDXDNA_GET_INFO, &arg);
  ::close(fd);
  if (err != 0) return err;

  uint32_t rows = requested_rows;
  uint32_t cols = requested_cols;
  if (rows == 0) rows = metadata.core.row_count;
  if (cols == 0) {
    cols = metadata.cols;
    if (query_npu_arch() == "Phoenix" && cols != 0) --cols;
  }
  if (rows == 0 || cols == 0) return EINVAL;

  *out_rows = rows;
  *out_cols = cols;
  return 0;
}

device::device(uint32_t n_rows, uint32_t n_cols)
    : device(n_rows, n_cols, std::filesystem::path()) {}

device::device(uint32_t n_rows, uint32_t n_cols,
               const std::filesystem::path &device_path)
    : m_pdev(device_path.empty() ? find_default_accel_device_path()
                                 : device_path),
      n_rows(n_rows),
      n_cols(n_cols) {
  m_init_errno = m_pdev.init_errno();
  if (m_init_errno) return;
  core_grid_size grid{};
  m_init_errno =
      resolve_core_grid_size(m_pdev, this->n_rows, this->n_cols, &grid);
  if (m_init_errno) return;
  this->n_rows = grid.rows;
  this->n_cols = grid.cols;
  SHIM_DEBUG("Created KMQ device n_rows %d n_cols %d", this->n_rows,
             this->n_cols);
}

device::device(uint32_t n_rows, uint32_t n_cols, power_mode mode)
    : device(n_rows, n_cols, std::filesystem::path(), mode) {}

device::device(uint32_t n_rows, uint32_t n_cols,
               const std::filesystem::path &device_path, power_mode mode)
    : device(n_rows, n_cols, device_path) {
  const int err = set_power_mode(mode);
  if (err != 0) {
    SHIM_DEBUG("Unable to set power_mode %s, errno=%d",
               stringify_power_mode(mode).c_str(), err);
  }
  SHIM_DEBUG("Created KMQ device n_rows %d n_cols %d with power_mode %s",
             n_rows, n_cols, stringify_power_mode(mode).c_str());
}

device::~device() { SHIM_DEBUG("Destroying KMQ device"); }

int device::create(uint32_t n_rows, uint32_t n_cols,
                   const std::filesystem::path &device_path,
                   std::unique_ptr<device> *out_device) {
  out_device->reset();
  std::unique_ptr<device> new_device =
      std::make_unique<device>(n_rows, n_cols, device_path);
  int err = new_device->init_errno();
  if (err) return err;
  *out_device = std::move(new_device);
  return 0;
}

int device::init_errno() const { return m_init_errno; }

const pdev &device::get_pdev() const { return m_pdev; }

int device::create_hw_context(const std::vector<uint8_t> &pdi,
                              const std::string &cu_name,
                              const std::map<std::string, uint32_t> &qos,
                              std::unique_ptr<hw_ctx> *out_context) {
  out_context->reset();
  std::unique_ptr<hw_ctx> new_context =
      std::make_unique<hw_ctx>(*this, pdi, cu_name, n_rows, n_cols, qos);
  int err = new_context->init_errno();
  if (err) return err;
  *out_context = std::move(new_context);
  return 0;
}

int device::create_hw_context(const std::vector<uint8_t> &pdi,
                              const std::string &cu_name,
                              std::unique_ptr<hw_ctx> *out_context) {
  return create_hw_context(pdi, cu_name, {}, out_context);
}

int device::alloc_bo(uint32_t ctx_id, size_t size, shim_xcl_bo_flags flags,
                     std::unique_ptr<bo> *out_bo) {
  return bo::create(this->m_pdev, ctx_id, size, flags, out_bo);
}

int device::alloc_bo(size_t size, shim_xcl_bo_flags flags,
                     std::unique_ptr<bo> *out_bo) {
  return alloc_bo(AMDXDNA_INVALID_CTX_HANDLE, size, flags, out_bo);
}

int device::alloc_bo(size_t size, uint32_t flags, std::unique_ptr<bo> *out_bo) {
  return alloc_bo(AMDXDNA_INVALID_CTX_HANDLE, size,
                  shim_xcl_bo_flags{.flags = flags}, out_bo);
}

int device::create_fence(fence_handle::access_mode,
                         std::unique_ptr<fence_handle> *out_fence) {
  return fence_handle::create(*this, out_fence);
}

int device::import_fence(pid_t pid, int ehdl,
                         std::unique_ptr<fence_handle> *out_fence) {
  int fd = -1;
  int err = import_fd_checked(pid, ehdl, &fd);
  if (err) return err;
  return fence_handle::create_imported(*this, fd, out_fence);
}

int device::read_aie_mem(uint16_t col, uint16_t row, uint32_t offset,
                         uint32_t size, std::vector<char> *out_buf) {
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
  int err = m_pdev.try_ioctl(DRM_IOCTL_AMDXDNA_GET_INFO, &arg);
  if (err) return err;
  *out_buf = std::move(store_buf);
  return 0;
}

int device::read_aie_reg(uint16_t col, uint16_t row, uint32_t reg_addr,
                         uint32_t *out_reg_val) {
  amdxdna_drm_aie_reg reg{};
  reg.col = col;
  reg.row = row;
  reg.addr = reg_addr;
  reg.val = 0;
  amdxdna_drm_get_info arg = {.param = DRM_AMDXDNA_READ_AIE_REG,
                              .buffer_size = sizeof(reg),
                              .buffer = reinterpret_cast<uintptr_t>(&reg)};
  int err = m_pdev.try_ioctl(DRM_IOCTL_AMDXDNA_GET_INFO, &arg);
  if (err) return err;
  *out_reg_val = reg.val;
  return 0;
}

int device::write_aie_mem(uint16_t col, uint16_t row, uint32_t offset,
                          const std::vector<char> &buf, size_t *out_size) {
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
  int err = m_pdev.try_ioctl(DRM_IOCTL_AMDXDNA_SET_STATE, &arg);
  if (err) return err;
  *out_size = size;
  return 0;
}

int device::write_aie_reg_checked(uint16_t col, uint16_t row, uint32_t reg_addr,
                                  uint32_t reg_val) {
  amdxdna_drm_aie_reg reg{};
  reg.col = col;
  reg.row = row;
  reg.addr = reg_addr;
  reg.val = reg_val;
  amdxdna_drm_get_info arg = {.param = DRM_AMDXDNA_WRITE_AIE_REG,
                              .buffer_size = sizeof(reg),
                              .buffer = reinterpret_cast<uintptr_t>(&reg)};
  return m_pdev.try_ioctl(DRM_IOCTL_AMDXDNA_SET_STATE, &arg);
}

int device::get_power_mode(power_mode *out_mode) const {
  amdxdna_drm_get_power_mode state;
  amdxdna_drm_get_info arg = {.param = DRM_AMDXDNA_GET_POWER_MODE,
                              .buffer_size = sizeof(state),
                              .buffer = reinterpret_cast<uintptr_t>(&state)};

  int err = m_pdev.try_ioctl(DRM_IOCTL_AMDXDNA_GET_INFO, &arg);
  if (err) return err;
  return from_amdxdna_power_mode(state.power_mode, out_mode);
}

int device::set_power_mode(power_mode mode) const {
  amdxdna_drm_set_power_mode state;
  state.power_mode = to_amdxdna_power_mode(mode);
  amdxdna_drm_set_state arg = {.param = DRM_AMDXDNA_SET_POWER_MODE,
                               .buffer_size = sizeof(state),
                               .buffer = reinterpret_cast<uintptr_t>(&state)};
  const int err = m_pdev.try_ioctl(DRM_IOCTL_AMDXDNA_SET_STATE, &arg);
  if (err != 0) return err;
  SHIM_DEBUG("set power_mode to %s", stringify_power_mode(mode).c_str());
  return 0;
}

std::string read_sysfs(const std::string &filename) {
  return read_first_line(filename);
}

std::string stringify_power_mode(power_mode mode) {
  switch (mode) {
    case power_mode::default_mode:
      return {"DEFAULT"};
    case power_mode::low:
      return {"LOW"};
    case power_mode::medium:
      return {"MEDIUM"};
    case power_mode::high:
      return {"HIGH"};
    case power_mode::turbo:
      return {"TURBO"};
  }
  return "UNKNOWN(" + std::to_string(static_cast<int>(mode)) + ")";
}

}  // namespace shim_xdna
