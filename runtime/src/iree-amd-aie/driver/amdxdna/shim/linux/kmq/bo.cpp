// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2024, Advanced Micro Devices, Inc. All rights reserved.

#include "bo.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <x86intrin.h>

#include <unordered_set>

#include "shim_debug.h"
#include "xrt_mem.h"

namespace {

int alloc_drm_bo(const shim_xdna::pdev &dev, amdxdna_bo_type type, size_t size,
                 uint32_t *out_handle) {
  *out_handle = AMDXDNA_INVALID_BO_HANDLE;
  amdxdna_drm_create_bo cbo = {
      .flags = 0,
      .vaddr = reinterpret_cast<uintptr_t>(nullptr),
      .size = size,
      .type = static_cast<uint32_t>(type),
  };
  int err = dev.try_ioctl(DRM_IOCTL_AMDXDNA_CREATE_BO, &cbo);
  if (err) return err;
  *out_handle = cbo.handle;
  return 0;
}

void free_drm_bo(const shim_xdna::pdev &dev, uint32_t boh) {
  drm_gem_close close_bo = {boh, 0};
  (void)dev.try_ioctl(DRM_IOCTL_GEM_CLOSE, &close_bo);
}

int get_drm_bo_info(const shim_xdna::pdev &dev, uint32_t boh,
                    amdxdna_drm_get_bo_info *bo_info) {
  bo_info->handle = boh;
  return dev.try_ioctl(DRM_IOCTL_AMDXDNA_GET_BO_INFO, bo_info);
}

int map_parent_range(size_t size, void **out_ptr) {
  *out_ptr = nullptr;
  auto p = ::mmap(nullptr, size, PROT_READ | PROT_WRITE,
                  MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (p == MAP_FAILED) return errno;

  *out_ptr = p;
  return 0;
}

int map_drm_bo(const shim_xdna::pdev &dev, size_t size, int prot,
               uint64_t offset, void **out_ptr) {
  return dev.try_mmap(nullptr, size, prot, MAP_SHARED | MAP_LOCKED, offset,
                      out_ptr);
}

int map_drm_bo(const shim_xdna::pdev &dev, void *addr, size_t size, int prot,
               int flags, uint64_t offset, void **out_ptr) {
  return dev.try_mmap(addr, size, prot, flags, offset, out_ptr);
}

void unmap_drm_bo(const shim_xdna::pdev &dev, void *addr, size_t size) {
  munmap(addr, size);
}

int attach_dbg_drm_bo(const shim_xdna::pdev &dev, uint32_t boh,
                      uint32_t ctx_id) {
  amdxdna_drm_config_hwctx adbo = {
      .handle = ctx_id,
      .param_type = DRM_AMDXDNA_HWCTX_ASSIGN_DBG_BUF,
      .param_val = boh,
  };
  return dev.try_ioctl(DRM_IOCTL_AMDXDNA_CONFIG_HWCTX, &adbo);
}

int detach_dbg_drm_bo(const shim_xdna::pdev &dev, uint32_t boh,
                      uint32_t ctx_id) {
  amdxdna_drm_config_hwctx adbo = {
      .handle = ctx_id,
      .param_type = DRM_AMDXDNA_HWCTX_REMOVE_DBG_BUF,
      .param_val = boh,
  };
  return dev.try_ioctl(DRM_IOCTL_AMDXDNA_CONFIG_HWCTX, &adbo);
}

int export_drm_bo(const shim_xdna::pdev &dev, uint32_t boh, int *out_fd) {
  *out_fd = -1;
  drm_prime_handle exp_bo = {boh, DRM_RDWR | DRM_CLOEXEC, -1};
  int err = dev.try_ioctl(DRM_IOCTL_PRIME_HANDLE_TO_FD, &exp_bo);
  if (err) return err;
  *out_fd = exp_bo.fd;
  return 0;
}

int import_drm_bo(const shim_xdna::pdev &dev,
                  const shim_xdna::shared_handle &share, amdxdna_bo_type *type,
                  size_t *size, uint32_t *out_handle) {
  *out_handle = AMDXDNA_INVALID_BO_HANDLE;
  int fd = share.get_export_handle();
  drm_prime_handle imp_bo = {AMDXDNA_INVALID_BO_HANDLE, 0, fd};
  int err = dev.try_ioctl(DRM_IOCTL_PRIME_FD_TO_HANDLE, &imp_bo);
  if (err) return err;

  *type = AMDXDNA_BO_SHMEM;
  off_t end = lseek(fd, 0, SEEK_END);
  if (end < 0) return errno;
  if (lseek(fd, 0, SEEK_SET) < 0) return errno;
  *size = end;

  *out_handle = imp_bo.handle;
  return 0;
}

bool is_power_of_two(size_t x) { return x > 0 && (x & (x - 1)) == 0; }

int addr_align(void *p, size_t align, void **out_ptr) {
  *out_ptr = nullptr;
  if (!is_power_of_two(align)) return EINVAL;

  *out_ptr = reinterpret_cast<void *>(((uintptr_t)p + align) & ~(align - 1));
  return 0;
}

amdxdna_bo_type flag_to_type(shim_xcl_bo_flags flags) {
  uint32_t boflags = (static_cast<uint32_t>(flags.boflags) << 24);
  switch (boflags) {
    case XCL_BO_FLAGS_NONE:
    case XCL_BO_FLAGS_HOST_ONLY:
      return AMDXDNA_BO_SHMEM;
    case XCL_BO_FLAGS_CACHEABLE:
      return AMDXDNA_BO_DEV;
    case XCL_BO_FLAGS_EXECBUF:
      return AMDXDNA_BO_CMD;
    default:
      break;
  }
  return AMDXDNA_BO_INVALID;
}

// flash cache line for non coherence memory
inline int clflush_data(const void *base, size_t offset, size_t len) {
  if (len == 0) return 0;
  static long cacheline_size = 0;

  if (!cacheline_size) {
    long sz = sysconf(_SC_LEVEL1_DCACHE_LINESIZE);
    if (sz <= 0) return EINVAL;
    cacheline_size = sz;
  }

  const char *cur = (const char *)base;
  cur += offset;
  uintptr_t lastline = (uintptr_t)(cur + len - 1) | (cacheline_size - 1);
  do {
    _mm_clflush(cur);
    cur += cacheline_size;
  } while (cur <= (const char *)lastline);
  return 0;
}

int sync_drm_bo(const shim_xdna::pdev &dev, uint32_t boh,
                shim_xdna::direction dir, size_t offset, size_t len) {
  amdxdna_drm_sync_bo sbo = {
      .handle = boh,
      .direction =
          (dir == shim_xdna::direction::host2device ? SYNC_DIRECT_TO_DEVICE
                                                    : SYNC_DIRECT_FROM_DEVICE),
      .offset = offset,
      .size = len,
  };
  return dev.try_ioctl(DRM_IOCTL_AMDXDNA_SYNC_BO, &sbo);
}

bool is_driver_sync() {
  static int drv_sync = -1;

  if (drv_sync == -1) {
    bool ds = std::getenv("Debug.force_driver_sync");
    drv_sync = ds ? 1 : 0;
  }
  return drv_sync == 1;
}

}  // namespace

namespace shim_xdna {

drm_bo::drm_bo(bo &parent, const amdxdna_drm_get_bo_info &bo_info)
    : m_parent(parent),
      m_handle(bo_info.handle),
      m_map_offset(bo_info.map_offset),
      m_xdna_addr(bo_info.xdna_addr),
      m_vaddr(bo_info.vaddr) {}

drm_bo::~drm_bo() {
  if (m_handle == AMDXDNA_INVALID_BO_HANDLE) return;
  free_drm_bo(m_parent.m_pdev, m_handle);
}

std::string bo::type_to_name() const {
  switch (m_type) {
    case AMDXDNA_BO_SHMEM:
      return {"AMDXDNA_BO_SHMEM"};
    case AMDXDNA_BO_DEV_HEAP:
      return {"AMDXDNA_BO_DEV_HEAP"};
    case AMDXDNA_BO_DEV:
      if (shim_xcl_bo_flags{m_flags}.use == XRT_BO_USE_DEBUG)
        return {"AMDXDNA_BO_DEV_DEBUG"};
      return {"AMDXDNA_BO_DEV"};
    case AMDXDNA_BO_CMD:
      return {"AMDXDNA_BO_CMD"};
    default:;
      return {"BO_UNKNOWN"};
  }
  return {"BO_UNKNOWN"};
}

std::string bo::describe() const {
  std::string desc = "type=";
  desc += type_to_name();
  desc += ", ";
  desc += "drm_bo=";
  desc += std::to_string(get_drm_bo_handle());
  desc += ", ";
  desc += "size=";
  desc += std::to_string(m_aligned_size);
  return desc;
}

int bo::mmap_bo(size_t align) {
  size_t a = align;
  if (!m_drm_bo) return EINVAL;

  if (m_drm_bo->m_map_offset == AMDXDNA_INVALID_ADDR) {
    m_aligned = reinterpret_cast<void *>(m_drm_bo->m_vaddr);
    return 0;
  }

  if (a == 0) {
    return map_drm_bo(m_pdev, m_aligned_size, PROT_READ | PROT_WRITE,
                      m_drm_bo->m_map_offset, &m_aligned);
  }

  /*
   * Handle special alignment
   * The first mmap() is just for reserved a range in user vritual address
   * space. The second mmap() uses an aligned addr as the first argument in mmap
   * syscall.
   */
  m_parent_size = align * 2 - 1;
  int err = map_parent_range(m_parent_size, &m_parent);
  if (err) return err;
  void *aligned = nullptr;
  err = addr_align(m_parent, align, &aligned);
  if (err) return err;
  return map_drm_bo(m_pdev, aligned, m_aligned_size, PROT_READ | PROT_WRITE,
                    MAP_SHARED | MAP_FIXED, m_drm_bo->m_map_offset, &m_aligned);
}

void bo::munmap_bo() {
  SHIM_DEBUG("Unmap BO, aligned %p parent %p", m_aligned, m_parent);
  if (!m_drm_bo) return;
  if (m_drm_bo->m_map_offset == AMDXDNA_INVALID_ADDR) return;

  if (m_aligned) unmap_drm_bo(m_pdev, m_aligned, m_aligned_size);
  if (m_parent) unmap_drm_bo(m_pdev, m_parent, m_parent_size);
}

int bo::import_bo() {
  uint32_t boh = AMDXDNA_INVALID_BO_HANDLE;
  int err = import_drm_bo(m_pdev, m_import, &m_type, &m_aligned_size, &boh);
  if (err) return err;

  amdxdna_drm_get_bo_info bo_info = {};
  err = get_drm_bo_info(m_pdev, boh, &bo_info);
  if (err) {
    free_drm_bo(m_pdev, boh);
    return err;
  }
  m_drm_bo = std::make_unique<drm_bo>(*this, bo_info);
  return 0;
}

void bo::free_bo() { m_drm_bo.reset(); }

bo::bo(const pdev &pdev, uint32_t ctx_id, size_t size, shim_xcl_bo_flags flags,
       amdxdna_bo_type type)
    : m_pdev(pdev),
      m_aligned_size(size),
      m_flags(flags),
      m_type(type),
      m_import(-1),
      m_owner_ctx_id(ctx_id) {
  if (m_type == AMDXDNA_BO_INVALID) {
    m_init_errno = EINVAL;
    return;
  }

  size_t align = 0;

  if (m_type == AMDXDNA_BO_DEV_HEAP)
    align = 64 * 1024 * 1024;  // Device mem heap must align at 64MB boundary.

  uint32_t boh = AMDXDNA_INVALID_BO_HANDLE;
  m_init_errno = alloc_drm_bo(m_pdev, m_type, m_aligned_size, &boh);
  if (m_init_errno) return;
  // CREATE_BO returns only a handle; GET_BO_INFO provides the map offset,
  // device address and effective size used by the rest of the shim.
  amdxdna_drm_get_bo_info bo_info = {};
  m_init_errno = get_drm_bo_info(m_pdev, boh, &bo_info);
  if (m_init_errno) {
    free_drm_bo(m_pdev, boh);
    return;
  }
  m_drm_bo = std::make_unique<drm_bo>(*this, bo_info);

  m_init_errno = mmap_bo(align);
  if (m_init_errno) return;

  // Newly allocated buffer may contain dirty pages. If used as output buffer,
  // the data in cacheline will be flushed onto memory and pollute the output
  // from device. We perform a cache flush right after the BO is allocated to
  // avoid this issue.
  if (m_type == AMDXDNA_BO_SHMEM) {
    m_init_errno = sync(direction::host2device, size, 0);
    if (m_init_errno) return;
  }

  m_init_errno = attach_to_ctx();
  if (m_init_errno) return;
#ifndef NDEBUG
  switch (m_flags.all) {
    case 0x0:
      SHIM_DEBUG("allocating dev heap");
      break;
    case 0x1000000:
      // pdi bo
      SHIM_DEBUG("allocating pdi bo");
      break;
    case 0x20000000:
      // XCL_BO_FLAGS_P2P in create_free_bo test
      SHIM_DEBUG("allocating XCL_BO_FLAGS_P2P");
      break;
    case 0x80000000:
      // XCL_BO_FLAGS_EXECBUF in create_free_bo test
      SHIM_DEBUG("allocating XCL_BO_FLAGS_EXECBUF");
      break;
    case 0x1001000000:
      // debug bo
      SHIM_DEBUG("allocating debug bo");
      break;
    default:
      SHIM_DEBUG("allocating BO with unrecognized flags 0x%llx", m_flags.all);
  }
#endif

  SHIM_DEBUG(
      "Allocated KMQ BO (userptr=0x%lx, size=%ld, flags=0x%llx, "
      "type=%d, drm_bo=%d)",
      m_aligned, m_aligned_size, m_flags, m_type, get_drm_bo_handle());
}

bo::bo(const pdev &p, uint32_t ctx_id, size_t size, shim_xcl_bo_flags flags)
    : bo(p, ctx_id, size, flags, flag_to_type(flags)) {}

bo::bo(const pdev &p, uint32_t ctx_id, size_t size, uint32_t flags)
    : bo(p, ctx_id, size, shim_xcl_bo_flags{.flags = flags}) {}

bo::bo(const pdev &p, int ehdl) : m_pdev(p), m_import(ehdl) {
  m_init_errno = import_bo();
  if (m_init_errno) return;
  m_init_errno = mmap_bo();
  if (m_init_errno) return;
  SHIM_DEBUG(
      "Imported KMQ BO (userptr=0x%lx, size=%ld, flags=0x%llx, type=%d, "
      "drm_bo=%d)",
      m_aligned, m_aligned_size, m_flags, m_type, get_drm_bo_handle());
}

bo::~bo() {
  SHIM_DEBUG("Freeing KMQ BO, %s", describe().c_str());

  munmap_bo();
  detach_from_ctx();
  // If BO is in use, we should block and wait in driver
  free_bo();
}

bo::bo(const pdev &p, size_t size, amdxdna_bo_type type)
    : bo(p, AMDXDNA_INVALID_CTX_HANDLE, size, shim_xcl_bo_flags{}, type) {}

int bo::create(const pdev &p, uint32_t ctx_id, size_t size,
               shim_xcl_bo_flags flags, std::unique_ptr<bo> *out_bo) {
  out_bo->reset();
  std::unique_ptr<bo> new_bo = std::make_unique<bo>(p, ctx_id, size, flags);
  int err = new_bo->init_errno();
  if (err) return err;
  *out_bo = std::move(new_bo);
  return 0;
}

int bo::create(const pdev &p, uint32_t ctx_id, size_t size, uint32_t flags,
               std::unique_ptr<bo> *out_bo) {
  return create(p, ctx_id, size, shim_xcl_bo_flags{.flags = flags}, out_bo);
}

int bo::create(const pdev &p, size_t size, amdxdna_bo_type type,
               std::unique_ptr<bo> *out_bo) {
  out_bo->reset();
  std::unique_ptr<bo> new_bo = std::make_unique<bo>(p, size, type);
  int err = new_bo->init_errno();
  if (err) return err;
  *out_bo = std::move(new_bo);
  return 0;
}

int bo::create_imported(const pdev &p, int ehdl, std::unique_ptr<bo> *out_bo) {
  out_bo->reset();
  std::unique_ptr<bo> new_bo = std::make_unique<bo>(p, ehdl);
  int err = new_bo->init_errno();
  if (err) return err;
  *out_bo = std::move(new_bo);
  return 0;
}

int bo::init_errno() const { return m_init_errno; }

properties bo::get_properties() const {
  return {m_flags, m_aligned_size, get_paddr(), get_drm_bo_handle()};
}

size_t bo::size() { return get_properties().size; }

void *bo::map() const { return m_aligned; }

void bo::unmap(void *addr) {}

uint64_t bo::get_paddr() const {
  if (m_drm_bo && m_drm_bo->m_xdna_addr != AMDXDNA_INVALID_ADDR)
    return m_drm_bo->m_xdna_addr;
  return reinterpret_cast<uintptr_t>(m_aligned);
}

void bo::set_cmd_id(uint64_t id) { m_cmd_id = id; }

uint64_t bo::get_cmd_id() const { return m_cmd_id; }

uint32_t bo::get_drm_bo_handle() const {
  return m_drm_bo ? m_drm_bo->m_handle : AMDXDNA_INVALID_BO_HANDLE;
}

int bo::attach_to_ctx() {
  if (m_owner_ctx_id == AMDXDNA_INVALID_CTX_HANDLE) return 0;
  if (!m_drm_bo) return EINVAL;

  auto boh = get_drm_bo_handle();
  SHIM_DEBUG("Attaching drm_bo %d to ctx: %d", boh, m_owner_ctx_id);
  return attach_dbg_drm_bo(m_pdev, boh, m_owner_ctx_id);
}

int bo::detach_from_ctx() {
  if (m_owner_ctx_id == AMDXDNA_INVALID_CTX_HANDLE) return 0;
  if (!m_drm_bo) return 0;

  auto boh = get_drm_bo_handle();
  SHIM_DEBUG("Detaching drm_bo %d from ctx: %d", boh, m_owner_ctx_id);
  return detach_dbg_drm_bo(m_pdev, boh, m_owner_ctx_id);
}

int bo::share(std::unique_ptr<shim_xdna::shared_handle> *out_handle) const {
  out_handle->reset();
  auto boh = get_drm_bo_handle();
  int fd = -1;
  int err = export_drm_bo(m_pdev, boh, &fd);
  if (err) return err;
  SHIM_DEBUG("Exported bo %d to fd %d", boh, fd);
  *out_handle = std::make_unique<shared_handle>(fd);
  return 0;
}

amdxdna_bo_type bo::get_type() const { return m_type; }

int bo::sync(direction dir, size_t size, size_t offset) {
  if (is_driver_sync()) {
    return sync_drm_bo(m_pdev, get_drm_bo_handle(), dir, offset, size);
  }

  if (offset > m_aligned_size || size > m_aligned_size - offset) return EINVAL;

  switch (m_type) {
    case AMDXDNA_BO_SHMEM:
    case AMDXDNA_BO_CMD:
      return clflush_data(m_aligned, offset, size);
    case AMDXDNA_BO_DEV:
      if (m_owner_ctx_id == AMDXDNA_INVALID_CTX_HANDLE)
        return clflush_data(m_aligned, offset, size);
      return sync_drm_bo(m_pdev, get_drm_bo_handle(), dir, offset, size);
    default:
      return ENOTSUP;
  }
}

int bo::sync(direction dir) { return sync(dir, size(), 0); }

int bo::bind_at(size_t pos, const bo &boh, size_t offset, size_t size) {
  std::lock_guard<std::mutex> lg(m_args_map_lock);

  if (m_type != AMDXDNA_BO_CMD) return EINVAL;
  if (offset > boh.m_aligned_size || size > boh.m_aligned_size - offset)
    return EINVAL;

  if (!pos) m_args_map.clear();

  if (boh.get_type() != AMDXDNA_BO_CMD) {
    auto h = boh.get_drm_bo_handle();
    m_args_map[pos] = h;
    SHIM_DEBUG("Added arg BO %d to cmd BO %d", h, get_drm_bo_handle());
  } else {
    const size_t max_args_order = 6;
    const size_t max_args = 1 << max_args_order;
    size_t key = pos << max_args_order;
    uint32_t hs[max_args];
    uint32_t arg_cnt = 0;
    int err = boh.get_arg_bo_handles(hs, max_args, &arg_cnt);
    if (err) return err;
    std::string bohs;
    for (int i = 0; i < arg_cnt; i++) {
      m_args_map[key + i] = hs[i];
      bohs += std::to_string(hs[i]) + " ";
    }
    SHIM_DEBUG("Added arg BO %s to cmd BO %d", bohs.c_str(),
               get_drm_bo_handle());
  }
  return 0;
}

int bo::get_arg_bo_handles(uint32_t *handles, size_t num,
                           uint32_t *out_count) const {
  std::lock_guard<std::mutex> lg(m_args_map_lock);
  *out_count = 0;

  // m_args_map is keyed by patch position, so the same BO handle can appear at
  // multiple positions when a command references one buffer more than once.
  // These handles are passed to DRM_IOCTL_AMDXDNA_EXEC_CMD, where the kernel
  // locks them via drm_gem_lock_reservations(), which rejects a duplicate GEM
  // object in the list with -EALREADY. Arg BOs are only used kernel-side for
  // reservation locking, fence tracking and HMM residency, all idempotent per
  // BO, so emit each distinct handle once.
  std::unordered_set<uint32_t> seen;
  uint32_t count = 0;
  for (auto &m : m_args_map) {
    if (!seen.insert(m.second).second) continue;
    if (count >= num) return E2BIG;
    handles[count++] = m.second;
  }

  *out_count = count;
  return 0;
}

}  // namespace shim_xdna
