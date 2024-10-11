// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2024, Advanced Micro Devices, Inc. All rights reserved.

#include "bo.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <x86intrin.h>

#include <cstring>
#include <iostream>

#include "shim_debug.h"
#include "xrt_mem.h"

namespace {

uint32_t alloc_drm_bo(const shim_xdna::pdev &dev, amdxdna_bo_type type,
                      size_t size) {
  amdxdna_drm_create_bo cbo = {
      .type = static_cast<uint32_t>(type),
      .vaddr = reinterpret_cast<uintptr_t>(nullptr),
      .size = size,
  };
  dev.ioctl(DRM_IOCTL_AMDXDNA_CREATE_BO, &cbo);
  return cbo.handle;
}

void free_drm_bo(const shim_xdna::pdev &dev, uint32_t boh) {
  drm_gem_close close_bo = {boh, 0};
  dev.ioctl(DRM_IOCTL_GEM_CLOSE, &close_bo);
}

void get_drm_bo_info(const shim_xdna::pdev &dev, uint32_t boh,
                     amdxdna_drm_get_bo_info *bo_info) {
  bo_info->handle = boh;
  dev.ioctl(DRM_IOCTL_AMDXDNA_GET_BO_INFO, bo_info);
}

void *map_parent_range(size_t size) {
  auto p = ::mmap(nullptr, size, PROT_READ | PROT_WRITE,
                  MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (!p) shim_xdna::shim_err(errno, "mmap(len=%ld) failed", size);

  return p;
}

void *map_drm_bo(const shim_xdna::pdev &dev, size_t size, int prot,
                 uint64_t offset) {
  return dev.mmap(nullptr, size, prot, MAP_SHARED | MAP_LOCKED, offset);
}

void *map_drm_bo(const shim_xdna::pdev &dev, void *addr, size_t size, int prot,
                 int flags, uint64_t offset) {
  return dev.mmap(addr, size, prot, flags, offset);
}

void unmap_drm_bo(const shim_xdna::pdev &dev, void *addr, size_t size) {
  munmap(addr, size);
}

void attach_dbg_drm_bo(const shim_xdna::pdev &dev, uint32_t boh,
                       uint32_t ctx_id) {
  amdxdna_drm_config_hwctx adbo = {
      .handle = ctx_id,
      .param_type = DRM_AMDXDNA_HWCTX_ASSIGN_DBG_BUF,
      .param_val = boh,
  };
  dev.ioctl(DRM_IOCTL_AMDXDNA_CONFIG_HWCTX, &adbo);
}

void detach_dbg_drm_bo(const shim_xdna::pdev &dev, uint32_t boh,
                       uint32_t ctx_id) {
  amdxdna_drm_config_hwctx adbo = {
      .handle = ctx_id,
      .param_type = DRM_AMDXDNA_HWCTX_REMOVE_DBG_BUF,
      .param_val = boh,
  };
  dev.ioctl(DRM_IOCTL_AMDXDNA_CONFIG_HWCTX, &adbo);
}

int export_drm_bo(const shim_xdna::pdev &dev, uint32_t boh) {
  drm_prime_handle exp_bo = {boh, DRM_RDWR | DRM_CLOEXEC, -1};
  dev.ioctl(DRM_IOCTL_PRIME_HANDLE_TO_FD, &exp_bo);
  return exp_bo.fd;
}

uint32_t import_drm_bo(const shim_xdna::pdev &dev,
                       const shim_xdna::shared_handle &share,
                       amdxdna_bo_type *type, size_t *size) {
  int fd = share.get_export_handle();
  drm_prime_handle imp_bo = {AMDXDNA_INVALID_BO_HANDLE, 0, fd};
  dev.ioctl(DRM_IOCTL_PRIME_FD_TO_HANDLE, &imp_bo);

  *type = AMDXDNA_BO_SHMEM;
  *size = lseek(fd, 0, SEEK_END);
  lseek(fd, 0, SEEK_SET);

  return imp_bo.handle;
}

bool is_power_of_two(size_t x) { return x > 0 && (x & x - 1) == 0; }

void *addr_align(void *p, size_t align) {
  if (!is_power_of_two(align))
    shim_xdna::shim_err(EINVAL, "Alignment 0x%lx is not power of two", align);

  return reinterpret_cast<void *>((uintptr_t)p + align & ~(align - 1));
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
inline void clflush_data(const void *base, size_t offset, size_t len) {
  static long cacheline_size = 0;

  if (!cacheline_size) {
    long sz = sysconf(_SC_LEVEL1_DCACHE_LINESIZE);
    if (sz <= 0)
      shim_xdna::shim_err(EINVAL, "Invalid cache line size: %ld", sz);
    cacheline_size = sz;
  }

  const char *cur = (const char *)base;
  cur += offset;
  uintptr_t lastline = (uintptr_t)(cur + len - 1) | (cacheline_size - 1);
  do {
    _mm_clflush(cur);
    cur += cacheline_size;
  } while (cur <= (const char *)lastline);
}

void sync_drm_bo(const shim_xdna::pdev &dev, uint32_t boh,
                 shim_xdna::direction dir, size_t offset, size_t len) {
  amdxdna_drm_sync_bo sbo = {
      .handle = boh,
      .direction =
          (dir == shim_xdna::direction::host2device ? SYNC_DIRECT_TO_DEVICE
                                                    : SYNC_DIRECT_FROM_DEVICE),
      .offset = offset,
      .size = len,
  };
  dev.ioctl(DRM_IOCTL_AMDXDNA_SYNC_BO, &sbo);
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
  try {
    free_drm_bo(m_parent.m_pdev, m_handle);
  } catch (const std::system_error &e) {
    shim_debug("Failed to free DRM BO: %s", e.what());
  }
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
  desc += std::to_string(m_drm_bo->m_handle);
  desc += ", ";
  desc += "size=";
  desc += std::to_string(m_aligned_size);
  return desc;
}

void bo::mmap_bo(size_t align) {
  size_t a = align;

  if (m_drm_bo->m_map_offset == AMDXDNA_INVALID_ADDR) {
    m_aligned = reinterpret_cast<void *>(m_drm_bo->m_vaddr);
    return;
  }

  if (a == 0) {
    m_aligned = map_drm_bo(m_pdev, m_aligned_size, PROT_READ | PROT_WRITE,
                           m_drm_bo->m_map_offset);
    return;
  }

  /*
   * Handle special alignment
   * The first mmap() is just for reserved a range in user vritual address
   * space. The second mmap() uses an aligned addr as the first argument in mmap
   * syscall.
   */
  m_parent_size = align * 2 - 1;
  m_parent = map_parent_range(m_parent_size);
  auto aligned = addr_align(m_parent, align);
  m_aligned =
      map_drm_bo(m_pdev, aligned, m_aligned_size, PROT_READ | PROT_WRITE,
                 MAP_SHARED | MAP_FIXED, m_drm_bo->m_map_offset);
}

void bo::munmap_bo() {
  shim_debug("Unmap BO, aligned %p parent %p", m_aligned, m_parent);
  if (m_drm_bo->m_map_offset == AMDXDNA_INVALID_ADDR) return;

  unmap_drm_bo(m_pdev, m_aligned, m_aligned_size);
  if (m_parent) unmap_drm_bo(m_pdev, m_parent, m_parent_size);
}

void bo::import_bo() {
  uint32_t boh = import_drm_bo(m_pdev, m_import, &m_type, &m_aligned_size);

  amdxdna_drm_get_bo_info bo_info = {};
  get_drm_bo_info(m_pdev, boh, &bo_info);
  m_drm_bo = std::make_unique<drm_bo>(*this, bo_info);
}

void bo::free_bo() { m_drm_bo.reset(); }

bo::bo(const pdev &p, uint32_t ctx_id, size_t size, shim_xcl_bo_flags flags)
    : bo(p, ctx_id, size, flags, flag_to_type(flags)) {
  if (m_type == AMDXDNA_BO_INVALID)
    shim_err(EINVAL, "Invalid BO flags: 0x%lx", flags);
}

bo::bo(const pdev &p, uint32_t ctx_id, size_t size, uint32_t flags)
    : bo(p, ctx_id, size, shim_xcl_bo_flags{.flags = flags}) {
  if (m_type == AMDXDNA_BO_INVALID)
    shim_err(EINVAL, "Invalid BO flags: 0x%lx", flags);
}

bo::bo(const pdev &pdev, uint32_t ctx_id, size_t size, shim_xcl_bo_flags flags,
       amdxdna_bo_type type)
    : m_pdev(pdev),
      m_aligned_size(size),
      m_flags(flags),
      m_type(type),
      m_import(-1),
      m_owner_ctx_id(ctx_id) {
  size_t align = 0;

  if (m_type == AMDXDNA_BO_DEV_HEAP)
    align = 64 * 1024 * 1024;  // Device mem heap must align at 64MB boundary.

  uint32_t boh = alloc_drm_bo(m_pdev, m_type, m_aligned_size);
  // TODO(max): this is dumb? performs an ioctl right after we just made one?
  amdxdna_drm_get_bo_info bo_info = {};
  get_drm_bo_info(m_pdev, boh, &bo_info);
  m_drm_bo = std::make_unique<drm_bo>(*this, bo_info);

  mmap_bo(align);

  // Newly allocated buffer may contain dirty pages. If used as output buffer,
  // the data in cacheline will be flushed onto memory and pollute the output
  // from device. We perform a cache flush right after the BO is allocated to
  // avoid this issue.
  if (m_type == AMDXDNA_BO_SHMEM) {
    sync(direction::host2device, size, 0);
  }

  attach_to_ctx();
#ifndef NDEBUG
  switch (m_flags.all) {
    case 0x0:
      shim_debug("allocating dev heap");
      break;
    case 0x1000000:
      // pdi bo
      shim_debug("allocating pdi bo");
      break;
    case 0x20000000:
      // XCL_BO_FLAGS_P2P in create_free_bo test
      shim_debug("allocating XCL_BO_FLAGS_P2P");
      break;
    case 0x80000000:
      // XCL_BO_FLAGS_EXECBUF in create_free_bo test
      shim_debug("allocating XCL_BO_FLAGS_EXECBUF");
      break;
    case 0x1001000000:
      // debug bo
      shim_debug("allocating debug bo");
      break;
    default:
      shim_err(-1, "unknown flags %d", flags);
  }
#endif

  shim_debug(
      "Allocated KMQ BO (userptr=0x%lx, size=%ld, flags=0x%llx, "
      "type=%d, drm_bo=%d)",
      m_aligned, m_aligned_size, m_flags, m_type, get_drm_bo_handle());
}

bo::bo(const pdev &p, int ehdl) : m_pdev(p), m_import(ehdl) {
  import_bo();
  mmap_bo();
  shim_debug(
      "Imported KMQ BO (userptr=0x%lx, size=%ld, flags=0x%llx, type=%d, "
      "drm_bo=%d)",
      m_aligned, m_aligned_size, m_flags, m_type, get_drm_bo_handle());
}

bo::~bo() {
  shim_debug("Freeing KMQ BO, %s", describe().c_str());

  munmap_bo();
  try {
    detach_from_ctx();
    // If BO is in use, we should block and wait in driver
    free_bo();
  } catch (const std::system_error &e) {
    shim_debug("Failed to free BO: %s", e.what());
  }
}

bo::bo(const pdev &p, size_t size, amdxdna_bo_type type)
    : bo(p, AMDXDNA_INVALID_CTX_HANDLE, size, shim_xcl_bo_flags{}, type) {}

properties bo::get_properties() const {
  return {m_flags, m_aligned_size, get_paddr(), get_drm_bo_handle()};
}

size_t bo::size() { return get_properties().size; }

void *bo::map() const { return m_aligned; }

void bo::unmap(void *addr) {}

uint64_t bo::get_paddr() const {
  if (m_drm_bo->m_xdna_addr != AMDXDNA_INVALID_ADDR)
    return m_drm_bo->m_xdna_addr;
  return reinterpret_cast<uintptr_t>(m_aligned);
}

void bo::set_cmd_id(uint64_t id) { m_cmd_id = id; }

uint64_t bo::get_cmd_id() const { return m_cmd_id; }

uint32_t bo::get_drm_bo_handle() const { return m_drm_bo->m_handle; }

void bo::attach_to_ctx() {
  if (m_owner_ctx_id == AMDXDNA_INVALID_CTX_HANDLE) return;

  auto boh = get_drm_bo_handle();
  shim_debug("Attaching drm_bo %d to ctx: %d", boh, m_owner_ctx_id);
  attach_dbg_drm_bo(m_pdev, boh, m_owner_ctx_id);
}

void bo::detach_from_ctx() {
  if (m_owner_ctx_id == AMDXDNA_INVALID_CTX_HANDLE) return;

  auto boh = get_drm_bo_handle();
  shim_debug("Detaching drm_bo %d from ctx: %d", boh, m_owner_ctx_id);
  detach_dbg_drm_bo(m_pdev, boh, m_owner_ctx_id);
}

std::unique_ptr<shim_xdna::shared_handle> bo::share() const {
  auto boh = get_drm_bo_handle();
  auto fd = export_drm_bo(m_pdev, boh);
  shim_debug("Exported bo %d to fd %d", boh, fd);
  return std::make_unique<shared_handle>(fd);
}

amdxdna_bo_type bo::get_type() const { return m_type; }

void bo::sync(direction dir, size_t size, size_t offset) {
  if (is_driver_sync()) {
    sync_drm_bo(m_pdev, get_drm_bo_handle(), dir, offset, size);
    return;
  }

  if (offset + size > m_aligned_size)
    shim_err(EINVAL, "Invalid BO offset and size for sync'ing: %ld, %ld",
             offset, size);

  switch (m_type) {
    case AMDXDNA_BO_SHMEM:
    case AMDXDNA_BO_CMD:
      clflush_data(m_aligned, offset, size);
      break;
    case AMDXDNA_BO_DEV:
      if (m_owner_ctx_id == AMDXDNA_INVALID_CTX_HANDLE)
        clflush_data(m_aligned, offset, size);
      else
        sync_drm_bo(m_pdev, get_drm_bo_handle(), dir, offset, size);
      break;
    default:
      shim_err(ENOTSUP, "Can't sync bo type %d", m_type);
  }
}

void bo::sync(direction dir) { sync(dir, size(), 0); }

void bo::bind_at(size_t pos, const bo &boh, size_t offset, size_t size) {
  std::lock_guard<std::mutex> lg(m_args_map_lock);

  if (m_type != AMDXDNA_BO_CMD)
    shim_err(EINVAL, "Can't call bind_at() on non-cmd BO");

  if (!pos) m_args_map.clear();

  if (boh.get_type() != AMDXDNA_BO_CMD) {
    auto h = boh.get_drm_bo_handle();
    m_args_map[pos] = h;
    shim_debug("Added arg BO %d to cmd BO %d", h, get_drm_bo_handle());
  } else {
    const size_t max_args_order = 6;
    const size_t max_args = 1 << max_args_order;
    size_t key = pos << max_args_order;
    uint32_t hs[max_args];
    auto arg_cnt = boh.get_arg_bo_handles(hs, max_args);
    std::string bohs;
    for (int i = 0; i < arg_cnt; i++) {
      m_args_map[key + i] = hs[i];
      bohs += std::to_string(hs[i]) + " ";
    }
    shim_debug("Added arg BO %s to cmd BO %d", bohs.c_str(),
               get_drm_bo_handle());
  }
}

uint32_t bo::get_arg_bo_handles(uint32_t *handles, size_t num) const {
  std::lock_guard<std::mutex> lg(m_args_map_lock);

  auto sz = m_args_map.size();
  if (sz > num)
    shim_err(E2BIG, "There are %ld BO args, provided buffer can hold only %ld",
             sz, num);

  for (auto m : m_args_map) *(handles++) = m.second;

  return sz;
}

exec_buf::exec_buf(const pdev &p, uint32_t op)
    : m_exec_buf_bo(std::make_unique<bo>(p, AMDXDNA_INVALID_CTX_HANDLE,
                                         MAX_EXEC_BO_SIZE,
                                         XCL_BO_FLAGS_EXECBUF)),
      m_cmd_pkt(reinterpret_cast<ert_start_kernel_cmd *>(m_exec_buf_bo->map())),
      m_cmd_size(m_exec_buf_bo->size()),
      m_op(op),
      m_arg_cnt(0),
      m_reg_idx(0) {
  std::memset(m_cmd_pkt, 0, m_cmd_size);
  m_cmd_pkt->state = ERT_CMD_STATE_NEW;
  m_cmd_pkt->opcode = m_op;
  m_cmd_pkt->type = ERT_CU;
  // One word for cu mask
  inc_pkt_count(sizeof(int32_t));
}

void exec_buf::set_cu_idx(bo &bo_execbuf, cuidx_t cu_idx) {
  ert_start_kernel_cmd *cmd_pkt =
      reinterpret_cast<ert_start_kernel_cmd *>(bo_execbuf.map());
  cmd_pkt->cu_mask = 0x1 << cu_idx.index;
}

void exec_buf::set_cu_idx(cuidx_t cu_idx) {
  m_cmd_pkt->cu_mask = 0x1 << cu_idx.index;
}

void exec_buf::add_ctrl_bo(bo &bo_ctrl) {
  ert_start_kernel_cmd *cmd_packet =
      reinterpret_cast<ert_start_kernel_cmd *>(m_exec_buf_bo->map());
  switch (m_op) {
    case ERT_START_CU:
      break;
    case ERT_START_NPU: {
      ert_npu_data *npu_data = get_ert_npu_data(cmd_packet);
      npu_data->instruction_buffer = bo_ctrl.get_paddr();
      npu_data->instruction_buffer_size = bo_ctrl.size();
      npu_data->instruction_prop_count = 0;
      inc_pkt_count(sizeof(*npu_data));
      break;
    }
    case ERT_START_DPU: {
      ert_dpu_data *dpu_data = get_ert_dpu_data(cmd_packet);
      dpu_data->instruction_buffer = bo_ctrl.get_paddr();
      dpu_data->instruction_buffer_size = bo_ctrl.size();
      dpu_data->chained = 0;
      inc_pkt_count(sizeof(*dpu_data));
      break;
    }
    default:
      throw std::runtime_error("Unknown exec buf op code: " +
                               std::to_string(m_op));
  }
}

void exec_buf::add_arg_32(uint32_t val) {
  inc_pkt_count(sizeof(val));
  auto args = get_ert_regmap_begin(m_cmd_pkt);
  args[m_reg_idx++] = val;
  m_arg_cnt++;
}

void exec_buf::add_arg_64(uint64_t val) {
  inc_pkt_count(sizeof(val));
  auto args = get_ert_regmap_begin(m_cmd_pkt);
  args[m_reg_idx++] = val;
  args[m_reg_idx++] = val >> 32;
  m_arg_cnt++;
}

void exec_buf::add_arg_bo(bo &bo_arg, std::string arg_name) {
  // Add to argument list for driver
  m_exec_buf_bo->bind_at(m_arg_cnt, bo_arg, 0, bo_arg.size());
  // Add to argument list for control code patching
  if (arg_name.empty())
    m_patching_args.emplace_back(std::to_string(m_arg_cnt), bo_arg.get_paddr());
  else
    m_patching_args.emplace_back(arg_name, bo_arg.get_paddr());
  // Only increase m_arg_cnt now after it's used by code above.
  add_arg_64(bo_arg.get_paddr());
}

void exec_buf::dump() {
  std::cout << "Dumping exec buf:";
  int *data = static_cast<int *>(m_exec_buf_bo->map());
  std::cout << std::hex;
  for (int i = 0; i < m_cmd_pkt->count + 1; i++) {
    if (i % 4 == 0) std::cout << "\n";
    std::cout << std::setfill('0') << std::setw(8) << data[i] << " ";
  }
  std::cout << std::setfill(' ') << std::setw(0) << std::dec << std::endl;

  std::cout << "Dumping patching arguement list:\n";
  for (auto &[arg_name, arg_addr] : m_patching_args)
    std::cout << "{ " << arg_name << ", 0x" << std::hex << arg_addr << std::dec
              << " }\n";
}

void exec_buf::inc_pkt_count(uint32_t n) {
  m_cmd_pkt->count += n / sizeof(int32_t);
  if (m_cmd_size <
      sizeof(m_cmd_pkt->header) + m_cmd_pkt->count * sizeof(int32_t))
    throw std::runtime_error("Size of exec buf too small: " +
                             std::to_string(m_cmd_size));
}

bo *exec_buf::get_exec_buf_bo() { return m_exec_buf_bo.get(); }

}  // namespace shim_xdna
