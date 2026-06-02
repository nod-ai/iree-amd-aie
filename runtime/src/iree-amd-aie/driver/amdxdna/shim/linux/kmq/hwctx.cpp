// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2024, Advanced Micro Devices, Inc. All rights reserved.

#include "hwctx.h"

#include <cerrno>
#include <cstring>

#include "amdxdna_accel.h"
#include "bo.h"
#include "hwq.h"
#include "shim_debug.h"

namespace shim_xdna {

static constexpr uint32_t kDefaultOpsPerCycle = 2048;

hw_ctx::hw_ctx(device &dev, const std::map<std::string, uint32_t> &qos,
               std::unique_ptr<hw_q> q, const std::vector<uint8_t> &pdi,
               const std::string &cu_name, uint32_t n_rows, uint32_t n_cols)
    : m_device(dev),
      m_q(std::move(q)),
      m_num_rows(n_rows),
      m_num_cols(n_cols),
      m_doorbell(0),
      m_syncobj(AMDXDNA_INVALID_FENCE_HANDLE),
      m_log_buf(nullptr) {
  SHIM_DEBUG("Creating HW context...");

  for (auto &[key, value] : qos) {
    if (key == "gops")
      m_qos.gops = value;
    else if (key == "fps")
      m_qos.fps = value;
    else if (key == "dma_bandwidth")
      m_qos.dma_bandwidth = value;
    else if (key == "latency")
      m_qos.latency = value;
    else if (key == "frame_execution_time")
      m_qos.frame_exec_time = value;
    else if (key == "priority")
      m_qos.priority = value;
  }

  // The flatbuffer executable path supplies one PDI/CU entry per hardware
  // context today. If multi-PDI contexts become useful, extend the constructor
  // API to accept a list and keep the CONFIG_CU packing below in lockstep.
  m_cu_info.push_back(
      {.m_name = cu_name, .m_func = /*functional*/ 0, .m_pdi = pdi});

  // max_opc is a kernel QoS hint. There is no HAL/device option for it yet, so
  // use the driver default value expected by the current XDNA stack.
  m_ops_per_cycle = kDefaultOpsPerCycle;
}

hw_ctx::hw_ctx(device &device, const std::vector<uint8_t> &pdi,
               const std::string &cu_name, uint32_t n_rows, uint32_t n_cols,
               const std::map<std::string, uint32_t> &qos)
    : hw_ctx(device, qos, std::make_unique<hw_q>(device), pdi, cu_name, n_rows,
             n_cols) {
  m_init_errno = create_ctx_on_device();
  if (m_init_errno) return;
  std::vector<char> cu_conf_param_buf(sizeof(amdxdna_hwctx_param_config_cu) +
                                      m_cu_info.size() *
                                          sizeof(amdxdna_cu_config));
  auto cu_conf_param = reinterpret_cast<amdxdna_hwctx_param_config_cu *>(
      cu_conf_param_buf.data());

  cu_conf_param->num_cus = m_cu_info.size();
  shim_xcl_bo_flags f = {};
  f.flags = XRT_BO_FLAGS_CACHEABLE;
  for (int i = 0; i < m_cu_info.size(); i++) {
    cu_info &ci = m_cu_info[i];

    std::unique_ptr<bo> pdi_bo;
    m_init_errno = alloc_bo(ci.m_pdi.size(), f, &pdi_bo);
    if (m_init_errno) return;
    char *pdi_vaddr = reinterpret_cast<char *>(pdi_bo->map());
    if (!pdi_vaddr) {
      m_init_errno = EINVAL;
      return;
    }

    // see cu_configs[1] in amdxdna_hwctx_param_config_cu
    if (i >= 1) {
      m_init_errno = EINVAL;
      return;
    }
    amdxdna_cu_config &cf = cu_conf_param->cu_configs[i];
    std::memcpy(pdi_vaddr, ci.m_pdi.data(), ci.m_pdi.size());
    m_init_errno =
        pdi_bo->sync(direction::host2device, pdi_bo->get_properties().size, 0);
    if (m_init_errno) return;
    cf.cu_bo = pdi_bo->get_drm_bo_handle();
    cf.cu_func = ci.m_func;
    m_pdi_bos.push_back(std::move(pdi_bo));
  }

  amdxdna_drm_config_hwctx arg = {};
  arg.handle = m_handle;
  arg.param_type = DRM_AMDXDNA_HWCTX_CONFIG_CU;
  arg.param_val = reinterpret_cast<uintptr_t>(cu_conf_param);
  arg.param_val_size = cu_conf_param_buf.size();
  m_init_errno =
      m_device.get_pdev().try_ioctl(DRM_IOCTL_AMDXDNA_CONFIG_HWCTX, &arg);
  if (m_init_errno) return;

  SHIM_DEBUG("Created KMQ HW context (%d)", m_handle);
}

hw_ctx::~hw_ctx() {
  delete_ctx_on_device();
  delete_syncobj();
  SHIM_DEBUG("Destroyed HW context (%d)...", m_handle);
  SHIM_DEBUG("Destroying KMQ HW context (%d)...", m_handle);
}

int hw_ctx::init_errno() const { return m_init_errno; }

int hw_ctx::open_cu_context(const std::string &cu_name, cuidx_t *out_cu_idx) {
  for (uint32_t i = 0; i < m_cu_info.size(); i++) {
    auto &ci = m_cu_info[i];
    SHIM_DEBUG("ci.m_name %s", ci.m_name.c_str());
    if (ci.m_name == cu_name) {
      *out_cu_idx = cuidx_t{.index = i};
      return 0;
    }
  }

  return ENOENT;
}

int hw_ctx::alloc_bo(size_t size, shim_xcl_bo_flags flags,
                     std::unique_ptr<bo> *out_bo) {
  // Debug buffer is specific to one context.
  if (flags.use == XRT_BO_USE_DEBUG)
    return m_device.alloc_bo(m_handle, size, flags, out_bo);
  // Other BOs are shared across all contexts.
  return m_device.alloc_bo(AMDXDNA_INVALID_CTX_HANDLE, size, flags, out_bo);
}

hw_q *hw_ctx::get_hw_queue() const { return m_q.get(); }

int hw_ctx::create_ctx_on_device() {
  amdxdna_drm_create_hwctx arg = {};
  arg.qos_p = reinterpret_cast<uintptr_t>(&m_qos);
  arg.umq_bo = m_q->m_queue_boh;
  arg.max_opc = m_ops_per_cycle;
  arg.num_tiles = m_num_rows * m_num_cols;
  arg.log_buf_bo =
      m_log_bo ? m_log_bo->get_drm_bo_handle() : AMDXDNA_INVALID_BO_HANDLE;
  int err = m_device.get_pdev().try_ioctl(DRM_IOCTL_AMDXDNA_CREATE_HWCTX, &arg);
  if (err) return err;

  m_handle = arg.handle;
  m_doorbell = arg.umq_doorbell;
  m_syncobj = arg.syncobj_handle;

  m_q->bind_hwctx(this);
  return 0;
}

void hw_ctx::delete_ctx_on_device() const {
  if (m_handle == AMDXDNA_INVALID_CTX_HANDLE) return;

  m_q->unbind_hwctx();
  amdxdna_drm_destroy_hwctx arg = {};
  arg.handle = m_handle;
  (void)m_device.get_pdev().try_ioctl(DRM_IOCTL_AMDXDNA_DESTROY_HWCTX, &arg);

  fini_log_buf();
}

void hw_ctx::delete_syncobj() const {
  if (m_syncobj == AMDXDNA_INVALID_FENCE_HANDLE) return;
  drm_syncobj_destroy dsobj = {.handle = m_syncobj};
  (void)m_device.get_pdev().try_ioctl(DRM_IOCTL_SYNCOBJ_DESTROY, &dsobj);
}

void hw_ctx::init_log_buf() {
  size_t column_size = 1024;
  auto log_buf_size = m_num_cols * column_size + sizeof(m_metadata);
  shim_xcl_bo_flags f;
  f.flags = XCL_BO_FLAGS_EXECBUF;
  m_init_errno = alloc_bo(log_buf_size, f, &m_log_bo);
  if (m_init_errno) return;
  m_log_buf = m_log_bo->map();
  if (!m_log_buf) {
    m_init_errno = EINVAL;
    return;
  }
  uint64_t bo_paddr = m_log_bo->get_properties().paddr;
  set_metadata(m_num_cols, column_size, bo_paddr, 1);
  std::memset(m_log_buf, 0, log_buf_size);
  std::memcpy(m_log_buf, &m_metadata, sizeof(m_metadata));
}

void hw_ctx::fini_log_buf() const {
  if (m_log_bo) m_log_bo->unmap(m_log_buf);
}

void hw_ctx::set_metadata(int num_cols, size_t size, uint64_t bo_paddr,
                          uint8_t flag) {
  m_metadata.magic_no = CERT_MAGIC_NO;
  m_metadata.major = 0;
  m_metadata.minor = 1;
  m_metadata.cert_log_flag = flag;
  m_metadata.num_cols = num_cols;
  for (int i = 0; i < num_cols; i++) {
    m_metadata.col_paddr[i] = bo_paddr + size * i + sizeof(m_metadata);
    m_metadata.col_size[i] = size;
  }
}

}  // namespace shim_xdna
