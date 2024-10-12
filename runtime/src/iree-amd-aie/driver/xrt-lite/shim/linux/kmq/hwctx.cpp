// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2024, Advanced Micro Devices, Inc. All rights reserved.

#include "hwctx.h"

#include <cassert>
#include <cstring>

#include "bo.h"
#include "hwq.h"
#include "shim_debug.h"

namespace shim_xdna {

hw_ctx::hw_ctx(device &dev, const std::map<std::string, uint32_t> &qos,
               std::unique_ptr<hw_q> q, const std::vector<uint8_t> &pdi,
               const std::string &cu_name, size_t functional)
    : m_device(dev), m_q(std::move(q)), m_doorbell(0), m_log_buf(nullptr) {
  shim_debug("Creating HW context...");

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

  m_cu_info.push_back({.m_name = cu_name, .m_func = functional, .m_pdi = pdi});

  if (m_cu_info.empty())
    shim_err(EINVAL, "No valid DPU kernel found in xclbin");
  m_ops_per_cycle = 2048 /*aie_partition.ops_per_cycle*/;
  m_num_cols = 4 /*aie_partition.ncol*/;
}

hw_ctx::hw_ctx(device &device, const std::vector<uint8_t> &pdi,
               const std::string &cu_name,
               const std::map<std::string, uint32_t> &qos)
    : hw_ctx(device, qos, std::make_unique<hw_q>(device), pdi, cu_name) {
  create_ctx_on_device();
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

    m_pdi_bos.push_back(alloc_bo(ci.m_pdi.size(), f));
    std::unique_ptr<bo> &pdi_bo = m_pdi_bos[i];
    char *pdi_vaddr = reinterpret_cast<char *>(pdi_bo->map());

    // see cu_configs[1] in amdxdna_hwctx_param_config_cu
    assert(i < 1 && "only 1 CU supported");
    amdxdna_cu_config &cf = cu_conf_param->cu_configs[i];
    std::memcpy(pdi_vaddr, ci.m_pdi.data(), ci.m_pdi.size());
    pdi_bo->sync(direction::host2device, pdi_bo->get_properties().size, 0);
    cf.cu_bo = pdi_bo->get_drm_bo_handle();
    cf.cu_func = ci.m_func;
  }

  amdxdna_drm_config_hwctx arg = {};
  arg.handle = m_handle;
  arg.param_type = DRM_AMDXDNA_HWCTX_CONFIG_CU;
  arg.param_val = reinterpret_cast<uintptr_t>(cu_conf_param);
  arg.param_val_size = cu_conf_param_buf.size();
  m_device.get_pdev().ioctl(DRM_IOCTL_AMDXDNA_CONFIG_HWCTX, &arg);

  shim_debug("Created KMQ HW context (%d)", m_handle);
}

hw_ctx::~hw_ctx() {
  try {
    delete_ctx_on_device();
  } catch (const std::system_error &e) {
    shim_debug("Failed to delete context on device: %s", e.what());
  }
  shim_debug("Destroyed HW context (%d)...", m_handle);
  shim_debug("Destroying KMQ HW context (%d)...", m_handle);
}

cuidx_t hw_ctx::open_cu_context(const std::string &cu_name) {
  for (uint32_t i = 0; i < m_cu_info.size(); i++) {
    auto &ci = m_cu_info[i];
    shim_debug("ci.m_name %s", ci.m_name.c_str());
    if (ci.m_name == cu_name) return cuidx_t{.index = i};
  }

  shim_err(ENOENT, "CU name (%s) not found", cu_name.c_str());
}

std::unique_ptr<bo> hw_ctx::alloc_bo(size_t size, shim_xcl_bo_flags flags) {
  // const_cast: alloc_bo() is not const yet in device class
  // Debug buffer is specific to one context.
  if (flags.use == XRT_BO_USE_DEBUG)
    return m_device.alloc_bo(m_handle, size, flags);
  // Other BOs are shared across all contexts.
  return m_device.alloc_bo(AMDXDNA_INVALID_CTX_HANDLE, size, flags);
}

std::unique_ptr<bo> hw_ctx::import_bo(pid_t pid, int ehdl) {
  // const_cast: import_bo() is not const yet in device class
  return m_device.import_bo(pid, ehdl);
}

hw_q *hw_ctx::get_hw_queue() const { return m_q.get(); }

void hw_ctx::create_ctx_on_device() {
  amdxdna_drm_create_hwctx arg = {};
  arg.qos_p = reinterpret_cast<uintptr_t>(&m_qos);
  arg.umq_bo = m_q->m_queue_boh;
  arg.max_opc = m_ops_per_cycle;
  // TODO(max)
  //  throw std::runtime_error("TODO(max): core_rows");
  //  arg.num_tiles = m_num_cols *
  //  xrt_core::device_query<xrt_core::query::aie_tiles_stats>(&m_device).core_rows;
  arg.num_tiles = m_num_cols * 4;
  arg.log_buf_bo =
      m_log_bo ? m_log_bo->get_drm_bo_handle() : AMDXDNA_INVALID_BO_HANDLE;
  m_device.get_pdev().ioctl(DRM_IOCTL_AMDXDNA_CREATE_HWCTX, &arg);

  m_handle = arg.handle;
  m_doorbell = arg.umq_doorbell;

  m_q->bind_hwctx(this);
}

void hw_ctx::delete_ctx_on_device() {
  if (m_handle == AMDXDNA_INVALID_CTX_HANDLE) return;

  m_q->unbind_hwctx();
  amdxdna_drm_destroy_hwctx arg = {};
  arg.handle = m_handle;
  m_device.get_pdev().ioctl(DRM_IOCTL_AMDXDNA_DESTROY_HWCTX, &arg);

  fini_log_buf();
}

void hw_ctx::init_log_buf() {
  auto log_buf_size = m_num_cols * 1024;
  shim_xcl_bo_flags f;
  f.flags = XCL_BO_FLAGS_EXECBUF;
  m_log_bo = alloc_bo(log_buf_size, f);
  m_log_buf = m_log_bo->map();
  std::memset(m_log_buf, 0, log_buf_size);
}

void hw_ctx::fini_log_buf() const {
  if (m_log_bo) m_log_bo->unmap(m_log_buf);
}

}  // namespace shim_xdna
