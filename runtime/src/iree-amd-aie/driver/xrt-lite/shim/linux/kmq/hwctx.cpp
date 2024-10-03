// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2024, Advanced Micro Devices, Inc. All rights reserved.

#include "hwctx.h"

#include "bo.h"
#include "device.h"
#include "hwq.h"
#include "pcidev.h"

namespace shim_xdna {

hw_ctx::hw_ctx(const device& dev, const qos_type& qos, std::unique_ptr<hw_q> q)
    : m_device(dev), m_q(std::move(q)), m_doorbell(0), m_log_buf(nullptr) {
  shim_debug("Creating HW context...");
  init_qos_info(qos);
}

hw_ctx::~hw_ctx() {
  delete_ctx_on_device();
  shim_debug("Destroyed HW context (%d)...", m_handle);
}

uint32_t hw_ctx::get_slotidx() const { return m_handle; }

void hw_ctx::set_slotidx(uint32_t id) { m_handle = id; }

cuidx_type hw_ctx::open_cu_context(const std::string& cu_name) {
  for (uint32_t i = 0; i < m_cu_info.size(); i++) {
    auto& ci = m_cu_info[i];
    if (ci.m_name == cu_name) return cuidx_type{.index = i};
  }

  shim_err(ENOENT, "CU name (%s) not found", cu_name.c_str());
}

void hw_ctx::close_cu_context(cuidx_type cuidx) {
  // Nothing to be done
}

std::unique_ptr<bo> hw_ctx::alloc_bo(size_t size, uint64_t flags) {
  return alloc_bo(nullptr, size, flags);
}

std::unique_ptr<bo> hw_ctx::import_bo(pid_t pid,
                                      shared_handle::export_handle ehdl) {
  // const_cast: import_bo() is not const yet in device class
  auto& dev = const_cast<device&>(get_device());
  return dev.import_bo(pid, ehdl);
}

hw_q* hw_ctx::get_hw_queue() { return m_q.get(); }

void hw_ctx::init_qos_info(const qos_type& qos) {
  for (auto& [key, value] : qos) {
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
}

const device& hw_ctx::get_device() { return m_device; }

const std::vector<hw_ctx::cu_info>& hw_ctx::get_cu_info() const {
  return m_cu_info;
}

void hw_ctx::create_ctx_on_device() {
  amdxdna_drm_create_hwctx arg = {};
  arg.qos_p = reinterpret_cast<uintptr_t>(&m_qos);
  arg.umq_bo = m_q->get_queue_bo();
  arg.max_opc = m_ops_per_cycle;
  // arg.num_tiles =
  //     m_num_cols *
  //     xrt_core::device_query<xrt_core::query::aie_tiles_stats>(&m_device)
  //         .core_rows;
  arg.log_buf_bo = m_log_bo
                       ? static_cast<bo*>(m_log_bo.get())->get_drm_bo_handle()
                       : AMDXDNA_INVALID_BO_HANDLE;
  m_device.get_pdev().ioctl(DRM_IOCTL_AMDXDNA_CREATE_HWCTX, &arg);

  set_slotidx(arg.handle);
  set_doorbell(arg.umq_doorbell);

  m_q->bind_hwctx(this);
}

void hw_ctx::delete_ctx_on_device() {
  if (m_handle == AMDXDNA_INVALID_CTX_HANDLE) return;

  m_q->unbind_hwctx();
  struct amdxdna_drm_destroy_hwctx arg = {};
  arg.handle = m_handle;
  m_device.get_pdev().ioctl(DRM_IOCTL_AMDXDNA_DESTROY_HWCTX, &arg);

  fini_log_buf();
}

void hw_ctx::init_log_buf() {
  auto log_buf_size = m_num_cols * 1024;
  m_log_bo = alloc_bo(nullptr, log_buf_size, XCL_BO_FLAGS_EXECBUF);
  m_log_buf = m_log_bo->map(bo::map_type::write);
  std::memset(m_log_buf, 0, log_buf_size);
}

void hw_ctx::fini_log_buf(void) {
  if (m_log_bo) m_log_bo->unmap(m_log_buf);
}

void hw_ctx::set_doorbell(uint32_t db) { m_doorbell = db; }

uint32_t hw_ctx::get_doorbell() const { return m_doorbell; }

}  // namespace shim_xdna
