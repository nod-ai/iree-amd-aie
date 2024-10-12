//
// Created by mlevental on 10/11/24.
//

#include "kernel.h"

#include <cstring>
#include <iostream>

#include "amdxdna_accel.h"
#include "bo.h"
#include "device.h"

#define MAX_EXEC_BO_SIZE 4096

namespace shim_xdna {
kernel::kernel(const pdev &p, uint32_t op)
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

void kernel::set_cu_idx(bo &bo_execbuf, cuidx_t cu_idx) {
  ert_start_kernel_cmd *cmd_pkt =
      reinterpret_cast<ert_start_kernel_cmd *>(bo_execbuf.map());
  cmd_pkt->cu_mask = 0x1 << cu_idx.index;
}

void kernel::set_cu_idx(cuidx_t cu_idx) {
  m_cmd_pkt->cu_mask = 0x1 << cu_idx.index;
}

void kernel::add_ctrl_bo(bo &bo_ctrl) {
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

void kernel::add_arg_32(uint32_t val) {
  inc_pkt_count(sizeof(val));
  auto args = get_ert_regmap_begin(m_cmd_pkt);
  args[m_reg_idx++] = val;
  m_arg_cnt++;
}

void kernel::add_arg_64(uint64_t val) {
  inc_pkt_count(sizeof(val));
  auto args = get_ert_regmap_begin(m_cmd_pkt);
  args[m_reg_idx++] = val;
  args[m_reg_idx++] = val >> 32;
  m_arg_cnt++;
}

void kernel::add_arg_bo(bo &bo_arg, const std::string &arg_name) {
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

void kernel::dump() {
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

void kernel::inc_pkt_count(uint32_t n) const {
  m_cmd_pkt->count += n / sizeof(int32_t);
  if (m_cmd_size <
      sizeof(m_cmd_pkt->header) + m_cmd_pkt->count * sizeof(int32_t))
    throw std::runtime_error("Size of exec buf too small: " +
                             std::to_string(m_cmd_size));
}

bo *kernel::get_exec_buf_bo() const { return m_exec_buf_bo.get(); }

}  // namespace shim_xdna
