// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2024, Advanced Micro Devices, Inc. All rights reserved.

#ifndef _PCIDRV_XDNA_H_
#define _PCIDRV_XDNA_H_

#include <string>

#include "pcidev.h"

namespace shim_xdna {

struct drv : std::enable_shared_from_this<drv> {
  std::string name() const;
  bool is_user() const;
  std::string dev_node_prefix() const;
  std::string dev_node_dir() const;
  std::string sysfs_dev_node_dir() const;
  std::shared_ptr<pdev> create_pcidev(const std::string& sysfs) const;
};

}  // namespace shim_xdna

#endif
