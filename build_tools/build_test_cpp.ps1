# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
Invoke-Expression ".\config_release.ps1"
Invoke-Expression ".\build.ps1"
Invoke-Expression ".\create_install.ps1"