// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "AMDAIETargets.h"
#include "aie/AIEDialect.h"
#include "aie/Passes.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/IR/Module.h"

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIE;

std::string utohexstr(uint32_t u) { return "0x" + llvm::utohexstr(u); }

namespace {}  // namespace

namespace mlir::iree_compiler::AMDAIE {

LogicalResult AIETranslateToBCF(ModuleOp module, raw_ostream &output,
                                int tileCol, int tileRow) {
  DenseMap<TileLoc, Operation *> tiles;
  DenseMap<Operation *, SmallVector<BufferOp, 4>> buffers;

  if (module.getOps<DeviceOp>().empty())
    module.emitOpError("expected aie.device operation at toplevel");
  DeviceOp deviceOp = *(module.getOps<DeviceOp>().begin());

  collectTiles(deviceOp, tiles);
  collectBuffers(deviceOp, buffers);

  // _entry_point _main_init
  // _symbol      _main _after _main_init
  // _symbol      _main_init 0
  // _reserved DMb      0x00000 0x20000
  // _symbol   a        0x38000 0x2000
  // _extern   a
  // _stack    DM_stack 0x20000  0x400 //stack for core
  // _reserved DMb 0x40000 0xc0000 // And everything else the core can't
  // see
  // // Include all symbols from rom.c
  // _include _file rom.o
  AMDAIEDeviceModel deviceModel =
      getDeviceModel(static_cast<AMDAIEDevice>(deviceOp.getDevice()));
  for (auto tile : deviceOp.getOps<TileOp>())
    if (tile.getCol() == tileCol && tile.getRow() == tileRow) {
      TileLoc srcCoord = {tile.getCol(), tile.getRow()};

      std::string corefunc = std::string("core_") +
                             std::to_string(tile.getCol()) + "_" +
                             std::to_string(tile.getRow());
      output << "_entry_point _main_init\n";
      output << "_symbol " << corefunc << " _after _main_init\n";
      output << "_symbol _main_init 0\n";
      std::string initReserved = "0x40000";
      output << "_reserved DMb 0x00000 " << initReserved
             << " // Don't put data in code memory\n";

      int stacksize = 0;
      if (auto core = getCoreOp(tile)) stacksize = core.getStackSize();
      output << "_stack DM_stack "
             << utohexstr(deviceModel.getMemInternalBaseAddress()) << " "
             << utohexstr(stacksize) << " // stack for core\n";

      auto doBuffer = [&](std::optional<TileLoc> tile, int offset,
                          const std::string &dir) {
        if (tile) {
          output << "// " + dir +
                        " -------------------------------------------------\n";
          uint32_t localMemSize =
              deviceModel.getLocalMemorySize(tile->col, tile->row);
          if (srcCoord != tile)
            output << "_reserved DMb " << utohexstr(offset) << " "
                   << utohexstr(localMemSize) << " "
                   << " // Don't allocate variables in " << dir
                   << " neighbor\n\n";
          // TODO How to set as reserved if no buffer exists (or reserve
          // remaining buffer)
          if (tiles.count(TileLoc(*tile))) {
            for (auto buf : buffers[tiles[TileLoc(*tile)]]) {
              std::string bufName(name(buf).getValue());
              int bufferBaseAddr = buf.getAddress().value();
              int numBytes = getAllocationSize(buf);
              output << "_symbol " << bufName << " "
                     << utohexstr(offset + bufferBaseAddr) << " " << numBytes
                     << '\n';
              output << "_extern " << bufName << "\n";
              output << "_reserved DMb " << utohexstr(offset + bufferBaseAddr)
                     << " " << numBytes << '\n';
              output << "\n";
            }
          }
        } else {
          uint32_t localMemSize = deviceModel.getCoreTileLocalMemorySize();
          output << "_reserved DMb " << utohexstr(offset) << " "
                 << utohexstr(localMemSize) << " "
                 << " // No tile with memory exists to the " << dir << ".\n";
        }
      };

      output << "\n// mapping neighbors tile memory\n";
      doBuffer(deviceModel.getMemSouth(srcCoord),
               deviceModel.getMemSouthBaseAddress(), std::string("south"));
      doBuffer(deviceModel.getMemWest(srcCoord),
               deviceModel.getMemWestBaseAddress(), std::string("west"));
      doBuffer(deviceModel.getMemNorth(srcCoord),
               deviceModel.getMemNorthBaseAddress(), std::string("north"));
      doBuffer(deviceModel.getMemEast(srcCoord),
               deviceModel.getMemEastBaseAddress(), std::string("east"));
      output << "// end mapping neighbors tile memory\n\n";
      output << "_reserved DMb 0x80000 0x80000 // And everything else "
                "the core can't see\n";
      // chess's libc expects a _main not a main (despite what me_basic.c looks
      // like...)
      output << "_resolve _main core_" << tile.getCol() << "_" << tile.getRow()
             << "\n";
    }

  return success();
}
}  // namespace mlir::iree_compiler::AMDAIE
