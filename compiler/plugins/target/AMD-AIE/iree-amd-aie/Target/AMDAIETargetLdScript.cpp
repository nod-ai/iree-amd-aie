// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "AMDAIETargets.h"
#include "aie/AIEDialect.h"

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIE;

// Output the memorymap in gnu linker format for the given buffer operations,
// with the given offset. The offset is different depending on where the buffers
// are accessed from.
static void writeLDScriptMap(raw_ostream &output, BufferOp buf, int offset) {
  std::string bufName(name(buf).getValue());
  int bufferBaseAddr = buf.getAddress().value();
  int numBytes = getAllocationSize(buf);
  output << ". = 0x" << llvm::utohexstr(offset + bufferBaseAddr) << ";\n";
  output << bufName << " = .;\n";
  output << ". += 0x" << llvm::utohexstr(numBytes) << ";\n";
}

LogicalResult mlir::iree_compiler::AMDAIE::AIETranslateToLdScript(
    DeviceOp deviceOp, raw_ostream &output, int tileCol, int tileRow) {
  DenseMap<TileLoc, Operation *> tiles;
  DenseMap<Operation *, SmallVector<BufferOp, 4>> buffers;

  collectTiles(deviceOp, tiles);
  ::collectBuffers(deviceOp, buffers);

  AMDAIEDeviceModel deviceModel = getDeviceModel(deviceOp.getDevice());
  for (auto tile : deviceOp.getOps<TileOp>())
    if (tile.getCol() == tileCol && tile.getRow() == tileRow) {
      TileLoc srcCoord = {tile.getCol(), tile.getRow()};

      // Figure out how much memory we have left for random allocations
      auto core = getCoreOp(tile);
      int max = core.getStackSize();
      for (auto buf : buffers[tiles[srcCoord]]) {
        int bufferBaseAddr = buf.getAddress().value();
        int numBytes = getAllocationSize(buf);
        max = std::max(max, bufferBaseAddr + numBytes);
      }
      int origin = deviceModel.getMemInternalBaseAddress() + max;
      int length = deviceModel.getCoreTileLocalMemorySize() - max;
      output << R"THESCRIPT(
MEMORY
{
   program (RX) : ORIGIN = 0, LENGTH = 0x0020000
)THESCRIPT";
      output << "   data (!RX) : ORIGIN = 0x" << llvm::utohexstr(origin)
             << ", LENGTH = 0x" << llvm::utohexstr(length);
      output << R"THESCRIPT(
}
ENTRY(__start)
SECTIONS
{
  . = 0x0;
  .text : {
    *crt0.o(.text*)
    *(.text*)
  } > program
  .data : {
     *(.data*);
     *(.rodata*)
  } > data
  .comment : {
     *(.comment*)
  }
  .symtab : {
     *(.symtab)
  }
  .shstrtab : {
     *(.shstrtab)
  }
  .strtab : {
     *(.strtab)
  }
)THESCRIPT";
      auto doBuffer = [&](std::optional<TileLoc> tile, int offset,
                          const std::string &dir) {
        if (tile) {
          if (tiles.count({tile->col, tile->row}))
            for (auto buf : buffers[tiles[{tile->col, tile->row}]])
              writeLDScriptMap(output, buf, offset);
        } else {
          output << "/* No tile with memory exists to the " << dir << ". */\n";
          output << ". = 0x" << llvm::utohexstr(offset) << ";\n";
          uint32_t localMemSize = deviceModel.getCoreTileLocalMemorySize();
          output << ". += 0x" << llvm::utohexstr(localMemSize) << ";\n";
        }
      };

      // Stack
      output << ". = 0x"
             << llvm::utohexstr(deviceModel.getMemInternalBaseAddress())
             << ";\n";
      output << "_sp_start_value_DM_stack = .;\n";

      if (auto core = getCoreOp(tile))
        output << ". += 0x" << llvm::utohexstr(core.getStackSize())
               << "; /* stack */\n";
      else
        output << "/* no stack allocated */\n";

      doBuffer(deviceModel.getMemSouth(srcCoord),
               deviceModel.getMemSouthBaseAddress(), std::string("south"));
      doBuffer(deviceModel.getMemWest(srcCoord),
               deviceModel.getMemWestBaseAddress(), std::string("west"));
      doBuffer(deviceModel.getMemNorth(srcCoord),
               deviceModel.getMemNorthBaseAddress(), std::string("north"));
      doBuffer(deviceModel.getMemEast(srcCoord),
               deviceModel.getMemEastBaseAddress(), std::string("east"));

      output << "  .bss : { *(.bss) } > data\n";
      output << "}\n";
      if (auto coreOp = getCoreOp(tile)) {
        output << "PROVIDE(main = core_" << std::to_string(tile.getCol()) << "_"
               << std::to_string(tile.getRow()) << ");\n";
      }
    }
  return success();
}
