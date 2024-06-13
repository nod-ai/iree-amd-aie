//===- AIETargetBCF.cpp -----------------------------------------*- C++ -*-===//
//
// Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "AIETargets.h"

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/IR/Module.h"

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIE;

std::string utohexstr(uint32_t u) { return "0x" + llvm::utohexstr(u); }

namespace xilinx {
namespace AIE {

LogicalResult AIETranslateToBCF(ModuleOp module, raw_ostream &output,
                                int tileCol, int tileRow) {
  DenseMap<TileID, Operation *> tiles;
  DenseMap<Operation *, SmallVector<BufferOp, 4>> buffers;

  if (module.getOps<DeviceOp>().empty())
    module.emitOpError("expected aie.device operation at toplevel");
  DeviceOp targetOp = *(module.getOps<DeviceOp>().begin());

  collectTiles(targetOp, tiles);
  collectBuffers(targetOp, buffers);

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
  for (auto tile : targetOp.getOps<TileOp>())
    if (tile.colIndex() == tileCol && tile.rowIndex() == tileRow) {
      const auto &targetModel = getTargetModel(tile);
      TileID srcCoord = {tile.colIndex(), tile.rowIndex()};

      std::string corefunc = std::string("core_") +
                             std::to_string(tile.getCol()) + "_" +
                             std::to_string(tile.getRow());
      output << "_entry_point _main_init\n";
      output << "_symbol " << corefunc << " _after _main_init\n";
      output << "_symbol _main_init 0\n";
      int dataMemoryStart = targetModel.getMemSouthBaseAddress();
      output << "_reserved DMb 0x00000 " << utohexstr(dataMemoryStart)
             << " // Don't put data in code memory\n";

      int stacksize = 0;
      if (auto core = tile.getCoreOp()) stacksize = core.getStackSize();
      output << "_stack DM_stack "
             << utohexstr(targetModel.getMemInternalBaseAddress(srcCoord))
             << " " << utohexstr(stacksize) << " // stack for core\n";

      auto doBuffer = [&](std::optional<TileID> tile, int offset,
                          const std::string &dir) {
        if (tile) {
          output << "// " + dir +
                        " -------------------------------------------------\n";
          uint32_t localMemSize = targetModel.getLocalMemorySize();
          if (tile != srcCoord)
            output << "_reserved DMb " << utohexstr(offset) << " "
                   << utohexstr(localMemSize) << " "
                   << " // Don't allocate variables in " << dir
                   << " neighbor\n\n";
          // TODO How to set as reserved if no buffer exists (or reserve
          // remaining buffer)
          if (tiles.count(*tile)) {
            for (auto buf : buffers[tiles[*tile]]) {
              std::string bufName(buf.name().getValue());
              int bufferBaseAddr = getBufferBaseAddress(buf);
              int numBytes = buf.getAllocationSize();
              if (buf.getInitialValue() && tile == srcCoord) {
                output << "_overlay " << bufName << " "
                       << utohexstr(offset + bufferBaseAddr) << " // "
                       << numBytes << " bytes\n";
              } else {
                output << "_symbol " << bufName << " "
                       << utohexstr(offset + bufferBaseAddr) << " " << numBytes
                       << '\n';
                output << "_extern " << bufName << "\n";
                output << "_reserved DMb " << utohexstr(offset + bufferBaseAddr)
                       << " " << numBytes << '\n';
              }
              output << "\n";
            }
          }
        } else {
          uint32_t localMemSize = targetModel.getLocalMemorySize();
          output << "_reserved DMb " << utohexstr(offset) << " "
                 << utohexstr(localMemSize) << " "
                 << " // No tile with memory exists to the " << dir << ".\n";
        }
      };

      output << "\n// mapping neighbors tile memory\n";
      doBuffer(targetModel.getMemSouth(srcCoord),
               targetModel.getMemSouthBaseAddress(), std::string("south"));
      doBuffer(targetModel.getMemWest(srcCoord),
               targetModel.getMemWestBaseAddress(), std::string("west"));
      doBuffer(targetModel.getMemNorth(srcCoord),
               targetModel.getMemNorthBaseAddress(), std::string("north"));
      doBuffer(targetModel.getMemEast(srcCoord),
               targetModel.getMemEastBaseAddress(), std::string("east"));
      output << "// end mapping neighbors tile memory\n\n";
      int addressSpaceSize = 0x100000;
      int dataMemoryEnd = targetModel.getMemEastBaseAddress() +
                          targetModel.getLocalMemorySize();
      output << "_reserved DMb " << utohexstr(dataMemoryEnd) << " "
             << utohexstr(addressSpaceSize - dataMemoryEnd)
             << " // And everything else the core can't see\n";
      if (tile.getCoreOp() && tile.getCoreOp().getLinkWith())
        output << "_include _file "
               << tile.getCoreOp().getLinkWith().value().str() << "\n";
      output << "_resolve _main core_" << tile.getCol() << "_" << tile.getRow()
             << "\n";
    }

  return success();
}
}  // namespace AIE
}  // namespace xilinx

//===- AIETargetLdScript.cpp -----------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2023 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
#include "AIETargets.h"
#include "aie/Dialect/AIE/IR/AIEDialect.h"

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIE;

// Output the memorymap in gnu linker format for the given buffer operations,
// with the given offset. The offset is different depending on where the buffers
// are accessed from.
static void writeLDScriptMap(raw_ostream &output, BufferOp buf, int offset) {
  std::string bufName(buf.name().getValue());
  int bufferBaseAddr = getBufferBaseAddress(buf);
  int numBytes = buf.getAllocationSize();
  output << ". = 0x" << llvm::utohexstr(offset + bufferBaseAddr) << ";\n";
  output << bufName << " = .;\n";
  output << ". += 0x" << llvm::utohexstr(numBytes) << ";\n";
}

///// ld.script format:
//
// MEMORY
// {
//    program (RX) : ORIGIN = 0, LENGTH = 0x0020000
//    data (!RX) : ORIGIN = 0x20000, LENGTH = 0x0020000
// }
// ENTRY(_main_init)
// INPUT(something.o)
// SECTIONS
// {
//   . = 0x0;
//   .text : {
//      // the _main_init symbol from me_basic.o has to come at address zero.
//      *me_basic.o(.text)
//      . = 0x200;
//      __ctors_start__ = .;
//      __init_array_start = .;
//      KEEP(SORT(*)(.init_array))
//      __ctors_end__ = .;
//      __init_array_end = .;
//      __dtors_start__ = .;
//      __dtors_end__ = .;
//      *(.text)
//   } > program
//   .data : { *(.data) } > data
//   . = 0x20000;
//   _sp_start_value_DM_stack = .;
//   . = 0x24000;
//   a = .;
//   . += 1024;
//   .bss : { *(.bss) } > data
// }
LogicalResult xilinx::AIE::AIETranslateToLdScript(ModuleOp module,
                                                  raw_ostream &output,
                                                  int tileCol, int tileRow) {
  DenseMap<TileID, Operation *> tiles;
  DenseMap<Operation *, SmallVector<BufferOp, 4>> buffers;

  if (module.getOps<DeviceOp>().empty()) {
    module.emitOpError("expected AIE.device operation at toplevel");
  }
  DeviceOp targetOp = *(module.getOps<DeviceOp>().begin());

  collectTiles(targetOp, tiles);
  collectBuffers(targetOp, buffers);

  for (auto tile : targetOp.getOps<TileOp>())
    if (tile.colIndex() == tileCol && tile.rowIndex() == tileRow) {
      TileID srcCoord = {tile.colIndex(), tile.rowIndex()};
      const auto &targetModel = getTargetModel(tile);

      // Figure out how much memory we have left for random allocations
      auto core = tile.getCoreOp();
      int max = core.getStackSize();
      for (auto buf : buffers[tiles[srcCoord]]) {
        int bufferBaseAddr = getBufferBaseAddress(buf);
        int numBytes = buf.getAllocationSize();
        max = std::max(max, bufferBaseAddr + numBytes);
      }
      int origin = targetModel.getMemInternalBaseAddress(srcCoord) + max;
      int length = targetModel.getLocalMemorySize() - max;
      output << R"THESCRIPT(
MEMORY
{
   program (RX) : ORIGIN = 0, LENGTH = 0x0020000
)THESCRIPT";
      output << "   data (!RX) : ORIGIN = 0x" << llvm::utohexstr(origin)
             << ", LENGTH = 0x" << llvm::utohexstr(length);
      output << R"THESCRIPT(
}
ENTRY(_main_init)
SECTIONS
{
  . = 0x0;
  .text : {
     /* the _main_init symbol from me_basic.o has to come at address zero. */
     *me_basic.o(.text)
     . = 0x200;
     _ctors_start = .;
     _init_array_start = .;
     KEEP(SORT(*.init_array))
     _ctors_end = .;
     _init_array_end = .;
     _dtors_start = .;
     _dtors_end = .;
     *(.text)
  } > program
  .data : {
     *(.data*);
     *(.rodata*)
  } > data
)THESCRIPT";
      auto doBuffer = [&](std::optional<TileID> tile, int offset,
                          std::string dir) {
        if (tile) {
          if (tiles.count(*tile))
            for (auto buf : buffers[tiles[*tile]])
              writeLDScriptMap(output, buf, offset);
        } else {
          output << "/* No tile with memory exists to the " << dir << ". */\n";
          output << ". = 0x" << llvm::utohexstr(offset) << ";\n";
          uint32_t localMemSize = targetModel.getLocalMemorySize();
          output << ". += 0x" << llvm::utohexstr(localMemSize) << ";\n";
        }
      };

      // Stack
      output << ". = 0x"
             << llvm::utohexstr(targetModel.getMemInternalBaseAddress(srcCoord))
             << ";\n";
      output << "_sp_start_value_DM_stack = .;\n";

      if (auto core = tile.getCoreOp())
        output << ". += 0x" << llvm::utohexstr(core.getStackSize())
               << "; /* stack */\n";
      else
        output << "/* no stack allocated */\n";

      doBuffer(targetModel.getMemSouth(srcCoord),
               targetModel.getMemSouthBaseAddress(), std::string("south"));
      doBuffer(targetModel.getMemWest(srcCoord),
               targetModel.getMemWestBaseAddress(), std::string("west"));
      doBuffer(targetModel.getMemNorth(srcCoord),
               targetModel.getMemNorthBaseAddress(), std::string("north"));
      doBuffer(targetModel.getMemEast(srcCoord),
               targetModel.getMemEastBaseAddress(), std::string("east"));

      output << "  .bss : { *(.bss) } > data\n";
      output << "  .bss.DMb.4 : { *(.bss.DMb.4) } > data\n";
      output << "}\n";
      if (auto coreOp = tile.getCoreOp()) {
        if (auto fileAttr = coreOp.getLinkWith())
          output << "INPUT(" << fileAttr.value().str() << ")\n";

        output << "PROVIDE(_main = core_" << tile.getCol() << "_"
               << tile.getRow() << ");\n";
      }
    }
  return success();
}
//===- AIETargetNPU.cpp -----------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2023 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#include <vector>

#include "AIETargets.h"
#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIEX/IR/AIEXDialect.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Format.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Tools/mlir-translate/MlirTranslateMain.h"

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIE;
using namespace xilinx::AIEX;

#define TXN_OPC_WRITE 0x0
#define TXN_OPC_BLOCKWRITE 0x1
#define TXN_OPC_TCT 0x80
#define TXN_OPC_DDR_PATCH 0x81

namespace {

// Example:
// - instructions = {3,4,5}
// - tailSize = 2
// instructions becomes {3,4,5,0,0} and
// a mutable reference to the tail {0,0} is returned.
llvm::MutableArrayRef<uint32_t> reserveAndGetTail(
    std::vector<uint32_t> &instructions, uint64_t tailSize) {
  auto oldSize = instructions.size();
  auto newSize = oldSize + tailSize;
  instructions.resize(newSize, 0);
  return llvm::MutableArrayRef<uint32_t>(instructions.data() + oldSize,
                                         tailSize);
}

void appendSync(std::vector<uint32_t> &instructions, NpuSyncOp op) {
  auto words = reserveAndGetTail(instructions, 4);

  // XAIE_IO_CUSTOM_OP_TCT
  words[0] = TXN_OPC_TCT;

  words[1] = words.size() * sizeof(uint32_t);  // Operation Size

  words[2] |= static_cast<uint32_t>(op.getDirection()) & 0xff;
  words[2] |= (op.getRow() & 0xff) << 8;
  words[2] |= (op.getColumn() & 0xff) << 16;

  words[3] |= (op.getRowNum() & 0xff) << 8;
  words[3] |= (op.getColumnNum() & 0xff) << 16;
  words[3] |= (op.getChannel() & 0xff) << 24;
}

void appendWrite32(std::vector<uint32_t> &instructions, NpuWrite32Op op) {
  auto words = reserveAndGetTail(instructions, 6);
  const AIETargetModel &tm = op->getParentOfType<DeviceOp>().getTargetModel();

  // XAIE_IO_WRITE
  words[0] = TXN_OPC_WRITE;
  words[1] = 0;
  words[2] = op.getAddress();
  auto col = op.getColumn();
  auto row = op.getRow();
  if (col && row)
    words[2] = ((*col & 0xff) << tm.getColumnShift()) |
               ((*row & 0xff) << tm.getRowShift()) | (words[2] & 0xFFFFF);
  words[3] = 0;
  words[4] = op.getValue();                    // Value
  words[5] = words.size() * sizeof(uint32_t);  // Operation Size
}

void appendAddressPatch(std::vector<uint32_t> &instructions,
                        NpuAddressPatchOp op) {
  auto words = reserveAndGetTail(instructions, 12);

  // XAIE_IO_CUSTOM_OP_DDR_PATCH
  words[0] = TXN_OPC_DDR_PATCH;
  words[1] = words.size() * sizeof(uint32_t);  // Operation Size

  words[6] = op.getAddr();
  words[7] = 0;

  words[8] = op.getArgIdx();
  words[9] = 0;

  words[10] = op.getArgPlus();
  words[11] = 0;
}

void appendWriteBdShimTile(std::vector<uint32_t> &instructions,
                           NpuWriteBdOp op) {
  auto words = reserveAndGetTail(instructions, 12);
  const AIETargetModel &tm = op->getParentOfType<DeviceOp>().getTargetModel();

  // XAIE_IO_BLOCKWRITE
  words[0] = TXN_OPC_BLOCKWRITE;
  words[1] = 0;

  // RegOff
  auto bd_id = op.getBdId();
  uint32_t bd_addr = (op.getColumn() << tm.getColumnShift()) |
                     (op.getRow() << tm.getRowShift()) |
                     (0x1D000 + bd_id * 0x20);
  words[2] = bd_addr;                          // ADDR
  words[3] = words.size() * sizeof(uint32_t);  // Operation Size

  // DMA_BDX_0
  words[4] = op.getBufferLength();

  // DMA_BDX_1
  words[5] = op.getBufferOffset();

  // DMA_BDX_2
  // En Packet , OoO BD ID , Packet ID , Packet Type
  words[6] |= (op.getEnablePacket() & 0x1) << 30;
  words[6] |= (op.getOutOfOrderId() & 0x3f) << 24;
  words[6] |= (op.getPacketId() & 0x1f) << 19;
  words[6] |= (op.getPacketType() & 0x7) << 16;

  // DMA_BDX_3
  // TODO: Secure Access
  words[7] |= (op.getD0Size() & 0x3ff) << 20;
  words[7] |= op.getD0Stride() & 0xfffff;

  // DMA_BDX_4
  words[8] = 0x80000000;  // burst length;
  words[8] |= (op.getD1Size() & 0x3ff) << 20;
  words[8] |= op.getD1Stride() & 0xfffff;

  // DMA_BDX_5
  // TODO: SIMID, AxCache, AXQoS
  words[9] = op.getD2Stride() & 0xfffff;

  // DMA_BDX_6
  words[10] |= (op.getIterationCurrent() & 0x3f) << 26;
  words[10] |= (op.getIterationSize() & 0x3f) << 20;
  words[10] |= op.getIterationStride() & 0xfffff;

  // DMA_BDX_7
  // TODO: TLAST Suppress
  words[11] |= (op.getNextBd() & 0xf) << 27;
  words[11] |= (op.getUseNextBd() & 0x1) << 26;
  words[11] |= (op.getValidBd() & 0x1) << 25;
  words[11] |= (op.getLockRelVal() & 0xef) << 18;
  words[11] |= (op.getLockRelId() & 0xf) << 13;
  words[11] |= (op.getLockAcqEnable() & 0x1) << 12;
  words[11] |= (op.getLockAcqVal() & 0xef) << 5;
  words[11] |= op.getLockAcqId() & 0xf;
}

}  // namespace

std::vector<uint32_t> xilinx::AIE::AIETranslateToNPU(ModuleOp module) {
  std::vector<uint32_t> instructions;

  auto words = reserveAndGetTail(instructions, 4);

  // setup txn header
  words[0] = 0x06030100;
  words[1] = 0x00000105;

  DeviceOp deviceOp = *module.getOps<DeviceOp>().begin();
  auto funcOps = deviceOp.getOps<func::FuncOp>();
  int count = 0;
  for (auto f : funcOps) {
    if (f.isDeclaration()) continue;
    Block &entry = f.getRegion().front();
    for (auto &o : entry) {
      llvm::TypeSwitch<Operation *>(&o)
          .Case<NpuSyncOp>([&](auto op) {
            count++;
            appendSync(instructions, op);
          })
          .Case<NpuWrite32Op>([&](auto op) {
            count++;
            appendWrite32(instructions, op);
          })
          .Case<NpuAddressPatchOp>([&](auto op) {
            count++;
            appendAddressPatch(instructions, op);
          })
          .Case<NpuWriteBdOp>([&](auto op) {
            count++;
            appendWriteBdShimTile(instructions, op);
          });
    }
  }

  // write size fields of the txn header
  instructions[2] = count;
  instructions[3] = instructions.size() * sizeof(uint32_t);
  return instructions;
}

LogicalResult xilinx::AIE::AIETranslateToNPU(ModuleOp module,
                                             raw_ostream &output) {
  auto instructions = AIETranslateToNPU(module);
  for (auto w : instructions) output << llvm::format("%08X\n", w);
  return success();
}