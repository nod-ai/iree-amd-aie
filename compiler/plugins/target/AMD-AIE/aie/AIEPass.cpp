//===- AIEAssignBufferAddressesBasic.cpp -------------------------------------*-
// C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2019 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

#include <numeric>
#include <set>

#include "Passes.h"
#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIEX/IR/AIEXDialect.h"
#include "d_ary_heap.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_os_ostream.h"
#include "mlir/Analysis/TopologicalSortUtils.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Iterators.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Tools/mlir-translate/MlirTranslateMain.h"
#include "mlir/Transforms/DialectConversion.h"

#define DEBUG_TYPE "aie-pass"

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIE;

template <typename DerivedT>
class AIEAssignBufferAddressesPassBasicBase
    : public ::mlir::OperationPass<DeviceOp> {
 public:
  using Base = AIEAssignBufferAddressesPassBasicBase;

  AIEAssignBufferAddressesPassBasicBase()
      : ::mlir::OperationPass<DeviceOp>(::mlir::TypeID::get<DerivedT>()) {}
  AIEAssignBufferAddressesPassBasicBase(
      const AIEAssignBufferAddressesPassBasicBase &other)
      : ::mlir::OperationPass<DeviceOp>(other) {}
  AIEAssignBufferAddressesPassBasicBase &operator=(
      const AIEAssignBufferAddressesPassBasicBase &) = delete;
  AIEAssignBufferAddressesPassBasicBase(
      AIEAssignBufferAddressesPassBasicBase &&) = delete;
  AIEAssignBufferAddressesPassBasicBase &operator=(
      AIEAssignBufferAddressesPassBasicBase &&) = delete;
  ~AIEAssignBufferAddressesPassBasicBase() = default;

  /// Returns the command-line argument attached to this pass.
  static constexpr ::llvm::StringLiteral getArgumentName() {
    return ::llvm::StringLiteral("aie-assign-buffer-addresses-basic");
  }
  ::llvm::StringRef getArgument() const override {
    return "aie-assign-buffer-addresses-basic";
  }

  ::llvm::StringRef getDescription() const override {
    return "Assign memory locations for buffers in each tile";
  }

  /// Returns the derived pass name.
  static constexpr ::llvm::StringLiteral getPassName() {
    return ::llvm::StringLiteral("AIEAssignBufferAddressesBasic");
  }
  ::llvm::StringRef getName() const override {
    return "AIEAssignBufferAddressesBasic";
  }

  /// Support isa/dyn_cast functionality for the derived pass class.
  static bool classof(const ::mlir::Pass *pass) {
    return pass->getTypeID() == ::mlir::TypeID::get<DerivedT>();
  }

  /// A clone method to create a copy of this pass.
  std::unique_ptr<::mlir::Pass> clonePass() const override {
    return std::make_unique<DerivedT>(*static_cast<const DerivedT *>(this));
  }

  /// Register the dialects that must be loaded in the context before this pass.
  void getDependentDialects(::mlir::DialectRegistry &registry) const override {}

  /// Explicitly declare the TypeID for this class. We declare an explicit
  /// private instantiation because Pass classes should only be visible by the
  /// current library.
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      AIEAssignBufferAddressesPassBasicBase<DerivedT>)
};

struct AIEAssignBufferAddressesPassBasic
    : AIEAssignBufferAddressesPassBasicBase<AIEAssignBufferAddressesPassBasic> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<func::FuncDialect>();
    registry.insert<AIEDialect>();
  }

  void runOnOperation() override {
    DeviceOp device = getOperation();
    OpBuilder builder = OpBuilder::atBlockEnd(device.getBody());
    // Make sure all the buffers have a name
    int counter = 0;
    device.walk<WalkOrder::PreOrder>([&](BufferOp buffer) {
      if (!buffer.hasName()) {
        std::string name = "_anonymous";
        name += std::to_string(counter++);
        buffer->setAttr(SymbolTable::getSymbolAttrName(),
                        builder.getStringAttr(name));
      }
    });

    for (auto tile : device.getOps<TileOp>()) {
      const auto &targetModel = getTargetModel(tile);
      int maxDataMemorySize = 0;
      if (tile.isMemTile())
        maxDataMemorySize = targetModel.getMemTileSize();
      else
        maxDataMemorySize = targetModel.getLocalMemorySize();
      SmallVector<BufferOp, 4> buffers;
      // Collect all the buffers for this tile.
      device.walk<WalkOrder::PreOrder>([&](BufferOp buffer) {
        if (buffer.getTileOp() == tile) buffers.push_back(buffer);
      });
      // Sort by allocation size.
      std::sort(buffers.begin(), buffers.end(), [](BufferOp a, BufferOp b) {
        return a.getAllocationSize() > b.getAllocationSize();
      });

      // Address range owned by the MemTile is 0x80000.
      // Address range owned by the tile is 0x8000,
      // but we need room at the bottom for stack.
      int stacksize = 0;
      int address = 0;
      if (auto core = tile.getCoreOp()) {
        stacksize = core.getStackSize();
        address += stacksize;
      }

      for (auto buffer : buffers) {
        if (buffer.getAddress())
          buffer->emitWarning("Overriding existing address");
        buffer.setAddress(address);
        address += buffer.getAllocationSize();
      }

      if (address > maxDataMemorySize) {
        InFlightDiagnostic error =
            tile.emitOpError("allocated buffers exceeded available memory\n");
        auto &note = error.attachNote() << "MemoryMap:\n";
        auto printbuffer = [&](StringRef name, int address, int size) {
          note << "\t" << name << " \t"
               << ": 0x" << llvm::utohexstr(address) << "-0x"
               << llvm::utohexstr(address + size - 1) << " \t(" << size
               << " bytes)\n";
        };
        if (stacksize > 0)
          printbuffer("(stack)", 0, stacksize);
        else
          error << "(no stack allocated)\n";

        for (auto buffer : buffers) {
          assert(buffer.getAddress().has_value() &&
                 "buffer must have address assigned");
          printbuffer(buffer.name(), buffer.getAddress().value(),
                      buffer.getAllocationSize());
        }
        return signalPassFailure();
      }
    }
  }
};

std::unique_ptr<OperationPass<DeviceOp>>
AIE::createAIEAssignBufferAddressesBasicPass() {
  return std::make_unique<AIEAssignBufferAddressesPassBasic>();
}

void xilinx::AIE::registerAIEAssignBufferAddressesBasic() {
  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return xilinx::AIE::createAIEAssignBufferAddressesBasicPass();
  });
}
//===- AIEAssignBufferDescriptorIDs.cpp -------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2024 Advanced Micro Devices Inc.
//
//===----------------------------------------------------------------------===//

#define EVEN_BD_ID_START 0
#define ODD_BD_ID_START 24

struct BdIdGenerator {
  BdIdGenerator(int col, int row, const AIETargetModel &targetModel)
      : col(col), row(row), isMemTile(targetModel.isMemTile(col, row)) {}

  int32_t nextBdId(int channelIndex) {
    int32_t bdId = isMemTile && channelIndex & 1 ? oddBdId++ : evenBdId++;
    while (bdIdAlreadyAssigned(bdId))
      bdId = isMemTile && channelIndex & 1 ? oddBdId++ : evenBdId++;
    assignBdId(bdId);
    return bdId;
  }

  void assignBdId(int32_t bdId) {
    assert(!alreadyAssigned.count(bdId) && "bdId has already been assigned");
    alreadyAssigned.insert(bdId);
  }

  bool bdIdAlreadyAssigned(int32_t bdId) { return alreadyAssigned.count(bdId); }

  int col;
  int row;
  int oddBdId = ODD_BD_ID_START;
  int evenBdId = EVEN_BD_ID_START;
  bool isMemTile;
  std::set<int32_t> alreadyAssigned;
};

struct AIEAssignBufferDescriptorIDsPass
    : xilinx::AIE::impl::AIEAssignBufferDescriptorIDsBase<
          AIEAssignBufferDescriptorIDsPass> {
  void runOnOperation() override {
    DeviceOp targetOp = getOperation();
    const AIETargetModel &targetModel = targetOp.getTargetModel();

    auto memOps = llvm::to_vector_of<TileElement>(targetOp.getOps<MemOp>());
    llvm::append_range(memOps, targetOp.getOps<MemTileDMAOp>());
    llvm::append_range(memOps, targetOp.getOps<ShimDMAOp>());
    for (TileElement memOp : memOps) {
      int col = memOp.getTileID().col;
      int row = memOp.getTileID().row;

      BdIdGenerator gen(col, row, targetModel);
      memOp->walk<WalkOrder::PreOrder>([&](DMABDOp bd) {
        if (bd.getBdId().has_value()) gen.assignBdId(bd.getBdId().value());
      });

      auto dmaOps = memOp.getOperation()->getRegion(0).getOps<DMAOp>();
      if (!dmaOps.empty()) {
        for (auto dmaOp : dmaOps) {
          auto bdRegions = dmaOp.getBds();
          for (auto &bdRegion : bdRegions) {
            auto &block = bdRegion.getBlocks().front();
            DMABDOp bd = *block.getOps<DMABDOp>().begin();
            if (bd.getBdId().has_value())
              assert(
                  gen.bdIdAlreadyAssigned(bd.getBdId().value()) &&
                  "bdId assigned by user but not found during previous walk");
            else
              bd.setBdId(gen.nextBdId(dmaOp.getChannelIndex()));
          }
        }
      } else {
        DenseMap<Block *, int> blockChannelMap;
        // Associate with each block the channel index specified by the
        // dma_start
        for (Block &block : memOp.getOperation()->getRegion(0))
          for (auto op : block.getOps<DMAStartOp>()) {
            int chNum = op.getChannelIndex();
            blockChannelMap[&block] = chNum;
            Block *dest = op.getDest();
            while (dest) {
              blockChannelMap[dest] = chNum;
              if (dest->hasNoSuccessors()) break;
              dest = dest->getSuccessors()[0];
              if (blockChannelMap.contains(dest)) dest = nullptr;
            }
          }

        for (Block &block : memOp.getOperation()->getRegion(0)) {
          if (block.getOps<DMABDOp>().empty()) continue;
          assert(blockChannelMap.count(&block));
          DMABDOp bd = (*block.getOps<DMABDOp>().begin());
          if (bd.getBdId().has_value())
            assert(gen.bdIdAlreadyAssigned(bd.getBdId().value()) &&
                   "bdId assigned by user but not found during previous walk");
          else
            bd.setBdId(gen.nextBdId(blockChannelMap[&block]));
        }
      }
    }
    for (TileElement memOp : memOps) {
      auto dmaOps = memOp.getOperation()->getRegion(0).getOps<DMAOp>();
      if (!dmaOps.empty()) {
        for (auto dmaOp : dmaOps) {
          auto bdRegions = dmaOp.getBds();
          for (auto *bdRegionIt = bdRegions.begin();
               bdRegionIt != bdRegions.end();) {
            auto &block = bdRegionIt->getBlocks().front();
            DMABDOp bd = *block.getOps<DMABDOp>().begin();
            std::optional<int> nextBdId;
            if (++bdRegionIt != bdRegions.end())
              nextBdId =
                  (*bdRegionIt->getBlocks().front().getOps<DMABDOp>().begin())
                      .getBdId();
            else if (dmaOp.getLoop())
              nextBdId = (*bdRegions.front()
                               .getBlocks()
                               .front()
                               .getOps<DMABDOp>()
                               .begin())
                             .getBdId();
            bd.setNextBdId(nextBdId);
          }
        }
      } else {
        DenseMap<Block *, int> blockBdIdMap;
        for (Block &block : memOp.getOperation()->getRegion(0)) {
          if (block.getOps<DMABDOp>().empty()) continue;
          DMABDOp bd = *block.getOps<DMABDOp>().begin();
          assert(bd.getBdId().has_value() &&
                 "DMABDOp should have bd_id assigned by now");
          blockBdIdMap[&block] = bd.getBdId().value();
        }

        for (Block &block : memOp.getOperation()->getRegion(0)) {
          if (block.getOps<DMABDOp>().empty()) continue;
          DMABDOp bd = *block.getOps<DMABDOp>().begin();
          std::optional<int> nextBdId;
          if (block.getNumSuccessors()) {
            assert(llvm::range_size(block.getSuccessors()) == 1 &&
                   "should have only one successor block");
            Block *nextBlock = block.getSuccessor(0);
            if (!blockBdIdMap.contains(nextBlock))
              assert(nextBlock->getOperations().size() == 1 &&
                     isa<EndOp>(nextBlock->getOperations().front()) &&
                     "bb that's not in blockMap can only have aie.end");
            else
              nextBdId = blockBdIdMap[nextBlock];
            bd.setNextBdId(nextBdId);
          }
        }
      }
    }
  }
};

std::unique_ptr<OperationPass<DeviceOp>>
AIE::createAIEAssignBufferDescriptorIDsPass() {
  return std::make_unique<AIEAssignBufferDescriptorIDsPass>();
}  //===- AIEAssignLockIDs.cpp -------------------------------------*- C++
   //-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2022 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

// This pass aims to assign lockIDs to AIE.lock operations. The lockID is
// numbered from the most recent AIE.lock within the same tile. If the lockID
// exceeds the number of locks on the tile, the pass generates an error and
// terminates. AIE.lock operations for different tiles are numbered
// independently. If there are existing lock IDs, this pass is idempotent
// and only assigns lock IDs to locks without an ID.

struct AIEAssignLockIDsPass
    : xilinx::AIE::impl::AIEAssignLockIDsBase<AIEAssignLockIDsPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<func::FuncDialect>();
    registry.insert<AIEDialect>();
  }

  void runOnOperation() override {
    DeviceOp device = getOperation();
    OpBuilder rewriter = OpBuilder::atBlockEnd(device.getBody());

    // All of the lock ops on a tile, separated into ops which have been
    // assigned to a lock, and ops which have not.
    struct TileLockOps {
      DenseSet<int> assigned;
      SmallVector<LockOp> unassigned;
    };

    DenseMap<TileOp, TileLockOps> tileToLocks;

    // Construct data structure storing locks by tile.
    device.walk<WalkOrder::PreOrder>([&](LockOp lockOp) {
      TileOp tileOp = lockOp.getTileOp();
      if (lockOp.getLockID().has_value()) {
        auto lockID = lockOp.getLockID().value();
        auto iter = tileToLocks.find(tileOp);
        if (iter == tileToLocks.end())
          tileToLocks.insert({tileOp, {{lockID}, /* unassigned = */ {}}});
        else {
          if (iter->second.assigned.find(lockID) !=
              iter->second.assigned.end()) {
            auto diag = lockOp->emitOpError("is assigned to the same lock (")
                        << lockID << ") as another op.";
            diag.attachNote(tileOp.getLoc())
                << "tile has lock ops assigned to same lock.";
            return signalPassFailure();
          }
          iter->second.assigned.insert(lockID);
        }
      } else {
        auto iter = tileToLocks.find(tileOp);
        if (iter == tileToLocks.end())
          tileToLocks.insert({tileOp, {/* assigned = */ {}, {lockOp}}});
        else
          iter->second.unassigned.push_back(lockOp);
      }
    });

    // IR mutation: assign locks to all unassigned lock ops.
    for (auto [tileOp, locks] : tileToLocks) {
      const auto locksPerTile =
          getTargetModel(tileOp).getNumLocks(tileOp.getCol(), tileOp.getRow());
      uint32_t nextID = 0;
      for (auto lockOp : locks.unassigned) {
        while (nextID < locksPerTile &&
               (locks.assigned.find(nextID) != locks.assigned.end())) {
          ++nextID;
        }
        if (nextID == locksPerTile) {
          mlir::InFlightDiagnostic diag =
              lockOp->emitOpError("not allocated a lock.");
          diag.attachNote(tileOp.getLoc()) << "because only " << locksPerTile
                                           << " locks available in this tile.";
          return signalPassFailure();
        }
        lockOp.setLockIDAttr(rewriter.getI32IntegerAttr(nextID));
        ++nextID;
      }
    }
  }
};

std::unique_ptr<OperationPass<DeviceOp>> AIE::createAIEAssignLockIDsPass() {
  return std::make_unique<AIEAssignLockIDsPass>();
}

//===- AIECoreToStandard.cpp ------------------------------------*- C++
//-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2019 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

using namespace mlir;
using namespace mlir::vector;
using namespace xilinx;
using namespace xilinx::AIE;

static StringRef getArchIntrinsicString(AIEArch arch) {
  switch (arch) {
    case AIEArch::AIE1:
      return "aie";
    case AIEArch::AIE2:
      return "aie2";
  }
  llvm::report_fatal_error("unsupported arch");
}

typedef std::tuple<const char *, std::vector<Type>, std::vector<Type>>
    IntrinsicDecl;
typedef std::vector<IntrinsicDecl> IntrinsicDecls;

static auto getAIE1Intrinsics(OpBuilder &builder) {
  Type int32Type = IntegerType::get(builder.getContext(), 32);
  Type int128Type = IntegerType::get(builder.getContext(), 128);
  Type int384Type = IntegerType::get(builder.getContext(), 384);
  Type floatType = FloatType::getF32(builder.getContext());

  // Note that not all of these are valid for a particular design, or needed.
  // For right now, we will just accept the noise.
  IntrinsicDecls functions = {
      {"debug_i32", {int32Type}, {}},
      {"llvm.aie.event0", {}, {}},
      {"llvm.aie.event1", {}, {}},
      {"llvm.aie.put.ms",
       {int32Type, int32Type},
       {}},  //(%channel, %value) -> ()
      {"llvm.aie.put.wms",
       {int32Type, int128Type},
       {}},  //(%channel, %value) -> ()
      {"llvm.aie.put.fms",
       {int32Type, floatType},
       {}},                                           //(%channel, %value) -> ()
      {"llvm.aie.get.ss", {int32Type}, {int32Type}},  //(%channel, %value) -> ()
      {"llvm.aie.get.wss",
       {int32Type},
       {int128Type}},  //(%channel, %value) -> ()
      {"llvm.aie.get.fss",
       {int32Type},
       {floatType}},  //(%channel, %value) -> ()
      {"llvm.aie.put.mcd", {int384Type}, {}},
      {"llvm.aie.get.scd", {}, {int384Type}},
      {"llvm.aie.lock.acquire.reg",
       {int32Type, int32Type},
       {}},  //(%lock_id, %lock_val) -> ()
      {"llvm.aie.lock.release.reg",
       {int32Type, int32Type},
       {}},  //(%lock_id, %lock_val) -> ()
  };
  return functions;
}

static auto getAIE2Intrinsics(OpBuilder &builder) {
  Type int32Type = IntegerType::get(builder.getContext(), 32);
  Type accType = VectorType::get({16}, int32Type);
  IntrinsicDecls functions = {
      {"debug_i32", {int32Type}, {}},
      {"llvm.aie2.put.ms",
       {int32Type, int32Type},
       {}},  //(%value, %tlast) -> ()
      {"llvm.aie2.get.ss",
       {},
       {int32Type, int32Type}},  //() -> (%value, %tlast)
      {"llvm.aie2.mcd.write.vec",
       {accType, int32Type},
       {}},  // (%value, %enable) -> ()
      {"llvm.aie2.scd.read.vec",
       {int32Type},
       {accType}},  // (%enable) -> (%value)
      {"llvm.aie2.acquire",
       {int32Type, int32Type},
       {}},  //(%lock_id, %lock_val) -> ()
      {"llvm.aie2.release",
       {int32Type, int32Type},
       {}},  //(%lock_id, %lock_val) -> ()
  };
  return functions;
}

static void declareAIEIntrinsics(AIEArch arch, OpBuilder &builder) {
  auto registerIntrinsics = [&builder](IntrinsicDecls functions) {
    for (auto &i : functions) {
      auto [name, argTypes, retTypes] = i;
      builder
          .create<func::FuncOp>(
              builder.getUnknownLoc(), name,
              FunctionType::get(builder.getContext(), argTypes, retTypes))
          .setPrivate();
    }
  };
  switch (arch) {
    case AIEArch::AIE1:
      registerIntrinsics(getAIE1Intrinsics(builder));
      return;
    case AIEArch::AIE2:
      registerIntrinsics(getAIE2Intrinsics(builder));
      return;
  }
  llvm::report_fatal_error("unsupported arch");
}

template <typename MyAIEOp>
struct AIEOpRemoval : OpConversionPattern<MyAIEOp> {
  using OpConversionPattern<MyAIEOp>::OpConversionPattern;
  using OpAdaptor = typename MyAIEOp::Adaptor;
  ModuleOp &module;

  AIEOpRemoval(MLIRContext *context, ModuleOp &m, PatternBenefit benefit = 1)
      : OpConversionPattern<MyAIEOp>(context, benefit), module(m) {}

  LogicalResult matchAndRewrite(
      MyAIEOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    rewriter.eraseOp(op);
    return success();
  }
};

struct AIEDebugOpToStdLowering : OpConversionPattern<DebugOp> {
  using OpConversionPattern::OpConversionPattern;
  ModuleOp &module;

  AIEDebugOpToStdLowering(MLIRContext *context, ModuleOp &m,
                          PatternBenefit benefit = 1)
      : OpConversionPattern(context, benefit), module(m) {}

  LogicalResult matchAndRewrite(
      DebugOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    std::string funcName = "debug_i32";
    auto func = module.lookupSymbol<func::FuncOp>(funcName);
    if (!func)
      return op.emitOpError("Could not find the intrinsic function ")
             << funcName;
    SmallVector<Value, 1> args;
    args.push_back(op.getArg());
    rewriter.create<func::CallOp>(rewriter.getUnknownLoc(), func, args);
    rewriter.eraseOp(op);
    return success();
  }
};

struct AIEPutStreamToStdLowering : OpConversionPattern<PutStreamOp> {
  using OpConversionPattern::OpConversionPattern;
  ModuleOp &module;

  AIEPutStreamToStdLowering(MLIRContext *context, ModuleOp &m,
                            PatternBenefit benefit = 1)
      : OpConversionPattern(context, benefit), module(m) {}

  LogicalResult matchAndRewrite(
      PutStreamOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto device = op->getParentOfType<DeviceOp>();
    const auto &targetModel = device.getTargetModel();
    std::string funcName;
    if (targetModel.getTargetArch() == AIEArch::AIE1)
      funcName = "llvm.aie.put.";
    else
      funcName = "llvm.aie2.put.";

    if (op.isWideStream())
      funcName += "wms";
    else if (op.isFloatStream())
      funcName += "fms";
    else
      funcName += "ms";

    auto putMSFunc = module.lookupSymbol<func::FuncOp>(funcName);
    if (!putMSFunc)
      return op.emitOpError("Could not find the intrinsic function ")
             << funcName;
    SmallVector<Value, 2> args;
    if (targetModel.getTargetArch() == AIEArch::AIE1) {
      args.push_back(op.getChannel());
      args.push_back(op.getStreamValue());
    } else {
      args.push_back(op.getStreamValue());
      args.push_back(rewriter.create<arith::ConstantOp>(
          op.getLoc(), IntegerType::get(rewriter.getContext(), 32),
          rewriter.getI32IntegerAttr(0)));  // tlast
    }
    rewriter.create<func::CallOp>(rewriter.getUnknownLoc(), putMSFunc, args);
    rewriter.eraseOp(op);
    return success();
  }
};

struct AIEGetStreamToStdLowering : OpConversionPattern<GetStreamOp> {
  using OpConversionPattern::OpConversionPattern;
  ModuleOp &module;

  AIEGetStreamToStdLowering(MLIRContext *context, ModuleOp &m,
                            PatternBenefit benefit = 1)
      : OpConversionPattern(context, benefit), module(m) {}

  LogicalResult matchAndRewrite(
      GetStreamOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto device = op->getParentOfType<DeviceOp>();
    const auto &targetModel = device.getTargetModel();
    std::string funcName;
    if (targetModel.getTargetArch() == AIEArch::AIE1)
      funcName = "llvm.aie.get.";
    else
      funcName = "llvm.aie2.get.";

    if (op.isWideStream())
      funcName += "wss";
    else if (op.isFloatStream())
      funcName += "fss";
    else
      funcName += "ss";

    auto getSSFunc = module.lookupSymbol<func::FuncOp>(funcName);
    if (!getSSFunc)
      return op.emitOpError("Could not find the intrinsic function ")
             << funcName;
    SmallVector<Value, 2> args;
    if (targetModel.getTargetArch() == AIEArch::AIE1)
      args.push_back(op.getChannel());
    auto getSSCall = rewriter.create<func::CallOp>(rewriter.getUnknownLoc(),
                                                   getSSFunc, args);
    rewriter.replaceOp(op, getSSCall.getResult(0));
    // Capture TLAST in AIEv2?
    return success();
  }
};

struct AIEPutCascadeToStdLowering : OpConversionPattern<PutCascadeOp> {
  using OpConversionPattern::OpConversionPattern;
  ModuleOp &module;

  AIEPutCascadeToStdLowering(MLIRContext *context, ModuleOp &m,
                             PatternBenefit benefit = 1)
      : OpConversionPattern(context, benefit), module(m) {}

  LogicalResult matchAndRewrite(
      PutCascadeOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto device = op->getParentOfType<DeviceOp>();
    const auto &targetModel = device.getTargetModel();
    std::string funcName;
    if (targetModel.getTargetArch() == AIEArch::AIE1)
      funcName = "llvm.aie.put.mcd";
    else
      funcName = "llvm.aie2.mcd.write.vec";
    auto putMCDFunc = module.lookupSymbol<func::FuncOp>(funcName);
    if (!putMCDFunc)
      return op.emitOpError("Could not find the intrinsic function ")
             << funcName;
    SmallVector<Value, 2> args;
    args.push_back(op.getCascadeValue());
    if (targetModel.getTargetArch() == AIEArch::AIE2)
      args.push_back(rewriter.create<arith::ConstantOp>(
          op.getLoc(), IntegerType::get(rewriter.getContext(), 32),
          rewriter.getI32IntegerAttr(1)));  // enable

    rewriter.create<func::CallOp>(rewriter.getUnknownLoc(), putMCDFunc, args);
    rewriter.eraseOp(op);
    return success();
  }
};

struct AIEGetCascadeToStdLowering : OpConversionPattern<GetCascadeOp> {
  using OpConversionPattern::OpConversionPattern;
  ModuleOp &module;

  AIEGetCascadeToStdLowering(MLIRContext *context, ModuleOp &m,
                             PatternBenefit benefit = 1)
      : OpConversionPattern(context, benefit), module(m) {}

  LogicalResult matchAndRewrite(
      GetCascadeOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto device = op->getParentOfType<DeviceOp>();
    const auto &targetModel = device.getTargetModel();
    std::string funcName;
    if (targetModel.getTargetArch() == AIEArch::AIE1)
      funcName = "llvm.aie.get.scd";
    else
      funcName = "llvm.aie2.scd.read.vec";
    auto getSCDFunc = module.lookupSymbol<func::FuncOp>(funcName);
    if (!getSCDFunc)
      return op.emitOpError("Could not find the intrinsic function ")
             << funcName;
    SmallVector<Value, 2> args;
    if (targetModel.getTargetArch() == AIEArch::AIE2)
      args.push_back(rewriter.create<arith::ConstantOp>(
          op.getLoc(), IntegerType::get(rewriter.getContext(), 32),
          rewriter.getI32IntegerAttr(1)));  // enable

    auto getSCDCall = rewriter.create<func::CallOp>(rewriter.getUnknownLoc(),
                                                    getSCDFunc, args);
    rewriter.replaceOp(op, getSCDCall.getResult(0));
    return success();
  }
};

struct AIEUseLockToStdLowering : OpConversionPattern<UseLockOp> {
  using OpConversionPattern::OpConversionPattern;
  ModuleOp &module;

  AIEUseLockToStdLowering(MLIRContext *context, ModuleOp &m,
                          PatternBenefit benefit = 1)
      : OpConversionPattern(context, benefit), module(m) {}
  LogicalResult matchAndRewrite(
      UseLockOp useLock, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    if (!isa<DeviceOp>(useLock->getParentOp())) {
      auto device = useLock->getParentOfType<DeviceOp>();
      if (!device) {
        return module.emitOpError("Device Not found!");
      }
      const auto &targetModel = device.getTargetModel();

      // Generate the intrinsic name
      std::string funcName;
      if (targetModel.getTargetArch() == AIEArch::AIE1)
        funcName = "llvm.aie.lock.";
      else
        funcName = "llvm.aie2.";
      if (useLock.acquire() || useLock.acquireGE())
        funcName += "acquire";
      else if (useLock.release())
        funcName += "release";
      if (targetModel.getTargetArch() == AIEArch::AIE1) funcName += ".reg";

      auto useLockFunc = module.lookupSymbol<func::FuncOp>(funcName);
      if (!useLockFunc)
        return useLock.emitOpError("Could not find the intrinsic function!");

      SmallVector<Value, 2> args;
      auto lockValue = useLock.getLockValue();

      // AIE2 acquire greater equal is encoded as a negative value.
      if (useLock.acquireGE()) {
        lockValue = -lockValue;
      }
      args.push_back(rewriter.create<arith::IndexCastOp>(
          useLock.getLoc(), IntegerType::get(rewriter.getContext(), 32),
          useLock.getLock()));
      args.push_back(rewriter.create<arith::ConstantOp>(
          useLock.getLoc(), IntegerType::get(rewriter.getContext(), 32),
          rewriter.getI32IntegerAttr(lockValue)));

      rewriter.create<func::CallOp>(rewriter.getUnknownLoc(), useLockFunc,
                                    args);
    }
    rewriter.eraseOp(useLock);
    return success();
  }
};

struct AIEBufferToStandard : OpConversionPattern<BufferOp> {
  using OpConversionPattern::OpConversionPattern;
  ModuleOp &module;
  int tileCol = 0;
  int tileRow = 0;
  AIEBufferToStandard(MLIRContext *context, ModuleOp &m,
                      PatternBenefit benefit = 1, int tileCol = -1,
                      int tileRow = -1)
      : OpConversionPattern(context, benefit),
        module(m),
        tileCol(tileCol),
        tileRow(tileRow) {}
  LogicalResult matchAndRewrite(
      BufferOp buffer, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    rewriter.setInsertionPointToStart(module.getBody());
    auto t = llvm::cast<MemRefType>(buffer.getType());
    int col = llvm::cast<TileOp>(buffer.getTile().getDefiningOp()).getCol();
    int row = llvm::cast<TileOp>(buffer.getTile().getDefiningOp()).getRow();
    auto symName = buffer.name().getValue();
    mlir::ElementsAttr initValue = buffer.getInitialValueAttr();
    // Don't emit initialization for cores that don't "own" the buffer (to
    // prevent duplication in the data section of the elf/object file)
    if ((tileRow != row && tileRow != -1) || (tileCol != col && tileCol != -1))
      initValue = nullptr;
    rewriter.create<memref::GlobalOp>(
        rewriter.getUnknownLoc(), symName, rewriter.getStringAttr("public"),
        buffer.getType(), initValue, /*constant*/ false,
        /*alignment*/ nullptr);

    for (auto &use : make_early_inc_range(buffer.getResult().getUses())) {
      Operation *user = use.getOwner();
      rewriter.setInsertionPoint(user);
      auto allocated = rewriter.create<memref::GetGlobalOp>(
          rewriter.getUnknownLoc(), t, symName);
      // Assume that buffers are aligned so they can be vectorized.
      rewriter.create<memref::AssumeAlignmentOp>(rewriter.getUnknownLoc(),
                                                 allocated, 32);

      use.set(allocated.getResult());
    }

    rewriter.eraseOp(buffer);
    return success();
  }
};

struct AIECoreToStandardFunc : OpConversionPattern<CoreOp> {
  using OpConversionPattern::OpConversionPattern;
  ModuleOp &module;
  IRMapping &mapper;
  DenseMap<Operation *, SmallVector<BufferOp, 4>> &tileToBuffers;
  int tileCol = 0;
  int tileRow = 0;

  AIECoreToStandardFunc(
      MLIRContext *context, ModuleOp &m, IRMapping &mapper,
      DenseMap<Operation *, SmallVector<BufferOp, 4>> &tileToBuffers,
      PatternBenefit benefit = 1, int tileCol = 1, int tileRow = 1)
      : OpConversionPattern(context, benefit),
        module(m),
        mapper(mapper),
        tileToBuffers(tileToBuffers),
        tileCol(tileCol),
        tileRow(tileRow) {}

  LogicalResult matchAndRewrite(
      CoreOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    int col = op.colIndex();
    int row = op.rowIndex();

    // Only pull code for the indicated function
    if ((tileRow != row && tileRow != -1) ||
        (tileCol != col && tileCol != -1)) {
      rewriter.eraseOp(op);
      return success();
    }

    // The parent should be an AIE.device op.
    rewriter.setInsertionPointAfter(op->getParentOp());

    std::string coreName("core_" + std::to_string(col) + "_" +
                         std::to_string(row));
    auto coreFunc = rewriter.create<func::FuncOp>(
        rewriter.getUnknownLoc(), coreName,
        FunctionType::get(rewriter.getContext(), {}, {}));

    rewriter.cloneRegionBefore(op.getBody(), coreFunc.getBody(),
                               coreFunc.getBody().begin(), mapper);

    // Rewrite the AIE.end() op
    coreFunc.getBody().walk([&](Operation *childOp) {
      rewriter.setInsertionPointAfter(childOp);

      if (isa<EndOp>(childOp)) {
        rewriter.create<func::ReturnOp>(rewriter.getUnknownLoc(),
                                        ValueRange({}));
        rewriter.eraseOp(childOp);
      }
    });

    rewriter.eraseOp(op);
    return success();
  }
};

// Move all the ops with OpTy inside device, to just before the device.
template <typename OpTy>
void outlineOps(DeviceOp device) {
  SmallVector<OpTy, 16> ops;
  for (const auto &op : device.getOps<OpTy>()) ops.push_back(op);

  for (const auto &op : ops) op->moveBefore(device);
}

// Lower AIE.event to llvm.aie.event intrinsic
struct AIEEventOpToStdLowering : OpConversionPattern<EventOp> {
  using OpConversionPattern::OpConversionPattern;
  ModuleOp &module;

  AIEEventOpToStdLowering(MLIRContext *context, ModuleOp &m,
                          PatternBenefit benefit = 1)
      : OpConversionPattern(context, benefit), module(m) {}

  LogicalResult matchAndRewrite(
      EventOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    std::string funcName = "llvm.aie.event" + std::to_string(op.getVal());
    auto eventFunc = module.lookupSymbol<func::FuncOp>(funcName);
    if (!eventFunc)
      return op.emitOpError("Could not find the intrinsic function ")
             << funcName;
    rewriter.create<func::CallOp>(rewriter.getUnknownLoc(), eventFunc,
                                  ValueRange({}));
    rewriter.eraseOp(op);
    return success();
  }
};

struct AIECoreToStandardPass
    : xilinx::AIE::impl::AIECoreToStandardBase<AIECoreToStandardPass> {
  void runOnOperation() override {
    ModuleOp m = getOperation();
    OpBuilder builder = OpBuilder::atBlockEnd(m.getBody());

    if (m.getOps<DeviceOp>().empty()) {
      m.emitOpError("expected AIE.device operation at toplevel");
      return signalPassFailure();
    }
    DeviceOp device = *m.getOps<DeviceOp>().begin();
    const auto &targetModel = device.getTargetModel();

    // Ensure that we don't have an incorrect target triple.  This may override
    // some bogus target triple in the original mlir.
    m->setAttr(LLVM::LLVMDialect::getTargetTripleAttrName(),
               builder.getStringAttr(
                   getArchIntrinsicString(targetModel.getTargetArch())));

    DenseMap<Operation *, SmallVector<BufferOp, 4>> tileToBuffers;

    // Populate intrinsic functions
    // Intrinsic information:
    // peano/llvm-project/llvm/lib/Target/AIE/AIEInstrInfo.td Also take a look
    // at the tests: peano/llvm-project/llvm/test/CodeGen/AIE
    builder.setInsertionPointToStart(m.getBody());
    declareAIEIntrinsics(targetModel.getTargetArch(), builder);

    IRMapping mapper;
    ConversionTarget target(getContext());
    target.addLegalDialect<func::FuncDialect>();
    target.addLegalDialect<cf::ControlFlowDialect>();
    target.addLegalDialect<memref::MemRefDialect>();
    target.addLegalDialect<VectorDialect>();
    target.addLegalDialect<arith::ArithDialect>();
    target.addLegalDialect<math::MathDialect>();
    target.addLegalOp<func::FuncOp, ModuleOp>();

    RewritePatternSet patterns(&getContext());
    patterns.add<AIEPutStreamToStdLowering, AIEGetStreamToStdLowering,
                 AIEPutCascadeToStdLowering, AIEGetCascadeToStdLowering,
                 AIEDebugOpToStdLowering, AIEUseLockToStdLowering,
                 AIEEventOpToStdLowering>(m.getContext(), m);

    patterns.add<AIEBufferToStandard>(m.getContext(), m, /*benefit*/ 1, tileCol,
                                      tileRow);
    if (failed(applyPartialConversion(m, target, std::move(patterns))))
      return signalPassFailure();

    RewritePatternSet outlinePatterns(&getContext());
    outlinePatterns.add<AIECoreToStandardFunc>(m.getContext(), m, mapper,
                                               tileToBuffers, /*benefit*/ 1,
                                               tileCol, tileRow);
    if (failed(applyPartialConversion(m, target, std::move(outlinePatterns))))
      return signalPassFailure();

    // Move all the func.func ops and memref.globals from the device to the
    // module
    outlineOps<memref::GlobalOp>(device);
    outlineOps<func::FuncOp>(device);

    RewritePatternSet removepatterns(&getContext());
    removepatterns.add<
        AIEOpRemoval<DeviceOp>, AIEOpRemoval<TileOp>, AIEOpRemoval<FlowOp>,
        AIEOpRemoval<MemOp>, AIEOpRemoval<ShimDMAOp>, AIEOpRemoval<ShimMuxOp>,
        AIEOpRemoval<SwitchboxOp>, AIEOpRemoval<LockOp>, AIEOpRemoval<BufferOp>,
        AIEOpRemoval<ExternalBufferOp>, AIEOpRemoval<ShimDMAAllocationOp>,
        AIEOpRemoval<CascadeFlowOp>, AIEOpRemoval<ConfigureCascadeOp>>(
        m.getContext(), m);

    if (failed(applyPartialConversion(m, target, std::move(removepatterns))))
      return signalPassFailure();
  }
};

std::unique_ptr<OperationPass<ModuleOp>> AIE::createAIECoreToStandardPass() {
  return std::make_unique<AIECoreToStandardPass>();
}
//===- AIECreatePathfindFlows.cpp -------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

namespace {

// allocates channels between switchboxes ( but does not assign them)
// instantiates shim-muxes AND allocates channels ( no need to rip these up in )
struct ConvertFlowsToInterconnect : OpConversionPattern<FlowOp> {
  using OpConversionPattern::OpConversionPattern;
  DeviceOp &device;
  DynamicTileAnalysis &analyzer;
  ConvertFlowsToInterconnect(MLIRContext *context, DeviceOp &d,
                             DynamicTileAnalysis &a, PatternBenefit benefit = 1)
      : OpConversionPattern(context, benefit), device(d), analyzer(a) {}

  LogicalResult match(FlowOp op) const override { return success(); }

  void addConnection(ConversionPatternRewriter &rewriter,
                     // could be a shim-mux or a switchbox.
                     Interconnect op, FlowOp flowOp, WireBundle inBundle,
                     int inIndex, WireBundle outBundle, int outIndex) const {
    Region &r = op.getConnections();
    Block &b = r.front();
    auto point = rewriter.saveInsertionPoint();
    rewriter.setInsertionPoint(b.getTerminator());

    rewriter.create<ConnectOp>(rewriter.getUnknownLoc(), inBundle, inIndex,
                               outBundle, outIndex);

    rewriter.restoreInsertionPoint(point);

    LLVM_DEBUG(llvm::dbgs()
               << "\t\taddConnection() (" << op.colIndex() << ","
               << op.rowIndex() << ") " << stringifyWireBundle(inBundle)
               << inIndex << " -> " << stringifyWireBundle(outBundle)
               << outIndex << "\n");
  }

  void rewrite(FlowOp flowOp, OpAdaptor adaptor,
               ConversionPatternRewriter &rewriter) const override {
    Operation *Op = flowOp.getOperation();

    auto srcTile = cast<TileOp>(flowOp.getSource().getDefiningOp());
    TileID srcCoords = {srcTile.colIndex(), srcTile.rowIndex()};
    auto srcBundle = flowOp.getSourceBundle();
    auto srcChannel = flowOp.getSourceChannel();
    Port srcPort = {srcBundle, srcChannel};

#ifndef NDEBUG
    auto dstTile = cast<TileOp>(flowOp.getDest().getDefiningOp());
    TileID dstCoords = {dstTile.colIndex(), dstTile.rowIndex()};
    auto dstBundle = flowOp.getDestBundle();
    auto dstChannel = flowOp.getDestChannel();
    LLVM_DEBUG(llvm::dbgs()
               << "\n\t---Begin rewrite() for flowOp: (" << srcCoords.col
               << ", " << srcCoords.row << ")" << stringifyWireBundle(srcBundle)
               << srcChannel << " -> (" << dstCoords.col << ", "
               << dstCoords.row << ")" << stringifyWireBundle(dstBundle)
               << dstChannel << "\n\t");
#endif

    // if the flow (aka "net") for this FlowOp hasn't been processed yet,
    // add all switchbox connections to implement the flow
    Switchbox srcSB = {srcCoords.col, srcCoords.row};
    if (PathEndPoint srcPoint = {srcSB, srcPort};
        !analyzer.processedFlows[srcPoint]) {
      SwitchSettings settings = analyzer.flowSolutions[srcPoint];
      // add connections for all the Switchboxes in SwitchSettings
      for (const auto &[curr, setting] : settings) {
        SwitchboxOp swOp = analyzer.getSwitchbox(rewriter, curr.col, curr.row);
        int shimCh = srcChannel;
        // TODO: must reserve N3, N7, S2, S3 for DMA connections
        if (curr == srcSB &&
            analyzer.getTile(rewriter, srcSB.col, srcSB.row).isShimNOCTile()) {
          // shim DMAs at start of flows
          if (srcBundle == WireBundle::DMA) {
            shimCh = srcChannel == 0
                         ? 3
                         : 7;  // must be either DMA0 -> N3 or DMA1 -> N7
            ShimMuxOp shimMuxOp = analyzer.getShimMux(rewriter, srcSB.col);
            addConnection(rewriter,
                          cast<Interconnect>(shimMuxOp.getOperation()), flowOp,
                          srcBundle, srcChannel, WireBundle::North, shimCh);
          } else if (srcBundle ==
                     WireBundle::NOC) {  // must be NOC0/NOC1 -> N2/N3 or
                                         // NOC2/NOC3 -> N6/N7
            shimCh = srcChannel >= 2 ? srcChannel + 4 : srcChannel + 2;
            ShimMuxOp shimMuxOp = analyzer.getShimMux(rewriter, srcSB.col);
            addConnection(rewriter,
                          cast<Interconnect>(shimMuxOp.getOperation()), flowOp,
                          srcBundle, srcChannel, WireBundle::North, shimCh);
          } else if (srcBundle ==
                     WireBundle::PLIO) {  // PLIO at start of flows with mux
            if (srcChannel == 2 || srcChannel == 3 || srcChannel == 6 ||
                srcChannel == 7) {  // Only some PLIO requrie mux
              ShimMuxOp shimMuxOp = analyzer.getShimMux(rewriter, srcSB.col);
              addConnection(
                  rewriter, cast<Interconnect>(shimMuxOp.getOperation()),
                  flowOp, srcBundle, srcChannel, WireBundle::North, shimCh);
            }
          }
        }
        for (const auto &[bundle, channel] : setting.dsts) {
          // handle special shim connectivity
          if (curr == srcSB && analyzer.getTile(rewriter, srcSB.col, srcSB.row)
                                   .isShimNOCorPLTile()) {
            addConnection(rewriter, cast<Interconnect>(swOp.getOperation()),
                          flowOp, WireBundle::South, shimCh, bundle, channel);
          } else if (analyzer.getTile(rewriter, curr.col, curr.row)
                         .isShimNOCorPLTile() &&
                     (bundle == WireBundle::DMA || bundle == WireBundle::PLIO ||
                      bundle == WireBundle::NOC)) {
            shimCh = channel;
            if (analyzer.getTile(rewriter, curr.col, curr.row)
                    .isShimNOCTile()) {
              // shim DMAs at end of flows
              if (bundle == WireBundle::DMA) {
                shimCh = channel == 0
                             ? 2
                             : 3;  // must be either N2 -> DMA0 or N3 -> DMA1
                ShimMuxOp shimMuxOp = analyzer.getShimMux(rewriter, curr.col);
                addConnection(
                    rewriter, cast<Interconnect>(shimMuxOp.getOperation()),
                    flowOp, WireBundle::North, shimCh, bundle, channel);
              } else if (bundle == WireBundle::NOC) {
                shimCh = channel + 2;  // must be either N2/3/4/5 -> NOC0/1/2/3
                ShimMuxOp shimMuxOp = analyzer.getShimMux(rewriter, curr.col);
                addConnection(
                    rewriter, cast<Interconnect>(shimMuxOp.getOperation()),
                    flowOp, WireBundle::North, shimCh, bundle, channel);
              } else if (channel >=
                         2) {  // must be PLIO...only PLIO >= 2 require mux
                ShimMuxOp shimMuxOp = analyzer.getShimMux(rewriter, curr.col);
                addConnection(
                    rewriter, cast<Interconnect>(shimMuxOp.getOperation()),
                    flowOp, WireBundle::North, shimCh, bundle, channel);
              }
            }
            addConnection(rewriter, cast<Interconnect>(swOp.getOperation()),
                          flowOp, setting.src.bundle, setting.src.channel,
                          WireBundle::South, shimCh);
          } else {
            // otherwise, regular switchbox connection
            addConnection(rewriter, cast<Interconnect>(swOp.getOperation()),
                          flowOp, setting.src.bundle, setting.src.channel,
                          bundle, channel);
          }
        }

        LLVM_DEBUG(llvm::dbgs() << curr << ": " << setting << " | "
                                << "\n");
      }

      LLVM_DEBUG(llvm::dbgs()
                 << "\n\t\tFinished adding ConnectOps to implement flowOp.\n");
      analyzer.processedFlows[srcPoint] = true;
    } else
      LLVM_DEBUG(llvm::dbgs() << "Flow already processed!\n");

    rewriter.eraseOp(Op);
  }
};

}  // namespace

namespace xilinx::AIE {

void AIEPathfinderPass::runOnOperation() {
  // create analysis pass with routing graph for entire device
  LLVM_DEBUG(llvm::dbgs() << "---Begin AIEPathfinderPass---\n");

  DeviceOp d = getOperation();
  if (failed(analyzer.runAnalysis(d))) return signalPassFailure();
  OpBuilder builder = OpBuilder::atBlockEnd(d.getBody());

  // Apply rewrite rule to switchboxes to add assignments to every 'connect'
  // operation inside
  ConversionTarget target(getContext());
  target.addLegalOp<TileOp>();
  target.addLegalOp<ConnectOp>();
  target.addLegalOp<SwitchboxOp>();
  target.addLegalOp<ShimMuxOp>();
  target.addLegalOp<EndOp>();

  RewritePatternSet patterns(&getContext());
  patterns.insert<ConvertFlowsToInterconnect>(d.getContext(), d, analyzer);
  if (failed(applyPartialConversion(d, target, std::move(patterns))))
    return signalPassFailure();

  // Populate wires between switchboxes and tiles.
  for (int col = 0; col <= analyzer.getMaxCol(); col++) {
    for (int row = 0; row <= analyzer.getMaxRow(); row++) {
      TileOp tile;
      if (analyzer.coordToTile.count({col, row}))
        tile = analyzer.coordToTile[{col, row}];
      else
        continue;
      SwitchboxOp sw;
      if (analyzer.coordToSwitchbox.count({col, row}))
        sw = analyzer.coordToSwitchbox[{col, row}];
      else
        continue;
      if (col > 0) {
        // connections east-west between stream switches
        if (analyzer.coordToSwitchbox.count({col - 1, row})) {
          auto westsw = analyzer.coordToSwitchbox[{col - 1, row}];
          builder.create<WireOp>(builder.getUnknownLoc(), westsw,
                                 WireBundle::East, sw, WireBundle::West);
        }
      }
      if (row > 0) {
        // connections between abstract 'core' of tile
        builder.create<WireOp>(builder.getUnknownLoc(), tile, WireBundle::Core,
                               sw, WireBundle::Core);
        // connections between abstract 'dma' of tile
        builder.create<WireOp>(builder.getUnknownLoc(), tile, WireBundle::DMA,
                               sw, WireBundle::DMA);
        // connections north-south inside array ( including connection to shim
        // row)
        if (analyzer.coordToSwitchbox.count({col, row - 1})) {
          auto southsw = analyzer.coordToSwitchbox[{col, row - 1}];
          builder.create<WireOp>(builder.getUnknownLoc(), southsw,
                                 WireBundle::North, sw, WireBundle::South);
        }
      } else if (row == 0) {
        if (tile.isShimNOCTile()) {
          if (analyzer.coordToShimMux.count({col, 0})) {
            auto shimsw = analyzer.coordToShimMux[{col, 0}];
            builder.create<WireOp>(
                builder.getUnknownLoc(), shimsw,
                WireBundle::North,  // Changed to connect into the north
                sw, WireBundle::South);
            // PLIO is attached to shim mux
            if (analyzer.coordToPLIO.count(col)) {
              auto plio = analyzer.coordToPLIO[col];
              builder.create<WireOp>(builder.getUnknownLoc(), plio,
                                     WireBundle::North, shimsw,
                                     WireBundle::South);
            }

            // abstract 'DMA' connection on tile is attached to shim mux ( in
            // row 0 )
            builder.create<WireOp>(builder.getUnknownLoc(), tile,
                                   WireBundle::DMA, shimsw, WireBundle::DMA);
          }
        } else if (tile.isShimPLTile()) {
          // PLIO is attached directly to switch
          if (analyzer.coordToPLIO.count(col)) {
            auto plio = analyzer.coordToPLIO[col];
            builder.create<WireOp>(builder.getUnknownLoc(), plio,
                                   WireBundle::North, sw, WireBundle::South);
          }
        }
      }
    }
  }

  // If the routing violates architecture-specific routing constraints, then
  // attempt to partially reroute.
  const auto &targetModel = d.getTargetModel();
  std::vector<ConnectOp> problemConnects;
  d.walk([&](ConnectOp connect) {
    if (auto sw = connect->getParentOfType<SwitchboxOp>()) {
      // Constraint: memtile stream switch constraints
      if (auto tile = sw.getTileOp();
          tile.isMemTile() &&
          !targetModel.isLegalMemtileConnection(
              connect.getSourceBundle(), connect.getSourceChannel(),
              connect.getDestBundle(), connect.getDestChannel())) {
        problemConnects.push_back(connect);
      }
    }
  });

  for (auto connect : problemConnects) {
    auto swBox = connect->getParentOfType<SwitchboxOp>();
    builder.setInsertionPoint(connect);
    auto northSw = getSwitchbox(d, swBox.colIndex(), swBox.rowIndex() + 1);
    if (auto southSw = getSwitchbox(d, swBox.colIndex(), swBox.rowIndex() - 1);
        !attemptFixupMemTileRouting(builder, northSw, southSw, connect))
      return signalPassFailure();
  }
}

bool AIEPathfinderPass::attemptFixupMemTileRouting(const OpBuilder &builder,
                                                   SwitchboxOp northSwOp,
                                                   SwitchboxOp southSwOp,
                                                   ConnectOp &problemConnect) {
  int problemNorthChannel;
  if (problemConnect.getSourceBundle() == WireBundle::North) {
    problemNorthChannel = problemConnect.getSourceChannel();
  } else if (problemConnect.getDestBundle() == WireBundle::North) {
    problemNorthChannel = problemConnect.getDestChannel();
  } else
    return false;  // Problem is not about n-s routing
  int problemSouthChannel;
  if (problemConnect.getSourceBundle() == WireBundle::South) {
    problemSouthChannel = problemConnect.getSourceChannel();
  } else if (problemConnect.getDestBundle() == WireBundle::South) {
    problemSouthChannel = problemConnect.getDestChannel();
  } else
    return false;  // Problem is not about n-s routing

  // Attempt to reroute northern neighbouring sw
  if (reconnectConnectOps(builder, northSwOp, problemConnect, true,
                          WireBundle::South, problemNorthChannel,
                          problemSouthChannel))
    return true;
  if (reconnectConnectOps(builder, northSwOp, problemConnect, false,
                          WireBundle::South, problemNorthChannel,
                          problemSouthChannel))
    return true;
  // Otherwise, attempt to reroute southern neighbouring sw
  if (reconnectConnectOps(builder, southSwOp, problemConnect, true,
                          WireBundle::North, problemSouthChannel,
                          problemNorthChannel))
    return true;
  if (reconnectConnectOps(builder, southSwOp, problemConnect, false,
                          WireBundle::North, problemSouthChannel,
                          problemNorthChannel))
    return true;
  return false;
}

bool AIEPathfinderPass::reconnectConnectOps(const OpBuilder &builder,
                                            SwitchboxOp sw,
                                            ConnectOp problemConnect,
                                            bool isIncomingToSW,
                                            WireBundle problemBundle,
                                            int problemChan, int emptyChan) {
  bool hasEmptyChannelSlot = true;
  bool foundCandidateForFixup = false;
  ConnectOp candidate;
  if (isIncomingToSW) {
    for (ConnectOp connect : sw.getOps<ConnectOp>()) {
      if (connect.getDestBundle() == problemBundle &&
          connect.getDestChannel() == problemChan) {
        candidate = connect;
        foundCandidateForFixup = true;
      }
      if (connect.getDestBundle() == problemBundle &&
          connect.getDestChannel() == emptyChan) {
        hasEmptyChannelSlot = false;
      }
    }
  } else {
    for (ConnectOp connect : sw.getOps<ConnectOp>()) {
      if (connect.getSourceBundle() == problemBundle &&
          connect.getSourceChannel() == problemChan) {
        candidate = connect;
        foundCandidateForFixup = true;
      }
      if (connect.getSourceBundle() == problemBundle &&
          connect.getSourceChannel() == emptyChan) {
        hasEmptyChannelSlot = false;
      }
    }
  }
  if (foundCandidateForFixup && hasEmptyChannelSlot) {
    WireBundle problemBundleOpposite = problemBundle == WireBundle::North
                                           ? WireBundle::South
                                           : WireBundle::North;
    // Found empty channel slot, perform reroute
    if (isIncomingToSW) {
      replaceConnectOpWithNewDest(builder, candidate, problemBundle, emptyChan);
      replaceConnectOpWithNewSource(builder, problemConnect,
                                    problemBundleOpposite, emptyChan);
    } else {
      replaceConnectOpWithNewSource(builder, candidate, problemBundle,
                                    emptyChan);
      replaceConnectOpWithNewDest(builder, problemConnect,
                                  problemBundleOpposite, emptyChan);
    }
    return true;
  }
  return false;
}

// Replace connect op
ConnectOp AIEPathfinderPass::replaceConnectOpWithNewDest(OpBuilder builder,
                                                         ConnectOp connect,
                                                         WireBundle newBundle,
                                                         int newChannel) {
  builder.setInsertionPoint(connect);
  auto newOp = builder.create<ConnectOp>(
      builder.getUnknownLoc(), connect.getSourceBundle(),
      connect.getSourceChannel(), newBundle, newChannel);
  connect.erase();
  return newOp;
}
ConnectOp AIEPathfinderPass::replaceConnectOpWithNewSource(OpBuilder builder,
                                                           ConnectOp connect,
                                                           WireBundle newBundle,
                                                           int newChannel) {
  builder.setInsertionPoint(connect);
  auto newOp = builder.create<ConnectOp>(builder.getUnknownLoc(), newBundle,
                                         newChannel, connect.getDestBundle(),
                                         connect.getDestChannel());
  connect.erase();
  return newOp;
}

SwitchboxOp AIEPathfinderPass::getSwitchbox(DeviceOp &d, int col, int row) {
  SwitchboxOp output = nullptr;
  d.walk([&](SwitchboxOp swBox) {
    if (swBox.colIndex() == col && swBox.rowIndex() == row) {
      output = swBox;
    }
  });
  return output;
}

std::unique_ptr<OperationPass<DeviceOp>> createAIEPathfinderPass() {
  return std::make_unique<AIEPathfinderPass>();
}

}  // namespace xilinx::AIE
//===- AIELocalizeLocks.cpp ---------------------------------------*- C++
//-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2019 Xilinx Inc.
//
//===-------------------------xilinx::AIE::impl::AIELocalizeLocksBase-------------------------===//

struct AIELocalizeLocksPass
    : xilinx::AIE::impl::AIELocalizeLocksBase<AIELocalizeLocksPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect>();
  }
  void runOnOperation() override {
    DeviceOp deviceOp = getOperation();

    for (auto coreOp : deviceOp.getOps<CoreOp>()) {
      // Collect the locks used in this core.
      const auto &targetModel = getTargetModel(coreOp);

      auto thisTile = dyn_cast<TileOp>(coreOp.getTile().getDefiningOp());
      int col = thisTile.colIndex();
      int row = thisTile.rowIndex();

      // Find the neighboring tiles
      SmallVector<TileOp, 4> accessibleTiles;
      for (auto tile : deviceOp.getOps<TileOp>())
        if (int dstRow = tile.rowIndex();
            targetModel.isLegalMemAffinity(col, row, tile.colIndex(), dstRow))
          accessibleTiles.push_back(tile);

      for (auto tile : accessibleTiles) {
        int dstCol = tile.colIndex();
        int dstRow = tile.rowIndex();
        int cardinalMemOffset = 0;

        const auto &targetModel = getTargetModel(tile);
        int numLocks = targetModel.getNumLocks(dstCol, dstRow);
        for (auto user : tile.getResult().getUsers())
          if (auto lock = dyn_cast<LockOp>(user)) {
            if (targetModel.isMemSouth(col, row, dstCol, dstRow))
              cardinalMemOffset = 0;
            else if (targetModel.isMemWest(col, row, dstCol, dstRow))
              cardinalMemOffset = numLocks;
            else if (targetModel.isMemNorth(col, row, dstCol, dstRow))
              cardinalMemOffset = 2 * numLocks;
            else if (targetModel.isMemEast(col, row, dstCol, dstRow))
              cardinalMemOffset = 3 * numLocks;
            else
              llvm_unreachable("Found illegal lock user!");

            int localLockIndex = cardinalMemOffset + lock.getLockIDValue();

            OpBuilder builder =
                OpBuilder::atBlockBegin(&coreOp.getBody().front());

            Value coreLockIDValue = builder.create<arith::ConstantIndexOp>(
                builder.getUnknownLoc(), localLockIndex);
            lock.getResult().replaceUsesWithIf(
                coreLockIDValue, [&](OpOperand &opOperand) {
                  return opOperand.getOwner()->getParentOp() == coreOp;
                });
          }
      }
    }
  }
};

std::unique_ptr<OperationPass<DeviceOp>> AIE::createAIELocalizeLocksPass() {
  return std::make_unique<AIELocalizeLocksPass>();
}  //===- AIEObjectFifoStatefulTransform.cpp ----------------------*- MLIR
   //-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
//
// Date: October 18th 2021
//
//===----------------------------------------------------------------------===//

#include <numeric>
#include <set>

#define LOOP_VAR_DEPENDENCY (-2)

//===----------------------------------------------------------------------===//
// Lock Analysis
//===----------------------------------------------------------------------===//
class LockAnalysis {
  DenseMap<std::pair<Value, int>, int> locksPerTile;

 public:
  LockAnalysis(DeviceOp &device) {
    // go over the locks created for each tile and update the index in
    // locksPerTile
    for (auto lockOp : device.getOps<LockOp>()) {
      auto tile = lockOp.getTile();
      auto lockID = lockOp.getLockIDValue();
      locksPerTile[{tile, lockID}] = 1;
    }
  }

  /// Given a tile, returns next usable lockID for that tile.
  int getLockID(TileOp &tileOp) {
    const auto &targetModel = getTargetModel(tileOp);
    for (unsigned i = 0;
         i < targetModel.getNumLocks(tileOp.getCol(), tileOp.getRow()); i++)
      if (int usageCnt = locksPerTile[{tileOp, i}]; usageCnt == 0) {
        locksPerTile[{tileOp, i}] = 1;
        return i;
      }
    return -1;
  }
};

//===----------------------------------------------------------------------===//
// TileDMA Channel Analysis
//===----------------------------------------------------------------------===//
class DMAChannelAnalysis {
  DenseMap<Value, int> masterChannelsPerTile;
  DenseMap<Value, int> slaveChannelsPerTile;

 public:
  DMAChannelAnalysis(DeviceOp &device) {
    // go over the channels used for each tile and update the master/slave
    // channel maps
    for (auto memOp : device.getOps<MemOp>()) {
      Region &r = memOp.getBody();
      for (auto &bl : r.getBlocks()) {
        for (auto op : bl.getOps<DMAStartOp>()) {
          if (op.isSend())
            getMasterDMAChannel(memOp.getTile());
          else
            getSlaveDMAChannel(memOp.getTile());
        }
      }
    }
  }

  /// Given an AIE tile, returns its next usable master channel.
  DMAChannel getMasterDMAChannel(Value tile) {
    if (masterChannelsPerTile.find(tile) == masterChannelsPerTile.end())
      masterChannelsPerTile[tile] = 0;
    else
      masterChannelsPerTile[tile]++;
    DMAChannel dmaChan = {DMAChannelDir::MM2S, masterChannelsPerTile[tile]};
    return dmaChan;
  }

  /// Given an AIE tile, returns its next usable slave channel.
  DMAChannel getSlaveDMAChannel(Value tile) {
    if (slaveChannelsPerTile.find(tile) == slaveChannelsPerTile.end())
      slaveChannelsPerTile[tile] = 0;
    else
      slaveChannelsPerTile[tile]++;
    DMAChannel dmaChan = {DMAChannelDir::S2MM, slaveChannelsPerTile[tile]};
    return dmaChan;
  }
};

//===----------------------------------------------------------------------===//
// Create objectFifos Pass
//===----------------------------------------------------------------------===//
struct AIEObjectFifoStatefulTransformPass
    : xilinx::AIE::impl::AIEObjectFifoStatefulTransformBase<
          AIEObjectFifoStatefulTransformPass> {
  DenseMap<ObjectFifoCreateOp, std::vector<BufferOp>>
      buffersPerFifo;  // maps each objFifo to its corresponding buffer
  DenseMap<ObjectFifoCreateOp, std::vector<ExternalBufferOp>>
      externalBuffersPerFifo;  // maps each objFifo to its corresponding
  // external buffers
  DenseMap<ObjectFifoCreateOp, std::vector<LockOp>>
      locksPerFifo;  // maps each objFifo to its corresponding locks
  std::vector<std::pair<ObjectFifoCreateOp, std::vector<ObjectFifoCreateOp>>>
      splitFifos;  // maps each objFifo between non-adjacent tiles to its
  // corresponding consumer objectFifos
  DenseMap<ObjectFifoLinkOp, ObjectFifoCreateOp>
      objFifoLinks;  // maps each ObjectFifoLinkOp to objFifo whose elements
  // have been created and should be used
  std::vector<ObjectFifoCreateOp>
      splitBecauseLink;  // objfifos which have been split because they are
  // part of a Link, not because they didn't have a shared memory module

  /// Function that returns true if two tiles in the AIE array share a memory
  /// module. share_direction is equal to:
  ///   * -1 if the shared memory module is that of the first input tile,
  ///   * 1 if it is that of the second input tile,
  ///   * 0 is no memory module is shared.
  bool isSharedMemory(TileOp a, TileOp b, int *share_direction) {
    const auto &targetModel = getTargetModel(a.getOperation());

    if ((a.isShimTile() && !b.isShimTile()) ||
        (!a.isShimTile() && b.isShimTile())) {
      *share_direction = 0;
      return false;
    }
    if ((targetModel.isMemTile(a.getCol(), a.getRow()) &&
         !targetModel.isMemTile(b.getCol(), b.getRow())) ||
        (!targetModel.isMemTile(a.getCol(), a.getRow()) &&
         targetModel.isMemTile(b.getCol(), b.getRow()))) {
      *share_direction = 0;
      return false;
    }
    bool rightShared = targetModel.isLegalMemAffinity(
        a.colIndex(), a.rowIndex(), b.colIndex(), b.rowIndex());

    bool leftShared = targetModel.isLegalMemAffinity(
        b.colIndex(), b.rowIndex(), a.colIndex(), a.rowIndex());

    if (leftShared)
      *share_direction = -1;
    else if (rightShared)
      *share_direction = 1;
    else
      *share_direction = 0;

    return leftShared || rightShared;
  }

  // Return true if the objectFifo created by createOp requires a DMA to be set
  // up. This is the case if the tiles are not adjacent (no shared memory), if
  // the objectFifo broadcasts to multiple tiles, if one of the consumers or
  // the producer wants to use the multi-dimensional address generation
  // features of the DMA, if the objectFifo is part of a LinkOp, or if the
  // via_DMA attribute of the objectFifo is set.
  bool requiresDMAs(ObjectFifoCreateOp createOp, int &share_direction) {
    bool hasSharedMemory = false;
    bool atLeastOneConsumerWantsTransform = false;
    bool isUsedInLinkOp = false;

    if (createOp.getVia_DMA()) return true;

    if (createOp.getConsumerTiles().size() == 1 &&
        createOp.getDimensionsToStream().empty()) {
      // Test for shared memory
      for (auto consumerTile : createOp.getConsumerTiles()) {
        if (auto consumerTileOp =
                dyn_cast<TileOp>(consumerTile.getDefiningOp())) {
          if (std::count(splitBecauseLink.begin(), splitBecauseLink.end(),
                         createOp))
            hasSharedMemory =
                isSharedMemory(createOp.getProducerTileOp(),
                               createOp.getProducerTileOp(), &share_direction);
          else
            hasSharedMemory = isSharedMemory(createOp.getProducerTileOp(),
                                             consumerTileOp, &share_direction);
        }
      }
    }

    // Only test for use of data layout transformations if we are in the shared
    // memory case; otherwise, we will return `true` in any case.
    if (hasSharedMemory) {
      // Even if just one of the consumers in the list of consumers wants to
      // perform a memory transform, we need to use DMAs.
      for (BDDimLayoutArrayAttr dims :
           createOp.getDimensionsFromStreamPerConsumer())
        if (!dims.empty()) {
          atLeastOneConsumerWantsTransform = true;
          break;
        }
    }

    // Only test for this objfifo belonging to a LinkOp if we are in the shared
    // memory case; otherwise, we will return `true` in any case.
    if (hasSharedMemory) {
      if (auto linkOp = getOptionalLinkOp(createOp)) {
        splitBecauseLink.push_back(createOp);
        isUsedInLinkOp = true;
      }
    }

    return !hasSharedMemory || atLeastOneConsumerWantsTransform ||
           isUsedInLinkOp;
  }

  /// Function to retrieve ObjectFifoLinkOp of ObjectFifoCreateOp,
  /// if it belongs to one.
  std::optional<ObjectFifoLinkOp> getOptionalLinkOp(ObjectFifoCreateOp op) {
    auto device = op->getParentOfType<DeviceOp>();
    for (ObjectFifoLinkOp linkOp : device.getOps<ObjectFifoLinkOp>()) {
      for (ObjectFifoCreateOp in : linkOp.getInputObjectFifos())
        if (in == op) return {linkOp};
      for (ObjectFifoCreateOp out : linkOp.getOutputObjectFifos())
        if (out == op) return {linkOp};
    }
    return {};
  }

  ObjectFifoCreateOp createObjectFifo(
      OpBuilder &builder, AIEObjectFifoType datatype, std::string name,
      Value prodTile, Value consTile, Attribute depth,
      BDDimLayoutArrayAttr dimensionsToStream,
      BDDimLayoutArrayArrayAttr dimensionsFromStreamPerConsumer) {
    auto ofName = builder.getStringAttr(name);
    auto fifo = builder.create<ObjectFifoCreateOp>(
        builder.getUnknownLoc(), ofName, prodTile, consTile, depth, datatype,
        dimensionsToStream, dimensionsFromStreamPerConsumer);
    return fifo;
  }

  /// Function used to create objectFifo locks based on target architecture.
  /// Called by createObjectFifoElements().
  std::vector<LockOp> createObjectFifoLocks(OpBuilder &builder,
                                            LockAnalysis &lockAnalysis,
                                            ObjectFifoCreateOp op, int numElem,
                                            TileOp creation_tile) {
    std::vector<LockOp> locks;
    auto dev = op->getParentOfType<DeviceOp>();
    auto &target = dev.getTargetModel();
    if (creation_tile.isShimTile()) numElem = externalBuffersPerFifo[op].size();
    if (target.getTargetArch() == AIEArch::AIE1) {
      int of_elem_index =
          0;  // used to give objectFifo elements a symbolic name
      for (int i = 0; i < numElem; i++) {
        // create corresponding aie1 locks
        int lockID = lockAnalysis.getLockID(creation_tile);
        assert(lockID >= 0 && "No more locks to allocate!");
        auto lock = builder.create<LockOp>(builder.getUnknownLoc(),
                                           creation_tile, lockID, 0);
        lock.getOperation()->setAttr(
            SymbolTable::getSymbolAttrName(),
            builder.getStringAttr(op.name().str() + "_lock_" +
                                  std::to_string(of_elem_index)));
        locks.push_back(lock);
        of_elem_index++;
      }
    } else {
      // create corresponding aie2 locks
      int prodLockID = lockAnalysis.getLockID(creation_tile);
      assert(prodLockID >= 0 && "No more locks to allocate!");
      auto prodLock = builder.create<LockOp>(
          builder.getUnknownLoc(), creation_tile, prodLockID, numElem);
      prodLock.getOperation()->setAttr(
          SymbolTable::getSymbolAttrName(),
          builder.getStringAttr(op.name().str() + "_prod_lock"));
      locks.push_back(prodLock);

      int consLockID = lockAnalysis.getLockID(creation_tile);
      assert(consLockID >= 0 && "No more locks to allocate!");
      auto consLock = builder.create<LockOp>(builder.getUnknownLoc(),
                                             creation_tile, consLockID, 0);
      consLock.getOperation()->setAttr(
          SymbolTable::getSymbolAttrName(),
          builder.getStringAttr(op.name().str() + "_cons_lock"));
      locks.push_back(consLock);
    }
    return locks;
  }

  /// Function used to create objectFifo elements and their locks.
  /// It maps the input objectFifo to associated buffers and locks.
  void createObjectFifoElements(OpBuilder &builder, LockAnalysis &lockAnalysis,
                                ObjectFifoCreateOp op, int share_direction) {
    if (!op.size()) return;

    std::vector<BufferOp> buffers;
    auto fifo = llvm::cast<AIEObjectFifoType>(op.getElemType());
    auto elemType = llvm::cast<MemRefType>(fifo.getElementType());
    int numElem = op.size();
    int of_elem_index = 0;  // used to give objectFifo elements a symbolic name

    // if this objectFifo is linked to another, check if the other's elements
    // have already been created (the elements that are created are those of
    // the objFifo with elements of bigger size)
    bool linked = false;
    auto linkOp = getOptionalLinkOp(op);
    if (linkOp) {
      auto fifoIn = linkOp->getInputObjectFifos()[0];
      auto fifoOut = linkOp->getOutputObjectFifos()[0];
      linked = true;
      if (objFifoLinks.find(*linkOp) != objFifoLinks.end())
        return;  // elements have already been created
      if (linkOp->isJoin()) {
        // if join, fifoOut has bigger size
        if (op.name() != fifoOut.name()) return;
      } else if (linkOp->isDistribute()) {
        // if distribute, fifoIn has bigger size
        if (op.name() != fifoIn.name()) return;
      } else {
        auto fifoInType = llvm::cast<AIEObjectFifoType>(
            linkOp->getInputObjectFifos()[0].getElemType());
        auto elemInType = llvm::cast<MemRefType>(fifoInType.getElementType());
        int inSize = elemInType.getNumElements();

        auto fifoOutType = llvm::cast<AIEObjectFifoType>(
            linkOp->getOutputObjectFifos()[0].getElemType());
        auto elemOutType = llvm::cast<MemRefType>(fifoOutType.getElementType());

        if (int outSize = elemOutType.getNumElements(); inSize >= outSize) {
          if (op.name() != fifoIn.name()) return;
        } else {
          if (linkOp->getOutputObjectFifos()[0] != op) return;
        }
      }
    }

    TileOp creation_tile;
    if (share_direction == 0 || share_direction == -1)
      creation_tile = op.getProducerTileOp();
    else {
      auto consumerTileOp =
          dyn_cast<TileOp>(op.getConsumerTiles()[0].getDefiningOp());
      creation_tile = consumerTileOp;
    }

    // Reset opbuilder location to after the last tile declaration
    Operation *t = nullptr;
    auto dev = op->getParentOfType<DeviceOp>();
    for (auto tile_op : dev.getBody()->getOps<TileOp>()) {
      t = tile_op.getOperation();
    }
    builder.setInsertionPointAfter(t);
    for (int i = 0; i < numElem; i++) {
      // if shimTile external buffers are collected from input code
      // create as many locks as there are external buffers
      if (!creation_tile.isShimTile()) {
        auto buff = builder.create<BufferOp>(
            builder.getUnknownLoc(), elemType, creation_tile,
            builder.getStringAttr(op.name().str() + "_buff_" +
                                  std::to_string(of_elem_index)),
            /*address*/ nullptr, /*initial_value*/ nullptr,
            /*mem_bank*/ nullptr);
        buffers.push_back(buff);
      }
      of_elem_index++;
    }
    if (linked) {
      if (linkOp->isDistribute())
        numElem *= linkOp->getFifoOuts().size();
      else if (linkOp->isJoin())
        numElem *= linkOp->getFifoIns().size();
      objFifoLinks[*linkOp] = op;
    }
    std::vector<LockOp> locks = createObjectFifoLocks(builder, lockAnalysis, op,
                                                      numElem, creation_tile);
    buffersPerFifo[op] = buffers;
    locksPerFifo[op] = locks;
  }

  /// Function that returns a pointer to the block of a Region
  /// that contains the AIEEndOp.
  Block *findEndOpBlock(Region &r) {
    Block *endBlock = nullptr;
    for (auto &bl : r.getBlocks())
      if (!bl.getOps<EndOp>().empty()) endBlock = &bl;
    return endBlock;
  }

  /// Function used to create a Bd block.
  template <typename MyOp>
  void createBd(OpBuilder &builder, LockOp acqLock, int acqMode,
                LockAction acqLockAction, LockOp relLock, int relMode,
                MyOp buff, int offset, int len, Block *succ,
                BDDimLayoutArrayAttr dims) {
    builder.create<UseLockOp>(builder.getUnknownLoc(), acqLock, acqLockAction,
                              acqMode);
    if (!dims.getValue().empty())
      builder.create<DMABDOp>(builder.getUnknownLoc(), buff, offset, len, dims);
    else
      builder.create<DMABDOp>(builder.getUnknownLoc(), buff, offset, len);

    builder.create<UseLockOp>(builder.getUnknownLoc(), relLock,
                              LockAction::Release, relMode);
    builder.create<NextBDOp>(builder.getUnknownLoc(), succ);
  }

  /// Function used to create a Bd block.
  /// If lockMode is 0 we create a consumerDMA (i.e. on producer tile) else a
  /// producerDMA (i.e. on consumer tile).
  template <typename MyOp>
  void createBdBlock(OpBuilder &builder, ObjectFifoCreateOp op, int lockMode,
                     int acqNum, int relNum, MyOp buff, int offset, int len,
                     DMAChannelDir channelDir, size_t blockIndex, Block *succ,
                     BDDimLayoutArrayAttr dims) {
    LockOp acqLock;
    LockOp relLock;
    int acqMode = 1;
    int relMode = 1;
    auto acqLockAction = LockAction::Acquire;
    auto dev = op->getParentOfType<DeviceOp>();
    if (auto &target = dev.getTargetModel();
        target.getTargetArch() == AIEArch::AIE1) {
      acqMode = lockMode == 0 ? 1 : 0;
      relMode = lockMode == 0 ? 0 : 1;
      acqLock = locksPerFifo[op][blockIndex];
      relLock = locksPerFifo[op][blockIndex];
    } else {
      acqMode = acqNum;
      relMode = relNum;
      acqLockAction = LockAction::AcquireGreaterEqual;
      acqLock = channelDir == DMAChannelDir::S2MM ? locksPerFifo[op][0]
                                                  : locksPerFifo[op][1];
      relLock = channelDir == DMAChannelDir::S2MM ? locksPerFifo[op][1]
                                                  : locksPerFifo[op][0];
    }
    createBd(builder, acqLock, acqMode, acqLockAction, relLock, relMode, buff,
             offset, len, succ, dims);
  }

  /// Function that either calls createAIETileDMA(), createShimDMA() or
  /// createMemTileDMA() based on op tile row value.
  void createDMA(DeviceOp &device, OpBuilder &builder, ObjectFifoCreateOp op,
                 DMAChannelDir channelDir, int channelIndex, int lockMode,
                 BDDimLayoutArrayAttr dims) {
    if (op.getProducerTileOp().isShimTile()) {
      createShimDMA(device, builder, op, channelDir, channelIndex, lockMode,
                    dims);
    } else if (op.getProducerTileOp().isMemTile()) {
      createMemTileDMA(device, builder, op, channelDir, channelIndex, lockMode,
                       dims);
    } else {
      createAIETileDMA(device, builder, op, channelDir, channelIndex, lockMode,
                       dims);
    }
  }

  /// Function used to create a MemOp region with a DMA channel.
  /// It uses creatBdBlock(), see there for lockMode input.
  void createAIETileDMA(DeviceOp &device, OpBuilder &builder,
                        ObjectFifoCreateOp op, DMAChannelDir channelDir,
                        int channelIndex, int lockMode,
                        BDDimLayoutArrayAttr dims) {
    size_t numBlocks = op.size();
    if (numBlocks == 0) return;

    int acqNum = 1;
    int relNum = 1;

    auto fifo = llvm::cast<AIEObjectFifoType>(op.getElemType());
    auto elemType = llvm::cast<MemRefType>(fifo.getElementType());
    int len = elemType.getNumElements();

    // search for the buffers/locks (based on if this objFifo has a link)
    ObjectFifoCreateOp target = op;
    if (std::optional<ObjectFifoLinkOp> linkOp = getOptionalLinkOp(op);
        linkOp.has_value())
      if (objFifoLinks.find(linkOp.value()) != objFifoLinks.end())
        target = objFifoLinks[linkOp.value()];

    // search for MemOp
    Operation *producerMem = nullptr;
    for (auto memOp : device.getOps<MemOp>()) {
      if (memOp.getTile() == op.getProducerTile()) {
        producerMem = memOp.getOperation();
        break;
      }
    }

    // if none exists, create one
    TileOp objFifoTileOp = target.getProducerTileOp();
    if (producerMem == nullptr) {
      if (device->getNumRegions() != 1)
        assert(false && "expected num regions for device op");
      OpBuilder::InsertionGuard g(builder);
      builder.setInsertionPointToEnd(device.getBody());
      auto newMemOp =
          builder.create<MemOp>(builder.getUnknownLoc(), objFifoTileOp);
      {
        OpBuilder::InsertionGuard g(builder);
        builder.setInsertionPointToStart(&newMemOp.getRegion().emplaceBlock());
        builder.create<EndOp>(builder.getUnknownLoc());
      }
      producerMem = newMemOp.getOperation();
    }
    Block *endBlock = findEndOpBlock(producerMem->getRegion(0));
    Block *lastDmaBlock = endBlock->getSinglePredecessor();
    Block *dmaBlock = builder.createBlock(endBlock);
    Block *bdBlock = builder.createBlock(endBlock);

    // create DMA channel
    builder.setInsertionPointToStart(dmaBlock);
    builder.create<DMAStartOp>(builder.getUnknownLoc(), channelDir,
                               channelIndex, /*repeatCount*/ 0, bdBlock,
                               endBlock);
    if (lastDmaBlock != nullptr)
      lastDmaBlock->getTerminator()->setSuccessor(dmaBlock, 1);

    // create Bd blocks
    Block *succ;
    Block *curr = bdBlock;
    size_t blockIndex = 0;
    for (size_t i = 0; i < numBlocks; i++) {
      if (blockIndex >= buffersPerFifo[target].size()) break;
      if (i == numBlocks - 1)
        succ = bdBlock;
      else
        succ = builder.createBlock(endBlock);

      builder.setInsertionPointToStart(curr);
      createBdBlock<BufferOp>(builder, target, lockMode, acqNum, relNum,
                              buffersPerFifo[target][blockIndex], /*offset*/ 0,
                              len, channelDir, blockIndex, succ, dims);
      curr = succ;
      blockIndex++;
    }
  }

  /// Function used to create a ShimDMAOp region with a DMA channel.
  /// It uses creatBdBlock(), see there for lockMode input.
  void createShimDMA(DeviceOp &device, OpBuilder &builder,
                     ObjectFifoCreateOp op, DMAChannelDir channelDir,
                     int channelIndex, int lockMode,
                     BDDimLayoutArrayAttr dims) {
    size_t numBlocks = externalBuffersPerFifo[op].size();
    if (numBlocks == 0) return;

    int acqNum = 1;
    int relNum = 1;

    // search for ShimDMAOp
    Operation *producerDMA = nullptr;
    for (auto dmaOp : device.getOps<ShimDMAOp>()) {
      if (dmaOp.getTile() == op.getProducerTile()) {
        producerDMA = dmaOp.getOperation();
        break;
      }
    }

    // if none exists, create one
    TileOp objFifoTileOp = op.getProducerTileOp();
    if (producerDMA == nullptr) {
      if (device->getNumRegions() != 1)
        assert(false && "expected num regions for device op");
      OpBuilder::InsertionGuard g(builder);
      builder.setInsertionPointToEnd(device.getBody());
      auto newDMAOp = builder.create<ShimDMAOp>(
          builder.getUnknownLoc(), builder.getIndexType(), objFifoTileOp);
      {
        OpBuilder::InsertionGuard g(builder);
        builder.setInsertionPointToStart(&newDMAOp.getRegion().emplaceBlock());
        builder.create<EndOp>(builder.getUnknownLoc());
      }
      producerDMA = newDMAOp.getOperation();
    }

    Block *endBlock = findEndOpBlock(producerDMA->getRegion(0));
    Block *lastDmaBlock = endBlock->getSinglePredecessor();
    Block *dmaBlock = builder.createBlock(endBlock);
    Block *bdBlock = builder.createBlock(endBlock);

    // create DMA channel
    builder.setInsertionPointToStart(dmaBlock);
    builder.create<DMAStartOp>(builder.getUnknownLoc(), channelDir,
                               channelIndex, /*repeatCout*/ 0, bdBlock,
                               endBlock);
    if (lastDmaBlock != nullptr)
      lastDmaBlock->getTerminator()->setSuccessor(dmaBlock, 1);

    // create Bd blocks
    Block *succ;
    Block *curr = bdBlock;
    size_t blockIndex = 0;
    for (size_t i = 0; i < numBlocks; i++) {
      if (blockIndex >= externalBuffersPerFifo[op].size()) break;
      if (i == numBlocks - 1)
        succ = bdBlock;
      else
        succ = builder.createBlock(endBlock);

      MemRefType buffer = externalBuffersPerFifo[op][blockIndex].getType();
      int len = buffer.getNumElements();
      builder.setInsertionPointToStart(curr);
      createBdBlock<ExternalBufferOp>(builder, op, lockMode, acqNum, relNum,
                                      externalBuffersPerFifo[op][blockIndex],
                                      /*offset*/ 0, len, channelDir, blockIndex,
                                      succ, dims);
      curr = succ;
      blockIndex++;
    }
  }

  /// Function used to create a MemTileDMAOp region with a DMA channel.
  /// It uses creatBdBlock(), see there for lockMode input.
  void createMemTileDMA(DeviceOp &device, OpBuilder &builder,
                        ObjectFifoCreateOp op, DMAChannelDir channelDir,
                        int channelIndex, int lockMode,
                        BDDimLayoutArrayAttr dims) {
    size_t numBlocks = op.size();
    if (numBlocks == 0) return;

    auto fifo = llvm::cast<AIEObjectFifoType>(op.getElemType());
    auto elemType = llvm::cast<MemRefType>(fifo.getElementType());
    int lenOut = elemType.getNumElements();
    int acqNum = 1;
    int relNum = 1;

    // search for the buffers/locks (based on if this objFifo has a link)
    // identify size difference between input and output memrefs
    ObjectFifoCreateOp target = op;
    bool isDistribute = false;
    bool isJoin = false;
    int extraOffset = 0;
    if (auto linkOp = getOptionalLinkOp(op)) {
      if (objFifoLinks.find(*linkOp) != objFifoLinks.end()) {
        target = objFifoLinks[*linkOp];

        if (linkOp->isJoin()) {
          // find offset based on order of this op in join list
          isJoin = true;
          if (target == op) {
            acqNum = linkOp->getFifoIns().size();
            relNum = linkOp->getFifoIns().size();
          } else {
            for (auto fifoIn : linkOp->getInputObjectFifos()) {
              auto fifoType =
                  llvm::cast<AIEObjectFifoType>(fifoIn.getElemType());
              auto elemType = llvm::cast<MemRefType>(fifoType.getElementType());
              if (fifoIn.name() == op.name()) break;
              extraOffset += elemType.getNumElements();
            }
          }
        } else if (linkOp->isDistribute()) {
          // find offset based on order of this op in distribute list
          isDistribute = true;
          if (target == op) {
            acqNum = linkOp->getFifoOuts().size();
            relNum = linkOp->getFifoOuts().size();
          } else {
            for (auto fifoOut : linkOp->getOutputObjectFifos()) {
              auto fifoType =
                  llvm::cast<AIEObjectFifoType>(fifoOut.getElemType());
              auto elemType = llvm::cast<MemRefType>(fifoType.getElementType());
              if (fifoOut.name() == op.name()) break;
              extraOffset += elemType.getNumElements();
            }
          }
        } else {
          if (target != op) {
            auto targetFifo =
                llvm::cast<AIEObjectFifoType>(target.getElemType());
            auto targetElemType =
                llvm::cast<MemRefType>(targetFifo.getElementType());
            lenOut = targetElemType.getNumElements();
          }
        }

        // check if current op is of smaller size in link
        if (target != op) numBlocks = target.size();
      }
    }

    // search for MemTileDMAOp
    Operation *producerDMA = nullptr;
    for (auto dmaOp : device.getOps<MemTileDMAOp>()) {
      if (dmaOp.getTile() == target.getProducerTile()) {
        producerDMA = dmaOp.getOperation();
        break;
      }
    }

    // if none exists, create one
    TileOp objFifoTileOp = target.getProducerTileOp();
    if (producerDMA == nullptr) {
      if (device->getNumRegions() != 1)
        assert(false && "expected num regions for device op");
      OpBuilder::InsertionGuard g(builder);
      builder.setInsertionPointToEnd(device.getBody());
      auto newDMAOp =
          builder.create<MemTileDMAOp>(builder.getUnknownLoc(), objFifoTileOp);
      {
        OpBuilder::InsertionGuard g(builder);
        builder.setInsertionPointToStart(&newDMAOp.getRegion().emplaceBlock());
        builder.create<EndOp>(builder.getUnknownLoc());
      }
      producerDMA = newDMAOp.getOperation();
    }

    Block *endBlock = findEndOpBlock(producerDMA->getRegion(0));
    Block *lastDmaBlock = endBlock->getSinglePredecessor();
    Block *dmaBlock = builder.createBlock(endBlock);
    Block *bdBlock = builder.createBlock(endBlock);

    // create DMA channel
    builder.setInsertionPointToStart(dmaBlock);
    builder.create<DMAStartOp>(builder.getUnknownLoc(), channelDir,
                               channelIndex, /*repeatCount*/ 0, bdBlock,
                               endBlock);
    if (lastDmaBlock != nullptr)
      lastDmaBlock->getTerminator()->setSuccessor(dmaBlock, 1);

    // create Bd blocks
    Block *succ;
    Block *curr = bdBlock;
    size_t blockIndex = 0;
    for (size_t i = 0; i < numBlocks; i++) {
      if (blockIndex >= buffersPerFifo[target].size()) break;
      if (i == numBlocks - 1)
        succ = bdBlock;
      else
        succ = builder.createBlock(endBlock);

      builder.setInsertionPointToStart(curr);
      int offset = 0;
      if (isDistribute || isJoin) offset = extraOffset;
      createBdBlock<BufferOp>(builder, target, lockMode, acqNum, relNum,
                              buffersPerFifo[target][blockIndex], offset,
                              lenOut, channelDir, blockIndex, succ, dims);
      curr = succ;
      blockIndex++;
    }
  }

  // Function that computes the Least Common Multiplier of the values
  // of a vector.
  int computeLCM(std::set<int> values) {
    int lcm = 1;
    for (int i : values) lcm = i * lcm / std::gcd(i, lcm);
    return lcm;
  }

  // Function that unrolls for-loops that contain objectFifo operations.
  LogicalResult unrollForLoops(DeviceOp &device, OpBuilder &builder,
                               std::set<TileOp> objectFifoTiles) {
    for (auto coreOp : device.getOps<CoreOp>()) {
      if (objectFifoTiles.count(coreOp.getTileOp()) > 0) {
        WalkResult res = coreOp.walk([&](scf::ForOp forLoop) {
          // look for operations on objectFifos
          // when multiple fifos in same loop, must use the smallest
          // common multiplier as the unroll factor
          bool found = false;
          std::set<int> objFifoSizes;
          Block *body = forLoop.getBody();

          for (auto acqOp : body->getOps<ObjectFifoAcquireOp>()) {
            if (acqOp.getOperation()->getParentOp() == forLoop) {
              found = true;
              ObjectFifoCreateOp op = acqOp.getObjectFifo();
              objFifoSizes.insert(op.size());
            }
          }

          int unrollFactor =
              computeLCM(objFifoSizes);  // also counts original loop body

          if (found) {
            if (failed(mlir::loopUnrollByFactor(forLoop, unrollFactor))) {
              forLoop.emitOpError()
                  << "could not be unrolled with unrollFactor: " << unrollFactor
                  << "\n";
              return WalkResult::interrupt();
            }
          }
          return WalkResult::advance();
        });
        if (res.wasInterrupted()) return failure();
      }
    }
    return success();
  }

  /// Function used to create a UseLockOp based on input parameters.
  /// acc is an accumulator map that tracks the indices of the next locks to
  /// acquire (or release). Uses op to find index of acc for next lockID.
  /// Updates acc.
  void createUseLocks(OpBuilder &builder, ObjectFifoCreateOp op,
                      ObjectFifoPort port,
                      DenseMap<std::pair<ObjectFifoCreateOp, int>, int> &acc,
                      int numLocks, LockAction lockAction) {
    ObjectFifoCreateOp target = op;
    auto portNum = port == ObjectFifoPort::Produce ? 0 : 1;
    if (auto linkOp = getOptionalLinkOp(op))
      if (objFifoLinks.find(*linkOp) != objFifoLinks.end())
        target = objFifoLinks[*linkOp];

    auto dev = op->getParentOfType<DeviceOp>();
    if (auto &targetArch = dev.getTargetModel();
        targetArch.getTargetArch() == AIEArch::AIE1) {
      int lockMode = 0;
      if ((port == ObjectFifoPort::Produce &&
           lockAction == LockAction::Release) ||
          (port == ObjectFifoPort::Consume &&
           lockAction == LockAction::Acquire))
        lockMode = 1;
      for (int i = 0; i < numLocks; i++) {
        int lockID = acc[{op, portNum}];
        builder.create<UseLockOp>(builder.getUnknownLoc(),
                                  locksPerFifo[target][lockID], lockAction,
                                  lockMode);
        acc[{op, portNum}] =
            (lockID + 1) % op.size();  // update to next objFifo elem
      }
    } else {
      if (numLocks == 0) return;
      // search for the correct lock based on the port of the acq/rel
      // operation e.g. acq as consumer is the read lock (second)
      LockOp lock;
      if (lockAction == LockAction::AcquireGreaterEqual) {
        if (port == ObjectFifoPort::Produce)
          lock = locksPerFifo[target][0];
        else
          lock = locksPerFifo[target][1];
      } else {
        if (port == ObjectFifoPort::Produce)
          lock = locksPerFifo[target][1];
        else
          lock = locksPerFifo[target][0];
      }
      builder.create<UseLockOp>(builder.getUnknownLoc(), lock, lockAction,
                                numLocks);
      acc[{op, portNum}] = (acc[{op, portNum}] + numLocks) %
                           op.size();  // update to next objFifo elem
    }
  }

  /// Function used to check whether op is already contained in map.
  /// If it is then return the associated int, if not create new entry and
  /// return 0.
  int updateAndReturnIndex(
      DenseMap<std::pair<ObjectFifoCreateOp, int>, int> &map,
      std::pair<ObjectFifoCreateOp, int> pair) {
    if (map.find(pair) == map.end()) {
      map[pair] = 0;
      return 0;
    }
    return map[pair];
  }

  /// Function used to add an external buffer to the externalBuffersPerFifo map.
  void addExternalBuffer(ObjectFifoCreateOp fifo, ExternalBufferOp buff) {
    if (externalBuffersPerFifo.find(fifo) == externalBuffersPerFifo.end()) {
      std::vector<ExternalBufferOp> buffs;
      externalBuffersPerFifo[fifo] = buffs;
    }
    externalBuffersPerFifo[fifo].push_back(buff);
  }

  /// Function used to detect all external buffers associated with parent
  /// objectFifo and tile then map them to child objectFifo.
  void detectExternalBuffers(DeviceOp &device, ObjectFifoCreateOp parent,
                             ObjectFifoCreateOp child, Value tile) {
    for (auto regOp : device.getOps<ObjectFifoRegisterExternalBuffersOp>())
      if (auto objFifo = regOp.getObjectFifo();
          regOp.getTile() == tile && objFifo == parent)
        for (auto extBuff : regOp.getExternalBuffers())
          addExternalBuffer(child, extBuff.getDefiningOp<ExternalBufferOp>());
  }

  /// Function used to replace uses of split objectFifos.
  void replaceSplitFifo(ObjectFifoCreateOp originalOp, ObjectFifoCreateOp newOp,
                        TileOp tile) {
    auto original =
        originalOp->getAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName());
    auto newSymbol =
        newOp->getAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName());
    for (auto user : tile->getUsers())
      if (isa<CoreOp>(user))
        if (auto res =
                SymbolTable::replaceAllSymbolUses(original, newSymbol, user);
            res.failed())
          llvm_unreachable("unreachable");
  }

  /// Function used to find the size of an objectFifo after split based on
  /// the maximum number of elements (of the original objectFifo) acquired
  /// by a process running on given tile. If no CoreOp exists for this tile
  /// return 0.
  int findObjectFifoSize(DeviceOp &device, Value tile,
                         ObjectFifoCreateOp objFifo) {
    if (objFifo.size() == 0) return 0;

    // if memTile, size is equal to objFifo size
    if (tile.getDefiningOp<TileOp>().isMemTile()) return objFifo.size();

    // if shimTile, size is equal to number of external buffers
    if (tile.getDefiningOp<TileOp>().isShimTile())
      for (auto regOp : device.getOps<ObjectFifoRegisterExternalBuffersOp>()) {
        if (regOp.getTile() == tile) return regOp.getExternalBuffers().size();
      }

    int maxAcquire = 0;
    for (auto coreOp : device.getOps<CoreOp>())
      if (coreOp.getTile() == tile)
        coreOp.walk([&](ObjectFifoAcquireOp acqOp) {
          if (auto createOp = acqOp.getObjectFifo(); createOp == objFifo)
            if (acqOp.acqNumber() > maxAcquire) maxAcquire = acqOp.acqNumber();
        });

    if (maxAcquire > 0) {
      if (maxAcquire == 1 && objFifo.size() == 1) return 1;
      return maxAcquire + 1;
      // +1 because objectFifo size is always 1 bigger than maxAcquire to allow
      // for prefetching: simplest case scenario is at least a ping-pong buffer
    }

    return objFifo.size();
  }

  /// Function used to generate, from an objectFifo with a shimTile endpoint, a
  /// shimDMAAllocationOp containing the channelDir, channelIndex and
  /// shimTile col assigned by the objectFifo lowering.
  void createObjectFifoAllocationInfo(OpBuilder &builder, MLIRContext *ctx,
                                      FlatSymbolRefAttr obj_fifo, int colIndex,
                                      DMAChannelDir channelDir,
                                      int channelIndex) {
    builder.create<ShimDMAAllocationOp>(builder.getUnknownLoc(), obj_fifo,
                                        DMAChannelDirAttr::get(ctx, channelDir),
                                        builder.getI64IntegerAttr(channelIndex),
                                        builder.getI64IntegerAttr(colIndex));
  }

  void runOnOperation() override {
    DeviceOp device = getOperation();
    LockAnalysis lockAnalysis(device);
    DMAChannelAnalysis dmaAnalysis(device);
    OpBuilder builder = OpBuilder::atBlockEnd(device.getBody());
    auto ctx = device->getContext();
    std::set<TileOp>
        objectFifoTiles;  // track cores to check for loops during unrolling

    //===------------------------------------------------------------------===//
    // Split objectFifos into a consumer end and producer end if needed
    //===------------------------------------------------------------------===//
    // We are going to create additional createObjectFifoOps, so get a copy of
    // all "original" ones before the loop to avoid looping over newly created
    // ones.
    std::vector<ObjectFifoCreateOp> createFifoOps;
    auto range = device.getOps<ObjectFifoCreateOp>();
    createFifoOps.insert(createFifoOps.end(), range.begin(), range.end());
    for (auto createOp : createFifoOps) {
      std::vector<ObjectFifoCreateOp> splitConsumerFifos;
      int consumerIndex = 0;
      int consumerDepth = createOp.size();
      ArrayRef<BDDimLayoutArrayAttr> consumerDims =
          createOp.getDimensionsFromStreamPerConsumer();

      // Only FIFOs using DMA are split into two ends;
      // skip in shared memory case
      if (int share_direction = 0; !requiresDMAs(createOp, share_direction))
        continue;

      for (auto consumerTile : createOp.getConsumerTiles()) {
        auto consumerTileOp = dyn_cast<TileOp>(consumerTile.getDefiningOp());

        if (isa<ArrayAttr>(createOp.getElemNumber())) {
          // +1 to account for 1st depth (producer)
          consumerDepth = createOp.size(consumerIndex + 1);
        } else {
          consumerDepth = findObjectFifoSize(device, consumerTileOp, createOp);
        }

        builder.setInsertionPointAfter(createOp);
        auto datatype = llvm::cast<AIEObjectFifoType>(createOp.getElemType());
        auto consumerObjFifoSize =
            builder.getIntegerAttr(builder.getI32Type(), consumerDepth);
        // rename and replace split objectFifo
        std::string consumerFifoName;
        if (createOp.getConsumerTiles().size() > 1) {
          consumerFifoName = createOp.name().str() + "_" +
                             std::to_string(consumerIndex) + "_cons";
        } else {
          consumerFifoName = createOp.name().str() + "_cons";
        }
        BDDimLayoutArrayAttr emptyDims =
            BDDimLayoutArrayAttr::get(builder.getContext(), {});
        BDDimLayoutArrayAttr singletonFromStreamDims =
            BDDimLayoutArrayAttr::get(
                builder.getContext(),
                ArrayRef<BDDimLayoutAttr>{consumerDims[consumerIndex]});
        BDDimLayoutArrayArrayAttr fromStreamDims =
            BDDimLayoutArrayArrayAttr::get(builder.getContext(),
                                           singletonFromStreamDims);

        ObjectFifoCreateOp consumerFifo = createObjectFifo(
            builder, datatype, consumerFifoName, consumerTile, consumerTile,
            consumerObjFifoSize, emptyDims, fromStreamDims);
        replaceSplitFifo(createOp, consumerFifo, consumerTileOp);

        // identify external buffers that were registered to the consumer fifo
        if (consumerTile.getDefiningOp<TileOp>().isShimTile())
          detectExternalBuffers(device, createOp, consumerFifo, consumerTile);

        // record that this objectFifo was split; it will require DMA config
        splitConsumerFifos.push_back(consumerFifo);

        // update the linkOp if the split objFifo was originally its start point
        if (auto linkOp = getOptionalLinkOp(createOp))
          for (ObjectFifoCreateOp fifoIn : linkOp->getInputObjectFifos())
            if (fifoIn.name() == createOp.name() &&
                consumerTile == *linkOp->getOptionalSharedTile())
              if (failed(SymbolTable::replaceAllSymbolUses(
                      createOp, consumerFifo.name(), linkOp->getOperation())))
                llvm::report_fatal_error("unable to update all symbol uses");

        consumerIndex++;
      }

      if (!splitConsumerFifos.empty()) {
        splitFifos.emplace_back(createOp, splitConsumerFifos);
      }
    }

    //===------------------------------------------------------------------===//
    // - Create objectFifo buffers and locks.
    // - Populate a list of tiles containing objectFifos for later processing of
    //   the acquires/releases (uses of the FIFO).
    //===------------------------------------------------------------------===//
    for (auto createOp : device.getOps<ObjectFifoCreateOp>()) {
      int share_direction = 0;
      bool shared = !requiresDMAs(createOp, share_direction);

      // add all tiles that contain an objectFifo to objectFifoTiles for later
      // loop unrolling pass
      objectFifoTiles.insert(createOp.getProducerTileOp());
      for (auto consumerTile : createOp.getConsumerTiles()) {
        auto consumerTileOp = dyn_cast<TileOp>(consumerTile.getDefiningOp());
        objectFifoTiles.insert(consumerTileOp);
      }

      // identify external buffers that were registered to
      // the producer objectFifo
      if (createOp.getProducerTileOp().isShimTile())
        detectExternalBuffers(device, createOp, createOp,
                              createOp.getProducerTile());

      // if split, the necessary size for producer fifo might change
      if (shared)
        createObjectFifoElements(builder, lockAnalysis, createOp,
                                 share_direction);
      else {
        if (isa<ArrayAttr>(createOp.getElemNumber()))
          createOp.setElemNumberAttr(
              builder.getI32IntegerAttr(createOp.size()));
        else {
          int prodMaxAcquire = findObjectFifoSize(
              device, createOp.getProducerTileOp(), createOp);
          createOp.setElemNumberAttr(builder.getI32IntegerAttr(prodMaxAcquire));
        }
        createObjectFifoElements(builder, lockAnalysis, createOp,
                                 share_direction);
      }
    }

    //===------------------------------------------------------------------===//
    // Create flows and tile DMAs
    //===------------------------------------------------------------------===//
    // Only the objectFifos we split above require DMA communication; the others
    // rely on shared memory and share the same buffers.
    for (auto &[producer, consumers] : splitFifos) {
      // create producer tile DMA
      DMAChannel producerChan =
          dmaAnalysis.getMasterDMAChannel(producer.getProducerTile());
      createDMA(device, builder, producer, producerChan.direction,
                producerChan.channel, 0, producer.getDimensionsToStreamAttr());
      // generate objectFifo allocation info
      builder.setInsertionPoint(&device.getBody()->back());
      if (producer.getProducerTileOp().isShimTile())
        createObjectFifoAllocationInfo(
            builder, ctx, SymbolRefAttr::get(ctx, producer.getName()),
            producer.getProducerTileOp().colIndex(), producerChan.direction,
            producerChan.channel);

      for (auto consumer : consumers) {
        // create consumer tile DMA
        DMAChannel consumerChan =
            dmaAnalysis.getSlaveDMAChannel(consumer.getProducerTile());
        BDDimLayoutArrayAttr consumerDims =
            consumer.getDimensionsFromStreamPerConsumer()[0];
        createDMA(device, builder, consumer, consumerChan.direction,
                  consumerChan.channel, 1, consumerDims);
        // generate objectFifo allocation info
        builder.setInsertionPoint(&device.getBody()->back());
        if (consumer.getProducerTileOp().isShimTile())
          createObjectFifoAllocationInfo(
              builder, ctx, SymbolRefAttr::get(ctx, producer.getName()),
              consumer.getProducerTileOp().colIndex(), consumerChan.direction,
              consumerChan.channel);

        // create flow
        builder.setInsertionPointAfter(producer);
        builder.create<FlowOp>(builder.getUnknownLoc(),
                               producer.getProducerTile(), WireBundle::DMA,
                               producerChan.channel, consumer.getProducerTile(),
                               WireBundle::DMA, consumerChan.channel);
      }
    }

    //===------------------------------------------------------------------===//
    // Unroll for loops
    //===------------------------------------------------------------------===//
    if (failed(unrollForLoops(device, builder, objectFifoTiles))) {
      signalPassFailure();
    }

    //===------------------------------------------------------------------===//
    // Replace ops
    //===------------------------------------------------------------------===//
    for (auto coreOp : device.getOps<CoreOp>()) {
      DenseMap<ObjectFifoAcquireOp, std::vector<BufferOp *>>
          subviews;  // maps each "subview" to its buffer references (subviews
      // are created by AcquireOps)
      DenseMap<std::pair<ObjectFifoCreateOp, int>, std::vector<int>>
          acquiresPerFifo;  // maps each objFifo to indices of buffers acquired
      // in latest subview of that objFifo (useful to
      // cascade acquired elements to next AcquireOp)
      DenseMap<std::pair<ObjectFifoCreateOp, int>,
               std::vector<ObjectFifoReleaseOp>>
          releaseOps;  // useful to check which ReleaseOp has taken place before
      // an AcquireOp per objFifo
      DenseMap<std::pair<ObjectFifoCreateOp, int>, int>
          acqPerFifo;  // maps each objFifo to its next index to acquire within
      // this CoreOp
      DenseMap<std::pair<ObjectFifoCreateOp, int>, int>
          relPerFifo;  // maps each objFifo to its next index to release within
      // this CoreOp

      //===----------------------------------------------------------------===//
      // Replace objectFifo.release ops
      //===----------------------------------------------------------------===//
      coreOp.walk([&](ObjectFifoReleaseOp releaseOp) {
        builder.setInsertionPointAfter(releaseOp);
        ObjectFifoCreateOp op = releaseOp.getObjectFifo();
        auto port = releaseOp.getPort();
        auto portNum = port == ObjectFifoPort::Produce ? 0 : 1;
        auto core = releaseOp->getParentOfType<CoreOp>();

        if (auto linkOp = getOptionalLinkOp(op)) {
          if (core.getTile() == *linkOp->getOptionalSharedTile()) {
            releaseOp->emitOpError(
                "currently cannot access objectFifo used in "
                "ObjectFifoLinkOp");
            return;
          }
        }

        // update index of next element to release for this objectFifo
        updateAndReturnIndex(relPerFifo, {op, portNum});

        // release locks
        int numLocks = releaseOp.relNumber();
        createUseLocks(builder, op, port, relPerFifo, numLocks,
                       LockAction::Release);

        // register release op
        if (releaseOps.find({op, portNum}) != releaseOps.end()) {
          releaseOps[{op, portNum}].push_back(releaseOp);
        } else {
          std::vector release = {releaseOp};
          releaseOps[{op, portNum}] = release;
        }
      });

      //===----------------------------------------------------------------===//
      // Replace objectFifo.acquire ops
      //===----------------------------------------------------------------===//
      coreOp.walk([&](ObjectFifoAcquireOp acquireOp) {
        ObjectFifoCreateOp op = acquireOp.getObjectFifo();
        builder.setInsertionPointAfter(acquireOp);
        auto port = acquireOp.getPort();
        auto portNum = port == ObjectFifoPort::Produce ? 0 : 1;
        auto core = acquireOp->getParentOfType<CoreOp>();

        auto linkOp = getOptionalLinkOp(op);
        if (linkOp) {
          if (core.getTile() == *linkOp->getOptionalSharedTile()) {
            acquireOp->emitOpError(
                "currently cannot access objectFifo used in "
                "ObjectFifoLinkOp");
            return;
          }
        }

        // index of next element to acquire for this objectFifo
        int start = updateAndReturnIndex(
            acqPerFifo, {op, portNum});  // useful for keeping track of which
        // indices are acquired

        // check how many elements have been released in between this AcquireOp
        // and the previous one
        int numRel = 0;
        for (auto relOp : releaseOps[{op, portNum}]) {
          // TODO: operations may not be in the same block: currently only
          // support one block level of difference

          if (ObjectFifoCreateOp otherOp = relOp.getObjectFifo();
              op == otherOp) {
            // if they are already in the same block, check if releaseOp
            // happened before
            if (acquireOp.getOperation()->getBlock() ==
                relOp.getOperation()->getBlock()) {
              if (!acquireOp->isBeforeInBlock(relOp)) {
                releaseOps[{op, portNum}].erase(
                    releaseOps[{op, portNum}].begin());
                // to ensure that we do not account
                // the ReleaseOps again later,
                // after the subview is created
                numRel += relOp.relNumber();
              }
            } else {
              // else, check if releaseOp happened before the block region
              // with the acquireOp
              if (Operation *acqBlockDefOp =
                      acquireOp.getOperation()->getBlock()->getParentOp();
                  relOp.getOperation()->getBlock() ==
                  acqBlockDefOp->getBlock()) {
                if (!acqBlockDefOp->isBeforeInBlock(relOp)) {
                  releaseOps[{op, portNum}].erase(
                      releaseOps[{op, portNum}]
                          .begin());  // to ensure that we do not account
                  // the ReleaseOps again later, after
                  // the subview is created
                  numRel += relOp.relNumber();
                }

                // else, check if the block region with releaseOp happened
                // before...
              } else {
                // ...the acquireOp
                if (Operation *relBlockDefOp =
                        relOp.getOperation()->getBlock()->getParentOp();
                    acquireOp.getOperation()->getBlock() ==
                    relBlockDefOp->getBlock()) {
                  if (!acquireOp->isBeforeInBlock(relBlockDefOp)) {
                    releaseOps[{op, portNum}].erase(
                        releaseOps[{op, portNum}]
                            .begin());  // to ensure that we do not account
                    // the ReleaseOps again later,
                    // after the subview is created
                    numRel += relOp.relNumber();
                  }

                  // ...the block region with the acquireOp
                } else if (acqBlockDefOp->getBlock() ==
                           relBlockDefOp->getBlock()) {
                  if (!acqBlockDefOp->isBeforeInBlock(relBlockDefOp)) {
                    releaseOps[{op, portNum}].erase(
                        releaseOps[{op, portNum}]
                            .begin());  // to ensure that we do not account
                    // the ReleaseOps again later,
                    // after the subview is created
                    numRel += relOp.relNumber();
                  }
                }
              }
            }
          }
        }

        // track indices of elements to acquire
        std::vector<int> acquiredIndices;
        if (!acquiresPerFifo[{op, portNum}].empty()) {
          // take into account what has already been acquired by previous
          // AcquireOp in program order
          acquiredIndices = acquiresPerFifo[{op, portNum}];
          // take into account what has been released in-between
          if (static_cast<size_t>(numRel) > acquiredIndices.size()) {
            acquireOp->emitOpError(
                "cannot release more elements than are "
                "already acquired");
            return;
          }
          for (int i = 0; i < numRel; i++)
            acquiredIndices.erase(acquiredIndices.begin());
        }

        // acquire locks
        int numLocks = acquireOp.acqNumber();
        int alreadyAcq = acquiredIndices.size();
        int numCreate;
        if (numLocks > alreadyAcq)
          numCreate = numLocks - alreadyAcq;
        else
          numCreate = 0;

        auto dev = op->getParentOfType<DeviceOp>();
        if (auto &targetArch = dev.getTargetModel();
            targetArch.getTargetArch() == AIEArch::AIE1)
          createUseLocks(builder, op, port, acqPerFifo, numCreate,
                         LockAction::Acquire);
        else
          createUseLocks(builder, op, port, acqPerFifo, numCreate,
                         LockAction::AcquireGreaterEqual);

        // if objFifo was linked with others, find which objFifos
        // elements to use
        ObjectFifoCreateOp target = op;
        if (linkOp)
          if (objFifoLinks.find(*linkOp) != objFifoLinks.end())
            target = objFifoLinks[*linkOp];

        // create subview: buffers that were already acquired + new acquires
        for (int i = 0; i < numCreate; i++) {
          acquiredIndices.push_back(start);
          start = (start + 1) % op.size();
        }
        std::vector<BufferOp *> subviewRefs;
        subviewRefs.reserve(acquiredIndices.size());
        for (auto index : acquiredIndices)
          subviewRefs.push_back(&buffersPerFifo[target][index]);

        subviews[acquireOp] = subviewRefs;
        acquiresPerFifo[{op, portNum}] = acquiredIndices;
      });

      //===----------------------------------------------------------------===//
      // Replace subview.access ops
      //===----------------------------------------------------------------===//
      coreOp.walk([&](ObjectFifoSubviewAccessOp accessOp) {
        auto acqOp = accessOp.getSubview().getDefiningOp<ObjectFifoAcquireOp>();
        if (ObjectFifoCreateOp op = acqOp.getObjectFifo();
            getOptionalLinkOp(op)) {
          accessOp->emitOpError(
              "currently cannot access objectFifo used in "
              "ObjectFifoLinkOp");
          return;
        }
        accessOp.getOutput().replaceAllUsesWith(
            subviews[acqOp][accessOp.getIndex()]->getBuffer());
      });
    }

    // make global symbols to replace the to be erased ObjectFifoCreateOps
    for (auto createOp : device.getOps<ObjectFifoCreateOp>()) {
      builder.setInsertionPointToStart(&device.getBodyRegion().front());
      auto sym_name = createOp.getName();
      createOp->setAttr(SymbolTable::getSymbolAttrName(),
                        builder.getStringAttr("__erase_" + sym_name));
      auto memrefType = llvm::cast<AIEObjectFifoType>(createOp.getElemType())
                            .getElementType();
      builder.create<memref::GlobalOp>(builder.getUnknownLoc(), sym_name,
                                       builder.getStringAttr("public"),
                                       memrefType, nullptr, false, nullptr);
    }

    //===------------------------------------------------------------------===//
    // Remove old ops
    //===------------------------------------------------------------------===//
    SetVector<Operation *> opsToErase;
    device.walk([&](Operation *op) {
      if (isa<ObjectFifoCreateOp, ObjectFifoLinkOp,
              ObjectFifoRegisterExternalBuffersOp, ObjectFifoAcquireOp,
              ObjectFifoSubviewAccessOp, ObjectFifoReleaseOp>(op))
        opsToErase.insert(op);
    });
    topologicalSort(opsToErase);
    IRRewriter rewriter(&getContext());
    for (auto it = opsToErase.rbegin(); it != opsToErase.rend(); ++it)
      (*it)->erase();
  }
};

std::unique_ptr<OperationPass<DeviceOp>>
AIE::createAIEObjectFifoStatefulTransformPass() {
  return std::make_unique<AIEObjectFifoStatefulTransformPass>();
}

//===- AIEPathfinder.cpp ----------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

#define OVER_CAPACITY_COEFF 0.02
#define USED_CAPACITY_COEFF 0.02
#define DEMAND_COEFF 1.1

LogicalResult DynamicTileAnalysis::runAnalysis(DeviceOp &device) {
  LLVM_DEBUG(llvm::dbgs() << "\t---Begin DynamicTileAnalysis Constructor---\n");
  // find the maxCol and maxRow
  maxCol = 0;
  maxRow = 0;
  for (TileOp tileOp : device.getOps<TileOp>()) {
    maxCol = std::max(maxCol, tileOp.colIndex());
    maxRow = std::max(maxRow, tileOp.rowIndex());
  }

  pathfinder->initialize(maxCol, maxRow, device.getTargetModel());

  // for each flow in the device, add it to pathfinder
  // each source can map to multiple different destinations (fanout)
  for (FlowOp flowOp : device.getOps<FlowOp>()) {
    TileOp srcTile = cast<TileOp>(flowOp.getSource().getDefiningOp());
    TileOp dstTile = cast<TileOp>(flowOp.getDest().getDefiningOp());
    TileID srcCoords = {srcTile.colIndex(), srcTile.rowIndex()};
    TileID dstCoords = {dstTile.colIndex(), dstTile.rowIndex()};
    Port srcPort = {flowOp.getSourceBundle(), flowOp.getSourceChannel()};
    Port dstPort = {flowOp.getDestBundle(), flowOp.getDestChannel()};
    LLVM_DEBUG(llvm::dbgs()
               << "\tAdding Flow: (" << srcCoords.col << ", " << srcCoords.row
               << ")" << stringifyWireBundle(srcPort.bundle) << srcPort.channel
               << " -> (" << dstCoords.col << ", " << dstCoords.row << ")"
               << stringifyWireBundle(dstPort.bundle) << dstPort.channel
               << "\n");
    pathfinder->addFlow(srcCoords, srcPort, dstCoords, dstPort);
  }

  // add existing connections so Pathfinder knows which resources are
  // available search all existing SwitchBoxOps for exising connections
  for (SwitchboxOp switchboxOp : device.getOps<SwitchboxOp>()) {
    for (ConnectOp connectOp : switchboxOp.getOps<ConnectOp>()) {
      if (!pathfinder->addFixedConnection(connectOp))
        return switchboxOp.emitOpError() << "Couldn't connect " << connectOp;
    }
  }

  // all flows are now populated, call the congestion-aware pathfinder
  // algorithm
  // check whether the pathfinder algorithm creates a legal routing
  if (auto maybeFlowSolutions = pathfinder->findPaths(maxIterations))
    flowSolutions = maybeFlowSolutions.value();
  else
    return device.emitError("Unable to find a legal routing");

  // initialize all flows as unprocessed to prep for rewrite
  for (const auto &[pathEndPoint, switchSetting] : flowSolutions) {
    processedFlows[pathEndPoint] = false;
    LLVM_DEBUG(llvm::dbgs() << "Flow starting at (" << pathEndPoint.sb.col
                            << "," << pathEndPoint.sb.row << "):\t");
    LLVM_DEBUG(llvm::dbgs() << switchSetting);
  }

  // fill in coords to TileOps, SwitchboxOps, and ShimMuxOps
  for (auto tileOp : device.getOps<TileOp>()) {
    int col, row;
    col = tileOp.colIndex();
    row = tileOp.rowIndex();
    maxCol = std::max(maxCol, col);
    maxRow = std::max(maxRow, row);
    assert(coordToTile.count({col, row}) == 0);
    coordToTile[{col, row}] = tileOp;
  }
  for (auto switchboxOp : device.getOps<SwitchboxOp>()) {
    int col = switchboxOp.colIndex();
    int row = switchboxOp.rowIndex();
    assert(coordToSwitchbox.count({col, row}) == 0);
    coordToSwitchbox[{col, row}] = switchboxOp;
  }
  for (auto shimmuxOp : device.getOps<ShimMuxOp>()) {
    int col = shimmuxOp.colIndex();
    int row = shimmuxOp.rowIndex();
    assert(coordToShimMux.count({col, row}) == 0);
    coordToShimMux[{col, row}] = shimmuxOp;
  }

  LLVM_DEBUG(llvm::dbgs() << "\t---End DynamicTileAnalysis Constructor---\n");
  return success();
}

TileOp DynamicTileAnalysis::getTile(OpBuilder &builder, int col, int row) {
  if (coordToTile.count({col, row})) {
    return coordToTile[{col, row}];
  }
  auto tileOp = builder.create<TileOp>(builder.getUnknownLoc(), col, row);
  coordToTile[{col, row}] = tileOp;
  maxCol = std::max(maxCol, col);
  maxRow = std::max(maxRow, row);
  return tileOp;
}

SwitchboxOp DynamicTileAnalysis::getSwitchbox(OpBuilder &builder, int col,
                                              int row) {
  assert(col >= 0);
  assert(row >= 0);
  if (coordToSwitchbox.count({col, row})) {
    return coordToSwitchbox[{col, row}];
  }
  auto switchboxOp = builder.create<SwitchboxOp>(builder.getUnknownLoc(),
                                                 getTile(builder, col, row));
  SwitchboxOp::ensureTerminator(switchboxOp.getConnections(), builder,
                                builder.getUnknownLoc());
  coordToSwitchbox[{col, row}] = switchboxOp;
  maxCol = std::max(maxCol, col);
  maxRow = std::max(maxRow, row);
  return switchboxOp;
}

ShimMuxOp DynamicTileAnalysis::getShimMux(OpBuilder &builder, int col) {
  assert(col >= 0);
  int row = 0;
  if (coordToShimMux.count({col, row})) {
    return coordToShimMux[{col, row}];
  }
  assert(getTile(builder, col, row).isShimNOCTile());
  auto switchboxOp = builder.create<ShimMuxOp>(builder.getUnknownLoc(),
                                               getTile(builder, col, row));
  SwitchboxOp::ensureTerminator(switchboxOp.getConnections(), builder,
                                builder.getUnknownLoc());
  coordToShimMux[{col, row}] = switchboxOp;
  maxCol = std::max(maxCol, col);
  maxRow = std::max(maxRow, row);
  return switchboxOp;
}

void Pathfinder::initialize(int maxCol, int maxRow,
                            const AIETargetModel &targetModel) {
  // make grid of switchboxes
  int id = 0;
  for (int row = 0; row <= maxRow; row++) {
    for (int col = 0; col <= maxCol; col++) {
      auto [it, _] = grid.insert({{col, row}, SwitchboxNode{col, row, id++}});
      (void)graph.addNode(it->second);
      SwitchboxNode &thisNode = grid.at({col, row});
      if (row > 0) {  // if not in row 0 add channel to North/South
        SwitchboxNode &southernNeighbor = grid.at({col, row - 1});
        // get the number of outgoing connections on the south side - outgoing
        // because these correspond to rhs of a connect op
        if (uint32_t maxCapacity = targetModel.getNumDestSwitchboxConnections(
                col, row, WireBundle::South)) {
          edges.emplace_back(thisNode, southernNeighbor, WireBundle::South,
                             maxCapacity);
          (void)graph.connect(thisNode, southernNeighbor, edges.back());
        }
        // get the number of incoming connections on the south side - incoming
        // because they correspond to connections on the southside that are then
        // routed using internal connect ops through the switchbox (i.e., lhs of
        // connect ops)
        if (uint32_t maxCapacity = targetModel.getNumSourceSwitchboxConnections(
                col, row, WireBundle::South)) {
          edges.emplace_back(southernNeighbor, thisNode, WireBundle::North,
                             maxCapacity);
          (void)graph.connect(southernNeighbor, thisNode, edges.back());
        }
      }

      if (col > 0) {  // if not in col 0 add channel to East/West
        SwitchboxNode &westernNeighbor = grid.at({col - 1, row});
        if (uint32_t maxCapacity = targetModel.getNumDestSwitchboxConnections(
                col, row, WireBundle::West)) {
          edges.emplace_back(thisNode, westernNeighbor, WireBundle::West,
                             maxCapacity);
          (void)graph.connect(thisNode, westernNeighbor, edges.back());
        }
        if (uint32_t maxCapacity = targetModel.getNumSourceSwitchboxConnections(
                col, row, WireBundle::West)) {
          edges.emplace_back(westernNeighbor, thisNode, WireBundle::East,
                             maxCapacity);
          (void)graph.connect(westernNeighbor, thisNode, edges.back());
        }
      }
    }
  }
}

// Add a flow from src to dst can have an arbitrary number of dst locations due
// to fanout.
void Pathfinder::addFlow(TileID srcCoords, Port srcPort, TileID dstCoords,
                         Port dstPort) {
  // check if a flow with this source already exists
  for (auto &[src, dsts] : flows) {
    SwitchboxNode *existingSrc = src.sb;
    assert(existingSrc && "nullptr flow source");
    if (Port existingPort = src.port; existingSrc->col == srcCoords.col &&
                                      existingSrc->row == srcCoords.row &&
                                      existingPort == srcPort) {
      // find the vertex corresponding to the destination
      auto *matchingSb = std::find_if(
          graph.begin(), graph.end(), [&](const SwitchboxNode *sb) {
            return sb->col == dstCoords.col && sb->row == dstCoords.row;
          });
      assert(matchingSb != graph.end() && "didn't find flow dest");
      dsts.emplace_back(*matchingSb, dstPort);
      return;
    }
  }

  // If no existing flow was found with this source, create a new flow.
  auto *matchingSrcSb =
      std::find_if(graph.begin(), graph.end(), [&](const SwitchboxNode *sb) {
        return sb->col == srcCoords.col && sb->row == srcCoords.row;
      });
  assert(matchingSrcSb != graph.end() && "didn't find flow source");
  auto *matchingDstSb =
      std::find_if(graph.begin(), graph.end(), [&](const SwitchboxNode *sb) {
        return sb->col == dstCoords.col && sb->row == dstCoords.row;
      });
  assert(matchingDstSb != graph.end() && "didn't add flow destinations");
  flows.push_back({PathEndPointNode{*matchingSrcSb, srcPort},
                   std::vector<PathEndPointNode>{{*matchingDstSb, dstPort}}});
}

// Keep track of connections already used in the AIE; Pathfinder algorithm will
// avoid using these.
bool Pathfinder::addFixedConnection(ConnectOp connectOp) {
  auto sb = connectOp->getParentOfType<SwitchboxOp>();
  // TODO: keep track of capacity?
  if (sb.getTileOp().isShimNOCTile()) return true;

  TileID sbTile = sb.getTileID();
  WireBundle sourceBundle = connectOp.getSourceBundle();
  WireBundle destBundle = connectOp.getDestBundle();

  // find the correct Channel and indicate the fixed direction
  // outgoing connection
  auto matchingCh =
      std::find_if(edges.begin(), edges.end(), [&](ChannelEdge &ch) {
        return static_cast<TileID>(ch.src) == sbTile && ch.bundle == destBundle;
      });
  if (matchingCh != edges.end())
    return matchingCh->fixedCapacity.insert(connectOp.getDestChannel())
               .second ||
           true;

  // incoming connection
  matchingCh = std::find_if(edges.begin(), edges.end(), [&](ChannelEdge &ch) {
    return static_cast<TileID>(ch.target) == sbTile &&
           ch.bundle == getConnectingBundle(sourceBundle);
  });
  if (matchingCh != edges.end())
    return matchingCh->fixedCapacity.insert(connectOp.getSourceChannel())
               .second ||
           true;

  return false;
}

static constexpr double INF = std::numeric_limits<double>::max();

std::map<SwitchboxNode *, SwitchboxNode *> dijkstraShortestPaths(
    const SwitchboxGraph &graph, SwitchboxNode *src) {
  // Use std::map instead of DenseMap because DenseMap doesn't let you overwrite
  // tombstones.
  auto distance = std::map<SwitchboxNode *, double>();
  auto preds = std::map<SwitchboxNode *, SwitchboxNode *>();
  std::map<SwitchboxNode *, uint64_t> indexInHeap;
  typedef d_ary_heap_indirect<
      /*Value=*/SwitchboxNode *, /*Arity=*/4,
      /*IndexInHeapPropertyMap=*/std::map<SwitchboxNode *, uint64_t>,
      /*DistanceMap=*/std::map<SwitchboxNode *, double> &,
      /*Compare=*/std::less<>>
      MutableQueue;
  MutableQueue Q(distance, indexInHeap);

  for (SwitchboxNode *sb : graph) distance.emplace(sb, INF);
  distance[src] = 0.0;

  std::map<SwitchboxNode *, std::vector<ChannelEdge *>> edges;

  enum Color { WHITE, GRAY, BLACK };
  std::map<SwitchboxNode *, Color> colors;
  for (SwitchboxNode *sb : graph) {
    colors[sb] = WHITE;
    edges[sb] = {sb->getEdges().begin(), sb->getEdges().end()};
    std::sort(edges[sb].begin(), edges[sb].end(),
              [](const ChannelEdge *c1, ChannelEdge *c2) {
                return c1->getTargetNode().id < c2->getTargetNode().id;
              });
  }

  Q.push(src);
  while (!Q.empty()) {
    src = Q.top();
    Q.pop();
    for (ChannelEdge *e : edges[src]) {
      SwitchboxNode *dest = &e->getTargetNode();
      bool relax = distance[src] + e->demand < distance[dest];
      if (colors[dest] == WHITE) {
        if (relax) {
          distance[dest] = distance[src] + e->demand;
          preds[dest] = src;
          colors[dest] = GRAY;
        }
        Q.push(dest);
      } else if (colors[dest] == GRAY && relax) {
        distance[dest] = distance[src] + e->demand;
        preds[dest] = src;
      }
    }
    colors[src] = BLACK;
  }
  return preds;
}

// Perform congestion-aware routing for all flows which have been added.
// Use Dijkstra's shortest path to find routes, and use "demand" as the weights.
// If the routing finds too much congestion, update the demand weights
// and repeat the process until a valid solution is found.
// Returns a map specifying switchbox settings for all flows.
// If no legal routing can be found after maxIterations, returns empty vector.
std::optional<std::map<PathEndPoint, SwitchSettings>> Pathfinder::findPaths(
    const int maxIterations) {
  LLVM_DEBUG(llvm::dbgs() << "Begin Pathfinder::findPaths\n");
  int iterationCount = 0;
  std::map<PathEndPoint, SwitchSettings> routingSolution;

  // initialize all Channel histories to 0
  for (auto &ch : edges) ch.overCapacityCount = 0;

  // Check that every channel does not exceed max capacity.
  auto isLegal = [&] {
    bool legal = true;  // assume legal until found otherwise
    for (auto &e : edges) {
      if (e.usedCapacity > e.maxCapacity) {
        LLVM_DEBUG(llvm::dbgs()
                   << "Too much capacity on Edge (" << e.getTargetNode().col
                   << ", " << e.getTargetNode().row << ") . "
                   << stringifyWireBundle(e.bundle) << "\t: used_capacity = "
                   << e.usedCapacity << "\t: Demand = " << e.demand << "\n");
        e.overCapacityCount++;
        LLVM_DEBUG(llvm::dbgs()
                   << "over_capacity_count = " << e.overCapacityCount << "\n");
        legal = false;
        break;
      }
    }

    return legal;
  };

  do {
    LLVM_DEBUG(llvm::dbgs()
               << "Begin findPaths iteration #" << iterationCount << "\n");
    // update demand on all channels
    for (auto &ch : edges) {
      if (ch.fixedCapacity.size() >=
          static_cast<std::set<int>::size_type>(ch.maxCapacity)) {
        ch.demand = INF;
      } else {
        double history = 1.0 + OVER_CAPACITY_COEFF * ch.overCapacityCount;
        double congestion = 1.0 + USED_CAPACITY_COEFF * ch.usedCapacity;
        ch.demand = history * congestion;
      }
    }
    // if reach maxIterations, throw an error since no routing can be found
    if (++iterationCount > maxIterations) {
      LLVM_DEBUG(llvm::dbgs()
                 << "Pathfinder: maxIterations has been exceeded ("
                 << maxIterations
                 << " iterations)...unable to find routing for flows.\n");
      return std::nullopt;
    }

    // "rip up" all routes, i.e. set used capacity in each Channel to 0
    routingSolution.clear();
    for (auto &ch : edges) ch.usedCapacity = 0;

    // for each flow, find the shortest path from source to destination
    // update used_capacity for the path between them
    for (const auto &[src, dsts] : flows) {
      // Use dijkstra to find path given current demand from the start
      // switchbox; find the shortest paths to each other switchbox. Output is
      // in the predecessor map, which must then be processed to get individual
      // switchbox settings
      assert(src.sb && "nonexistent flow source");
      std::set<SwitchboxNode *> processed;
      std::map<SwitchboxNode *, SwitchboxNode *> preds =
          dijkstraShortestPaths(graph, src.sb);

      // trace the path of the flow backwards via predecessors
      // increment used_capacity for the associated channels
      SwitchSettings switchSettings;
      // set the input bundle for the source endpoint
      switchSettings[*src.sb].src = src.port;
      processed.insert(src.sb);
      for (const PathEndPointNode &endPoint : dsts) {
        SwitchboxNode *curr = endPoint.sb;
        assert(curr && "endpoint has no source switchbox");
        // set the output bundle for this destination endpoint
        switchSettings[*curr].dsts.insert(endPoint.port);

        // trace backwards until a vertex already processed is reached
        while (!processed.count(curr)) {
          // find the edge from the pred to curr by searching incident edges
          SmallVector<ChannelEdge *, 10> channels;
          graph.findIncomingEdgesToNode(*curr, channels);
          auto *matchingCh = std::find_if(
              channels.begin(), channels.end(),
              [&](ChannelEdge *ch) { return ch->src == *preds[curr]; });
          assert(matchingCh != channels.end() && "couldn't find ch");
          // incoming edge
          ChannelEdge *ch = *matchingCh;

          // don't use fixed channels
          while (ch->fixedCapacity.count(ch->usedCapacity)) ch->usedCapacity++;

          // add the entrance port for this Switchbox
          switchSettings[*curr].src = {getConnectingBundle(ch->bundle),
                                       ch->usedCapacity};
          // add the current Switchbox to the map of the predecessor
          switchSettings[*preds[curr]].dsts.insert(
              {ch->bundle, ch->usedCapacity});

          ch->usedCapacity++;
          // if at capacity, bump demand to discourage using this Channel
          if (ch->usedCapacity >= ch->maxCapacity) {
            LLVM_DEBUG(llvm::dbgs() << "ch over capacity: " << ch << "\n");
            // this means the order matters!
            ch->demand *= DEMAND_COEFF;
          }

          processed.insert(curr);
          curr = preds[curr];
        }
      }
      // add this flow to the proposed solution
      routingSolution[src] = switchSettings;
    }
  } while (!isLegal());  // continue iterations until a legal routing is found

  return routingSolution;
}
//===- AIEXToStandard.cpp ---------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2023, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

using namespace xilinx::AIEX;

template <typename MyAIEXOp>
struct AIEXOpRemoval : OpConversionPattern<MyAIEXOp> {
  using OpConversionPattern<MyAIEXOp>::OpConversionPattern;
  using OpAdaptor = typename MyAIEXOp::Adaptor;
  ModuleOp &module;

  AIEXOpRemoval(MLIRContext *context, ModuleOp &m, PatternBenefit benefit = 1)
      : OpConversionPattern<MyAIEXOp>(context, benefit), module(m) {}

  LogicalResult matchAndRewrite(
      MyAIEXOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    Operation *Op = op.getOperation();
    rewriter.eraseOp(Op);
    return success();
  }
};

struct AIEXToStandardPass
    : xilinx::AIEX::impl::AIEXToStandardBase<AIEXToStandardPass> {
  void runOnOperation() override {
    ModuleOp m = getOperation();
    ConversionTarget target(getContext());
    RewritePatternSet removepatterns(&getContext());
    removepatterns.add<AIEXOpRemoval<NpuDmaMemcpyNdOp>>(m.getContext(), m);
    removepatterns.add<AIEXOpRemoval<NpuDmaWaitOp>>(m.getContext(), m);
    removepatterns.add<AIEXOpRemoval<NpuPushQueueOp>>(m.getContext(), m);
    removepatterns.add<AIEXOpRemoval<NpuWriteRTPOp>>(m.getContext(), m);
    removepatterns.add<AIEXOpRemoval<NpuWrite32Op>>(m.getContext(), m);
    removepatterns.add<AIEXOpRemoval<NpuSyncOp>>(m.getContext(), m);
    removepatterns.add<AIEXOpRemoval<NpuWriteBdOp>>(m.getContext(), m);
    removepatterns.add<AIEXOpRemoval<NpuAddressPatchOp>>(m.getContext(), m);

    if (failed(applyPartialConversion(m, target, std::move(removepatterns))))
      signalPassFailure();
  }
};

std::unique_ptr<OperationPass<ModuleOp>> AIEX::createAIEXToStandardPass() {
  return std::make_unique<AIEXToStandardPass>();
}

namespace mlir::iree_compiler::AMDAIE {
void registerAIETransformPasses() {
  xilinx::AIE::registerAIEAssignLockIDs();
  xilinx::AIE::registerAIEAssignBufferDescriptorIDs();
  xilinx::AIE::registerAIEAssignBufferAddressesBasic();
  xilinx::AIE::registerAIECoreToStandard();
  xilinx::AIE::registerAIERoutePathfinderFlows();
  xilinx::AIE::registerAIELocalizeLocks();
  xilinx::AIE::registerAIEObjectFifoStatefulTransform();
}
}  // namespace mlir::iree_compiler::AMDAIE

namespace mlir::iree_compiler::AMDAIE {
void registerAIEXTransformPasses() {
  xilinx::AIEX::registerAIEXToStandard();
  xilinx::AIEX::registerAIEDmaToNpu();
}
}  // namespace mlir::iree_compiler::AMDAIE
