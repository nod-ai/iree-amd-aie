//===- XCLBinGen.cpp -------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2024 Xilinx Inc.
//
//===---------------------------------------------------------------------===//

#include "AIEChessHack.h"
#include "XCLBinGen.h"

#include <regex>
#include <sstream>
#include <unordered_map>
#include <utility>

#include "AIETargets.h"
#include "Passes.h"
#include "aie/AIEAssignBufferAddressesBasic.h"
#include "aie/Conversion/AIEVecToLLVM/AIEVecToLLVM.h"
#include "aie/Dialect/AIEVec/Pipelines/Passes.h"
#include "aie/Target/LLVMIR/Dialect/XLLVM/XLLVMToLLVMIRTranslation.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/ToolOutputFile.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVMPass.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Transforms/Passes.h"

#ifdef _WIN32
#include "windows.h"
// For UUID stuff
#include "rpcdce.h"

#define setenv(name, var, ignore) _putenv_s(name, var)
#else
#include <uuid/uuid.h>
#endif

using namespace llvm;
using namespace mlir;
using namespace xilinx;

namespace {

// Apply the pass manager specific options of the XCLBinGenConfig to the pass
// manager. These control when (if ever) and what IR gets printed between
// passes, and whether the pass manager uses multi-theading.
void applyConfigToPassManager(XCLBinGenConfig &TK, PassManager &pm) {
  //  pm.getContext()->disableMultithreading(TK.DisableThreading);

  bool printBefore = TK.PrintIRBeforeAll;
  auto shouldPrintBeforePass = [printBefore](Pass *, Operation *) {
    return printBefore;
  };

  bool printAfter = TK.PrintIRAfterAll;
  auto shouldPrintAfterPass = [printAfter](Pass *, Operation *) {
    return printAfter;
  };

  pm.enableIRPrinting(shouldPrintBeforePass, shouldPrintAfterPass,
                      TK.PrintIRModuleScope);

  bool timing = TK.Timing;
  if (timing) pm.enableTiming();
}
}  // namespace

LogicalResult xilinx::findVitis(XCLBinGenConfig &TK) {
  const char *env_vitis = ::getenv("VITIS");
  if (env_vitis == nullptr) {
    if (auto vpp = sys::findProgramByName("v++")) {
      SmallString<64> real_vpp;
      std::error_code err = sys::fs::real_path(vpp.get(), real_vpp);
      if (!err) {
        sys::path::remove_filename(real_vpp);
        sys::path::remove_filename(real_vpp);
        ::setenv("VITIS", real_vpp.c_str(), 1);
        dbgs() << "Found Vitis at " << real_vpp.c_str() << "\n";
      }
    }
  }
  env_vitis = ::getenv("VITIS");
  if (env_vitis != nullptr) {
    SmallString<64> vitis_path(env_vitis);
    SmallString<64> vitis_bin_path(vitis_path);
    sys::path::append(vitis_bin_path, "bin");

    SmallString<64> aietools_path(vitis_path);
    sys::path::append(aietools_path, "aietools");
    if (!sys::fs::exists(aietools_path)) {
      aietools_path = vitis_path;
      sys::path::append(aietools_path, "cardano");
    }
    TK.AIEToolsDir = std::string(aietools_path);
    ::setenv("AIETOOLS", TK.AIEToolsDir.c_str(), 1);

    SmallString<64> aietools_bin_path(aietools_path);
    sys::path::append(aietools_bin_path, "bin", "unwrapped", "lnx64.o");
    const char *env_path = ::getenv("PATH");
    if (env_path == nullptr) env_path = "";
    SmallString<128> new_path(env_path);
    if (new_path.size()) new_path += sys::EnvPathSeparator;
    new_path += aietools_bin_path;
    new_path += sys::EnvPathSeparator;
    new_path += vitis_bin_path;

    SmallString<64> chessccPath(aietools_path);
    sys::path::append(chessccPath, "tps", "lnx64", "target");
    sys::path::append(chessccPath, "bin", "LNa64bin");
    new_path += sys::EnvPathSeparator;
    new_path += chessccPath;

    ::setenv("PATH", new_path.c_str(), 1);

    SmallString<64> lnx64o(TK.AIEToolsDir);
    sys::path::append(lnx64o, "lib", "lnx64.o");
    SmallString<64> dotLib(TK.AIEToolsDir);
    sys::path::append(dotLib, "lnx64", "tools", "dot", "lib");
    SmallString<64> ldLibraryPath(::getenv("LD_LIBRARY_PATH"));
    ::setenv(
        "LD_LIBRARY_PATH",
        (lnx64o + std::string{sys::EnvPathSeparator} + dotLib + ldLibraryPath)
            .str()
            .c_str(),
        1);

    SmallString<64> rdiDataDir_(TK.AIEToolsDir);
    sys::path::append(rdiDataDir_, "data");
    ::setenv("RDI_DATADIR", rdiDataDir_.c_str(), 1);

    return success();
  } else {
    return failure();
  }
}

static std::string getUUIDString() {
  std::string val;
#ifdef _WIN32
  UUID *uuid;
  RPC_STATUS status;
  status = UuidCreate(uuid);
  if (status != RPC_S_OK) errs() << "Failed to create UUID\n";
  RPC_CSTR *uuidstring;
  status = UuidToStringA(uuid, uuidstring);
  if (status != RPC_S_OK) errs() << "Failed to convert UUID to string\n";
  val = std::string((char *)uuidstring);
  status = RpcStringFreeA(uuidstring);
  if (status != RPC_S_OK) errs() << "Failed to free UUID string\n";
#else
  uuid_t binuuid;
  uuid_generate_random(binuuid);
  char uuid[37];
  uuid_unparse_lower(binuuid, uuid);
  val = std::string(uuid);
#endif
  return val;
}

static void addLowerToLLVMPasses(OpPassManager &pm) {
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());
  pm.addPass(xilinx::aievec::createConvertAIEVecToLLVMPass());

  pm.addPass(createConvertVectorToLLVMPass());
  pm.addPass(memref::createExpandStridedMetadataPass());
  pm.addPass(createLowerAffinePass());
  pm.addPass(createConvertMathToLLVMPass());
  pm.addPass(createArithToLLVMConversionPass());
  pm.addPass(createFinalizeMemRefToLLVMConversionPass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());
  ConvertFuncToLLVMPassOptions opts;
  opts.useBarePtrCallConv = true;
  pm.addPass(createConvertFuncToLLVMPass(opts));
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());
  pm.addPass(createConvertControlFlowToLLVMPass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());
}

int runTool(StringRef Program, ArrayRef<std::string> Args, bool Verbose,
            std::optional<ArrayRef<StringRef>> Env = std::nullopt) {
  if (Verbose) {
    llvm::outs() << "Run:";
    if (Env)
      for (auto &s : *Env) llvm::outs() << " " << s;
    llvm::outs() << " " << Program;
    for (auto &s : Args) llvm::outs() << " " << s;
    llvm::outs() << "\n";
  }
  std::string err_msg;
  sys::ProcessStatistics stats;
  std::optional<sys::ProcessStatistics> opt_stats(stats);
  SmallVector<StringRef, 8> PArgs = {Program};
  PArgs.append(Args.begin(), Args.end());
  int result = sys::ExecuteAndWait(Program, PArgs, Env, {}, 0, 0, &err_msg,
                                   nullptr, &opt_stats);
  if (Verbose)
    llvm::outs() << (result == 0 ? "Succeeded " : "Failed ") << "in "
                 << std::chrono::duration_cast<std::chrono::duration<float>>(
                        stats.TotalTime)
                        .count()
                 << " code: " << result << "\n";
  return result;
}

const char *_CHESS_INTRINSIC_WRAPPER_LL = R"chess(
; ModuleID = 'aie_runtime_lib/AIE2/chess_intrinsic_wrapper.cpp'
source_filename = "aie_runtime_lib/AIE2/chess_intrinsic_wrapper.cpp"

%struct.ipd.custom_type.uint2_t.uint2_t = type { i2 }

; Function Attrs: mustprogress nounwind
define dso_local void @llvm___aie2___acquire(i32 noundef %0, i32 noundef %1) local_unnamed_addr addrspace(1) #0 {
  tail call addrspace(1) void @llvm.chess_memory_fence()
  tail call addrspace(1) void @_Z25chess_separator_schedulerv() #4
  tail call x86_regcallcc addrspace(1) void @__regcall3__chessintr_void_acquire_guarded___uint___uint(i32 zeroext %0, i32 zeroext %1) #4
  tail call addrspace(1) void @_Z25chess_separator_schedulerv() #4
  tail call addrspace(1) void @llvm.chess_memory_fence()
  ret void
}

; Function Attrs: mustprogress nounwind
define dso_local void @llvm___aie2___release(i32 noundef %0, i32 noundef %1) local_unnamed_addr addrspace(1) #0 {
  tail call addrspace(1) void @llvm.chess_memory_fence()
  tail call addrspace(1) void @_Z25chess_separator_schedulerv() #4
  tail call x86_regcallcc addrspace(1) void @__regcall3__chessintr_void_release_guarded___uint___sint(i32 zeroext %0, i32 signext %1) #4
  tail call addrspace(1) void @_Z25chess_separator_schedulerv() #4
  tail call addrspace(1) void @llvm.chess_memory_fence()
  ret void
}

; Function Attrs: nounwind
define dso_local void @llvm___aie___event0() local_unnamed_addr addrspace(1) #1 {
  tail call x86_regcallcc addrspace(1) void @__regcall3__chessintr_void_event_uint2_t(%struct.ipd.custom_type.uint2_t.uint2_t zeroinitializer) #4
  ret void
}

; Function Attrs: nounwind
define dso_local void @llvm___aie___event1() local_unnamed_addr addrspace(1) #1 {
  tail call x86_regcallcc addrspace(1) void @__regcall3__chessintr_void_event_uint2_t(%struct.ipd.custom_type.uint2_t.uint2_t { i2 1 }) #4
  ret void
}

; Function Attrs: mustprogress nounwind willreturn
declare void @llvm.chess_memory_fence() addrspace(1) #2

; Function Attrs: inaccessiblememonly nounwind
declare dso_local void @_Z25chess_separator_schedulerv() local_unnamed_addr addrspace(1) #3

; Function Attrs: inaccessiblememonly nounwind
declare dso_local x86_regcallcc void @__regcall3__chessintr_void_acquire_guarded___uint___uint(i32 zeroext, i32 zeroext) local_unnamed_addr addrspace(1) #3

; Function Attrs: inaccessiblememonly nounwind
declare dso_local x86_regcallcc void @__regcall3__chessintr_void_release_guarded___uint___sint(i32 zeroext, i32 signext) local_unnamed_addr addrspace(1) #3

; Function Attrs: inaccessiblememonly nounwind
declare dso_local x86_regcallcc void @__regcall3__chessintr_void_event_uint2_t(%struct.ipd.custom_type.uint2_t.uint2_t) local_unnamed_addr addrspace(1) #3

attributes #0 = { mustprogress nounwind "frame-pointer"="all" "min-legal-vector-width"="0" "no-builtin-memcpy" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #1 = { nounwind "frame-pointer"="all" "min-legal-vector-width"="0" "no-builtin-memcpy" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #2 = { mustprogress nounwind willreturn }
attributes #3 = { inaccessiblememonly nounwind "frame-pointer"="all" "no-builtin-memcpy" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #4 = { inaccessiblememonly nounwind "no-builtin-memcpy" }

!llvm.linker.options = !{}
!llvm.module.flags = !{!0, !1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 7, !"frame-pointer", i32 2}
)chess";

std::vector<std::string> chessArgs(const std::string &AIEToolsDir,
                                   std::string workDir) {
  SmallString<64> chessClang(AIEToolsDir);
  sys::path::append(chessClang, "tps", "lnx64", "target");
  sys::path::append(chessClang, "bin", "LNa64bin", "chess-clang");
  SmallString<64> procModelLib(AIEToolsDir);
  sys::path::append(procModelLib, "data", "aie_ml", "lib");
  return {
      "+P",
      "4",  // parallel compilation (function + file level)
      "-p",
      "me",  // parallel compilation (function level only)
      "-C",
      "Release_LLVM",  // configuration
      "-D__AIENGINE__",
      "-D__AIE_ARCH__=20",
      "-D__AIEARCH__=20",
      "-Y",
      "clang=" + chessClang.str().str(),
      "-P",
      procModelLib.str().str(),  // processor model directory
      "-d",                      // disassemble output
      "-f",                      // use LLVM frontend
      "+w",
      std::move(workDir),
  };
}

// Generate the elf files for the core
static LogicalResult generateCoreElfFiles(ModuleOp moduleOp,
                                          const StringRef objFile,
                                          XCLBinGenConfig &TK) {
  auto deviceOps = moduleOp.getOps<AIE::DeviceOp>();
  if (!llvm::hasSingleElement(deviceOps))
    return moduleOp.emitOpError(": expected a single device op");

  AIE::DeviceOp deviceOp = *deviceOps.begin();
  auto tileOps = deviceOp.getOps<AIE::TileOp>();

  std::string errorMessage;

  for (auto tileOp : tileOps) {
    int col = tileOp.colIndex();
    int row = tileOp.rowIndex();
    auto coreOp = tileOp.getCoreOp();
    if (!coreOp) continue;

    std::string elfFileName;
    if (auto fileAttr = coreOp.getElfFileAttr()) {
      elfFileName = std::string(fileAttr.getValue());
    } else {
      elfFileName = std::string("core_") + std::to_string(col) + "_" +
                    std::to_string(row) + ".elf";
      coreOp.setElfFile(elfFileName);
    }

    SmallString<64> elfFile(TK.TempDir);
    sys::path::append(elfFile, elfFileName);

    // Use xbridge (to remove any peano dependency with use-chess option)
    SmallString<64> bcfPath(TK.TempDir);
    sys::path::append(bcfPath, elfFileName + ".bcf");

    {
      auto bcfOutput = openOutputFile(bcfPath, &errorMessage);
      if (!bcfOutput) return coreOp.emitOpError(errorMessage);

      if (failed(AIE::AIETranslateToBCF(moduleOp, bcfOutput->os(), col, row)))
        return coreOp.emitOpError(": Failed to generate BCF");
      bcfOutput->keep();
    }

    std::vector<std::string> extractedIncludes;
    {
      auto bcfFileIn = openInputFile(bcfPath, &errorMessage);
      if (!bcfFileIn) return moduleOp.emitOpError(errorMessage);

      std::string bcfFile = std::string(bcfFileIn->getBuffer());
      std::regex r("_include _file (.*)");
      auto begin = std::sregex_iterator(bcfFile.begin(), bcfFile.end(), r);
      auto end = std::sregex_iterator();
      for (std::sregex_iterator i = begin; i != end; ++i)
        extractedIncludes.push_back(i->str(1));
    }

    SmallString<64> chessExe(TK.AIEToolsDir);
    sys::path::append(chessExe, "bin", "unwrapped", "lnx64.o", "xchesscc");
    SmallString<64> chessworkDir(TK.TempDir);
    sys::path::append(chessworkDir, "chesswork");
    SmallVector<std::string> flags{"+l", std::string(bcfPath),
                                   "-o", std::string(elfFile),
                                   "-f", std::string(objFile)};
    for (const auto &inc : extractedIncludes) flags.push_back(inc);
    auto chessArgs_ = chessArgs(TK.AIEToolsDir, chessworkDir.str().str());
    chessArgs_.insert(chessArgs_.end(), flags.begin(), flags.end());
    if (!sys::fs::exists(chessExe))
      return moduleOp.emitOpError(": chess can't be found");
    if (runTool(chessExe, chessArgs_, TK.Verbose) != 0)
      return coreOp.emitOpError(": Failed to link with xbridge");
  }
  return success();
}

static LogicalResult generateCDO(MLIRContext *context, ModuleOp moduleOp,
                                 XCLBinGenConfig &TK) {
  ModuleOp copy = moduleOp.clone();
  if (failed(AIE::AIETranslateToCDODirect(copy, TK.TempDir)))
    return moduleOp.emitOpError(": failed to emit CDO");
  copy->erase();
  return success();
}

static json::Object makeKernelJSON(std::string name, std::string id,
                                   std::string instance, int numArgs) {
  json::Array args{json::Object{{"name", "instr"},
                                {"memory-connection", "SRAM"},
                                {"address-qualifier", "GLOBAL"},
                                {"type", "char *"},
                                {"offset", "0x00"}},
                   json::Object{{"name", "ninstr"},
                                {"address-qualifier", "SCALAR"},
                                {"type", "uint64_t"},
                                {"offset", "0x08"}}};
  for (int arg = 0; arg < numArgs; ++arg) {
    args.push_back(json::Object{{"name", "bo" + std::to_string(arg)},
                                {"memory-connection", "HOST"},
                                {"address-qualifier", "GLOBAL"},
                                {"type", "char *"},
                                {"offset", std::to_string(0x10 + 0x8 * arg)}});
  }

  return json::Object{
      {"name", name},
      {"type", "dpu"},
      {"extended-data",
       json::Object{
           {"subtype", "DPU"}, {"functional", "1"}, {"dpu_kernel_id", id}}},
      {"arguments", std::move(args)},
      {"instances", json::Array{json::Object{{"name", instance}}}}};
}

static LogicalResult generateXCLBin(MLIRContext *context, ModuleOp moduleOp,
                                    XCLBinGenConfig &TK,
                                    const StringRef &Output) {
  std::string errorMessage;
  // Create mem_topology.json.
  SmallString<64> memTopologyJsonFile(TK.TempDir);
  sys::path::append(memTopologyJsonFile, "mem_topology.json");
  {
    auto memTopologyJsonOut =
        openOutputFile(memTopologyJsonFile, &errorMessage);
    if (!memTopologyJsonOut) return moduleOp.emitOpError(errorMessage);

    std::string mem_topology_data = R"({
      "mem_topology": {
          "m_count": "2",
          "m_mem_data": [
              {
                  "m_type": "MEM_DRAM",
                  "m_used": "1",
                  "m_sizeKB": "0x10000",
                  "m_tag": "HOST",
                  "m_base_address": "0x4000000"
              },
              {
                  "m_type": "MEM_DRAM",
                  "m_used": "1",
                  "m_sizeKB": "0xc000",
                  "m_tag": "SRAM",
                  "m_base_address": "0x4000000"
              }
          ]
      }
    })";
    memTopologyJsonOut->os() << mem_topology_data;
    memTopologyJsonOut->keep();
  }

  // Create aie_partition.json.
  SmallString<64> aiePartitionJsonFile(TK.TempDir);
  sys::path::append(aiePartitionJsonFile, "aie_partition.json");
  {
    auto aiePartitionJsonOut =
        openOutputFile(aiePartitionJsonFile, &errorMessage);
    if (!aiePartitionJsonOut) return moduleOp.emitOpError(errorMessage);

    std::string uuid_str = getUUIDString();
    std::string aie_partition_json_data = R"(
      {
        "aie_partition": {
          "name": "QoS",
          "operations_per_cycle": "2048",
          "inference_fingerprint": "23423",
          "pre_post_fingerprint": "12345",
          "partition": {
            "column_width": 4,
            "start_columns": [
              1
            ]
          },
          "PDIs": [
            {
              "uuid": ")" + uuid_str + R"(",
              "file_name": "./design.pdi",
              "cdo_groups": [
                {
                  "name": "DPU",
                  "type": "PRIMARY",
                  "pdi_id": "0x01",
                  "dpu_kernel_ids": [
                    ")" + TK.XCLBinKernelID +
                                          R"("
                  ],
                  "pre_cdo_groups": [
                    "0xC1"
                  ]
                }
              ]
            }
          ]
        }
      }
    )";
    aiePartitionJsonOut->os() << aie_partition_json_data;
    aiePartitionJsonOut->keep();
  }

  // Create kernels.json.
  SmallString<64> kernelsJsonFile(TK.TempDir);
  sys::path::append(kernelsJsonFile, "kernels.json");
  {
    auto kernelsJsonOut = openOutputFile(kernelsJsonFile, &errorMessage);
    if (!kernelsJsonOut) return moduleOp.emitOpError(errorMessage);

    // TODO(max): should be gotten from the dispatch not this func (which will
    // eventually disappear)
    std::optional<int> numArgs;
    moduleOp.walk([&numArgs](func::FuncOp sequenceFunc) {
      if (sequenceFunc.getName() == "sequence")
        numArgs = sequenceFunc.getArgumentTypes().size();
    });
    if (!numArgs)
      return moduleOp.emitOpError(
          "Couldn't find func.func @sequence to count args");

    json::Object kernels_data{
        {"ps-kernels",
         json::Object{
             {"kernels",
              json::Array{// TODO: Support for multiple kernels
                          makeKernelJSON(TK.XCLBinKernelName, TK.XCLBinKernelID,
                                         TK.XCLBinInstanceName, *numArgs)}}}}};
    kernelsJsonOut->os() << formatv("{0:2}",
                                    json::Value(std::move(kernels_data)));
    kernelsJsonOut->keep();
  }
  // Create design.bif.
  SmallString<64> designBifFile(TK.TempDir);
  sys::path::append(designBifFile, "design.bif");
  {
    auto designBifOut = openOutputFile(designBifFile, &errorMessage);
    if (!designBifOut) return moduleOp.emitOpError(errorMessage);

    designBifOut->os() << "all:\n"
                       << "{\n"
                       << "  id_code = 0x14ca8093\n"
                       << "  extended_id_code = 0x01\n"
                       << "  image\n"
                       << "  {\n"
                       << "    name=aie_image, id=0x1c000000\n"
                       << "    { type=cdo\n"
                       << "      file=" << TK.TempDir << "/aie_cdo_elfs.bin\n"
                       << "      file=" << TK.TempDir << "/aie_cdo_init.bin\n"
                       << "      file=" << TK.TempDir << "/aie_cdo_enable.bin\n"
                       << "    }\n"
                       << "  }\n"
                       << "}";
    designBifOut->keep();
  }

  // Execute the bootgen command.
  SmallString<64> designPdiFile(TK.TempDir);
  sys::path::append(designPdiFile, "design.pdi");
  {
    SmallVector<std::string, 7> flags{"-arch",  "versal",
                                      "-image", std::string(designBifFile),
                                      "-o",     std::string(designPdiFile),
                                      "-w"};

    // use ./Xilinx/Vitis/2023.2/bin/bootgen for now (will link to lib soon)

    if (auto bootgen = sys::findProgramByName("bootgen")) {
      if (runTool(*bootgen, flags, TK.Verbose) != 0)
        return moduleOp.emitOpError(": failed to execute bootgen");
    } else {
      return moduleOp.emitOpError(": could not find bootgen");
    }
  }

  // Execute the xclbinutil command.
  {
    std::string memArg =
        "MEM_TOPOLOGY:JSON:" + std::string(memTopologyJsonFile);
    std::string partArg =
        "AIE_PARTITION:JSON:" + std::string(aiePartitionJsonFile);
    SmallVector<std::string, 20> flags{"--add-replace-section",
                                       memArg,
                                       "--add-kernel",
                                       std::string(kernelsJsonFile),
                                       "--add-replace-section",
                                       partArg,
                                       "--force",
                                       "--output",
                                       std::string(Output)};

    if (auto xclbinutil = sys::findProgramByName("xclbinutil")) {
      if (runTool(*xclbinutil, flags, TK.Verbose) != 0)
        return moduleOp.emitOpError(": failed to execute xclbinutil");
    } else {
      return moduleOp.emitOpError(": could not find xclbinutil");
    }
  }
  return success();
}

static std::string chesshack(const std::string &input) {
  std::string result(input);
  static const std::unordered_map<std::string, std::string> substitutions{
      {"memory\\(none\\)", "readnone"},
      {"memory\\(read\\)", "readonly"},
      {"memory\\(write\\)", "writeonly"},
      {"memory\\(argmem: readwrite\\)", "argmemonly"},
      {"memory\\(argmem: read\\)", "argmemonly readonly"},
      {"memory\\(argmem: write\\)", "argmemonly writeonly"},
      {"memory\\(inaccessiblemem: write\\)", "inaccessiblememonly writeonly"},
      {"memory\\(inaccessiblemem: readwrite\\)", "inaccessiblememonly"},
      {"memory\\(inaccessiblemem: read\\)", "inaccessiblememonly readonly"},
      {"memory(argmem: readwrite, inaccessiblemem: readwrite)",
       "inaccessiblemem_or_argmemonly"},
      {"memory(argmem: read, inaccessiblemem: read)",
       "inaccessiblemem_or_argmemonly readonly"},
      {"memory(argmem: write, inaccessiblemem: write)",
       "inaccessiblemem_or_argmemonly writeonly"},
  };
  for (const auto &pair : substitutions)
    result = std::regex_replace(result, std::regex(pair.first), pair.second);
  return result;
}

// A pass which removes the alignment attribute from llvm load operations, if
// the alignment is less than 4 (2 or 1).
//
// Example replaces:
//
// ```
//  %113 = llvm.load %112 {alignment = 2 : i64} : !llvm.ptr -> vector<32xbf16>
// ```
//
// with
//
// ```
//  %113 = llvm.load %112 : !llvm.ptr -> vector<32xbf16>
// ```
//
// If this pass is not included in the pipeline, there is an alignment error
// later in the compilation. This is a temporary workaround while a better
// solution is found: propagation of memref.assume_alignment is one option. See
// also https://jira.xilinx.com/projects/AIECC/issues/AIECC-589
namespace {
struct RemoveAlignment2FromLLVMLoadPass
    : public PassWrapper<RemoveAlignment2FromLLVMLoadPass,
                         OperationPass<ModuleOp>> {
  void runOnOperation() override {
    getOperation().walk([](Operation *op) {
      if (auto loadOp = dyn_cast<LLVM::LoadOp>(op)) {
        auto alignmentAttr = loadOp.getAlignmentAttr();
        if (alignmentAttr) {
          int alignmentVal = alignmentAttr.getValue().getSExtValue();
          if (alignmentVal == 2 || alignmentVal == 1) {
            loadOp.setAlignment(std::optional<uint64_t>());
          }
        }
      }
    });
  }

 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      RemoveAlignment2FromLLVMLoadPass);
};
}  // namespace

static LogicalResult generateObject(MLIRContext *context, ModuleOp moduleOp,
                                    XCLBinGenConfig &TK,
                                    const std::string &outputFile) {
  PassManager pm(context, moduleOp.getOperationName());
  applyConfigToPassManager(TK, pm);

  pm.addNestedPass<AIE::DeviceOp>(AIE::createAIELocalizeLocksPass());
  pm.addPass(AIE::createAIECoreToStandardPass());
  pm.addPass(AIEX::createAIEXToStandardPass());

  // Convert specific vector dialect ops (like vector.contract) to the AIEVec
  // dialect
  {
    xilinx::aievec::ConvertVectorToAIEVecOptions vectorToAIEVecOptions{};

    std::string optionsString = [&]() {
      std::ostringstream optionsStringStream;
      optionsStringStream << "target-backend=";
      optionsStringStream << (TK.UseChess ? "cpp" : "llvmir");
      optionsStringStream << ' ' << "aie-target=aieml";
      return optionsStringStream.str();
    }();

    if (failed(vectorToAIEVecOptions.parseFromString(optionsString))) {
      return moduleOp.emitOpError(": Failed to parse options from '")
             << optionsString
             << "': Failed to construct ConvertVectorToAIEVecOptions.";
    }
    xilinx::aievec::buildConvertVectorToAIEVec(pm, vectorToAIEVecOptions);
  }

  addLowerToLLVMPasses(pm);
  pm.addPass(std::make_unique<RemoveAlignment2FromLLVMLoadPass>());

  if (TK.Verbose) {
    llvm::outs() << "Running: ";
    pm.printAsTextualPipeline(llvm::outs());
    llvm::outs() << "\n";
  }

  ModuleOp copy = moduleOp.clone();
  if (failed(pm.run(copy)))
    return moduleOp.emitOpError(": Failed to lower to LLVM");

  std::string llvmDialectString;
  raw_string_ostream llvmDialectStream(llvmDialectString);
  copy.print(llvmDialectStream);
  std::optional<std::string> llvmirString;
  if (llvmirString = xilinx::AIE::AIETranslateModuleToLLVMIR(llvmDialectString);
      !llvmirString.has_value())
    return moduleOp.emitOpError(": couldn't translate llvm dialect to llvm ir");
  std::string chessWrapper(_CHESS_INTRINSIC_WRAPPER_LL);
  SmallString<64> chesslinkedFile(TK.TempDir);
  sys::path::append(chesslinkedFile, "input.chesslinked.ll");
  if (auto chesslinked =
          xilinx::AIE::AIELLVMLink({*llvmirString, chessWrapper});
      chesslinked.has_value()) {
    std::string errorMessage;
    auto chesslinkedOut = openOutputFile(chesslinkedFile, &errorMessage);
    if (!chesslinkedOut) return moduleOp.emitOpError(errorMessage);
    chesslinkedOut->os() << *chesslinked;
    chesslinkedOut->keep();
  } else
    return moduleOp.emitOpError(": couldn't llvm-link chess wrapper");

  SmallString<64> chessExe(TK.AIEToolsDir);
  sys::path::append(chessExe, "bin", "unwrapped", "lnx64.o", "xchesscc");
  SmallString<64> chessworkDir(TK.TempDir);
  sys::path::append(chessworkDir, "chesswork");
  auto chessArgs_ = chessArgs(TK.AIEToolsDir, chessworkDir.str().str());
  chessArgs_.push_back("-c");
  chessArgs_.push_back(std::string(chesslinkedFile));
  chessArgs_.push_back("-o");
  chessArgs_.push_back(std::string(outputFile));
  if (!sys::fs::exists(chessExe))
    return moduleOp.emitOpError(": chess can't be found");

  if (runTool(chessExe, chessArgs_, TK.Verbose) != 0)
    return moduleOp.emitOpError(": Failed to assemble with chess");
  copy->erase();
  return success();
}

LogicalResult xilinx::aie2xclbin(MLIRContext *ctx, ModuleOp moduleOp,
                                 XCLBinGenConfig &TK, StringRef OutputNPU,
                                 StringRef OutputXCLBin) {
  if (failed(xilinx::findVitis(TK)))
    return moduleOp.emitOpError(": VITIS not found");

  TK.TargetArch = StringRef(TK.TargetArch).trim();

  std::regex target_regex("AIE.?");
  if (!std::regex_search(TK.TargetArch, target_regex))
    return moduleOp.emitOpError()
           << "Unexpected target architecture: " << TK.TargetArch;

  // generateNPUInstructions
  {
    PassManager pm(ctx, moduleOp.getOperationName());
    applyConfigToPassManager(TK, pm);

    pm.addNestedPass<AIE::DeviceOp>(AIEX::createAIEDmaToNpuPass());
    ModuleOp copy = moduleOp.clone();
    if (failed(pm.run(copy)))
      return moduleOp.emitOpError(": NPU Instruction pipeline failed");

    std::string errorMessage;
    auto output = openOutputFile(OutputNPU, &errorMessage);
    if (!output) return moduleOp.emitOpError(errorMessage);

    if (failed(AIE::AIETranslateToNPU(copy, output->os())))
      return moduleOp.emitOpError(": NPU Instruction translation failed");

    output->keep();
    copy->erase();
  }

  SmallString<64> object(TK.TempDir);
  sys::path::append(object, "input.o");
  if (failed(generateObject(ctx, moduleOp, TK, std::string(object))))
    return moduleOp.emitOpError(": Failed to generate object");

  if (failed(generateCoreElfFiles(moduleOp, object, TK)))
    return moduleOp.emitOpError(": Failed to generate core ELF file(s)");

  if (failed(generateCDO(ctx, moduleOp, TK)))
    return moduleOp.emitOpError(": Failed to generate CDO");

  if (failed(generateXCLBin(ctx, moduleOp, TK, OutputXCLBin)))
    return moduleOp.emitOpError(": Failed to generate XCLBin");

  return success();
}
