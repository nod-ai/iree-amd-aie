// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "XCLBinGen.h"

#include <fstream>
#include <regex>
#include <sstream>
#include <unordered_map>

#include "AMDAIETargets.h"
#include "aie/Targets/AIETargets.h"
#include "aievec/Passes.h"
#include "iree-amd-aie/Transforms/Passes.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/ToolOutputFile.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Target/LLVMIR/Export.h"

#define DEBUG_TYPE "amdaie-xclbingen"

#ifdef _WIN32
#include "windows.h"
// For UUID stuff
#include "rpcdce.h"

#define setenv(name, var, ignore) _putenv_s(name, var)
#else
#include <uuid/uuid.h>
#endif

// This is a string that contains the wrapped chess intrinsics (see top of the
// included file for deeper explanation).
static const std::string _CHESS_INTRINSIC_WRAPPER_CPP{
#include "chess_intrinsic_wrapper.cpp"
};

using namespace llvm;
using namespace mlir;
using namespace xilinx;

namespace {

// Apply the pass manager specific options of the XCLBinGenConfig to the pass
// manager. These control when (if ever) and what IR gets printed between
// passes, and whether the pass manager uses multi-theading.
void applyConfigToPassManager(PassManager &pm, bool printIRBeforeAll,
                              bool printIRAfterAll, bool printIRModuleScope,
                              bool timing) {
  auto shouldPrintBeforePass = [printIRBeforeAll](Pass *, Operation *) {
    return printIRBeforeAll;
  };

  auto shouldPrintAfterPass = [printIRAfterAll](Pass *, Operation *) {
    return printIRAfterAll;
  };

  pm.enableIRPrinting(shouldPrintBeforePass, shouldPrintAfterPass,
                      printIRModuleScope);

  if (timing) pm.enableTiming();
}
}  // namespace

FailureOr<std::string> findVitis(std::optional<std::string> &vitisDir) {
  if (!vitisDir) {
    const char *envVitis = ::getenv("VITIS");
    if (!envVitis) {
      if (auto vpp = sys::findProgramByName("v++")) {
        SmallString<64> realVpp;
        std::error_code err = sys::fs::real_path(vpp.get(), realVpp);
        if (!err) {
          sys::path::remove_filename(realVpp);
          sys::path::remove_filename(realVpp);
          vitisDir = realVpp.str().str();
          LLVM_DEBUG(dbgs() << "Found Vitis at " << realVpp.c_str() << "\n");
        }
      }
    }
  }
  if (!vitisDir) {
    llvm::errs() << "ERROR: couldn't find vitis directory\n";
    return failure();
  };

  SmallString<64> aieToolsPath(*vitisDir);
  sys::path::append(aieToolsPath, "aietools");
  if (!sys::fs::exists(aieToolsPath)) {
    llvm::errs() << "ERROR: couldn't find aietools directory\n";
    return failure();
  }

  SmallString<64> chessccPath(aieToolsPath);
  sys::path::append(chessccPath, "tps", "lnx64", "target_aie_ml");
  sys::path::append(chessccPath, "bin", "LNa64bin");

  SmallString<64> path(::getenv("PATH"));
  ::setenv(
      "PATH",
      (chessccPath + std::string{sys::EnvPathSeparator} + path).str().c_str(),
      1);

  SmallString<64> chessClangExe(chessccPath);
  sys::path::append(chessClangExe, "chess-clang");
  SmallString<64> chessLLVMLinkExe(chessccPath);
  sys::path::append(chessLLVMLinkExe, "chess-llvm-link");
  if (!sys::fs::exists(chessClangExe)) {
    llvm::errs() << "ERROR: couldn't find chess-clang\n";
    return failure();
  }
  if (!sys::fs::exists(chessLLVMLinkExe)) {
    llvm::errs() << "ERROR: couldn't find chess-llvm-link\n";
    return failure();
  }

  SmallString<64> lnx64o(aieToolsPath);
  sys::path::append(lnx64o, "lib", "lnx64.o");
  SmallString<64> dotLib(aieToolsPath);
  sys::path::append(dotLib, "lnx64", "tools", "dot", "lib");
  SmallString<64> ldLibraryPath(::getenv("LD_LIBRARY_PATH"));
  ::setenv("LD_LIBRARY_PATH",
           (lnx64o + std::string{sys::EnvPathSeparator} + dotLib +
            std::string{sys::EnvPathSeparator} + ldLibraryPath)
               .str()
               .c_str(),
           1);

  SmallString<64> rdiDataDir(aieToolsPath);
  sys::path::append(rdiDataDir, "data");
  ::setenv("RDI_DATADIR", rdiDataDir.c_str(), 1);

  return *vitisDir;
}

std::pair<std::string, std::vector<std::string>> makeChessArgs(
    const std::string &vitisDir, const std::string &tempDir, bool verbose) {
  SmallString<64> aieToolsDir(vitisDir);
  sys::path::append(aieToolsDir, "aietools");
  SmallString<64> chessworkDir(tempDir);
  sys::path::append(chessworkDir, "chesswork");
  SmallString<64> xChessCCExe(aieToolsDir);
  sys::path::append(xChessCCExe, "bin", "unwrapped", "lnx64.o", "xchesscc");

  SmallString<64> procModelLib(aieToolsDir);
  sys::path::append(procModelLib, "data", "aie_ml", "lib");

  std::vector<std::string> flags{
      // -j <threads> : parallel compilation (function + file level)
      "-j4",
      // -p <name> : processor
      "-pme",
      // -P <dir> : processor model directory
      "-P" + procModelLib.str().str(),
      // -f : use LLVM frontend (chess-clang)
      "-f",
      // -C <cfg> : configuration (for chess-clang)
      "-CRelease_LLVM",
      // +w <dir> : work directory
      "+w" + tempDir,
      // for adf headers
      "-D__AIENGINE__",
      // for aie_api headers
      "-D__AIE_ARCH__=20",
  };
  // disassemble output
  if (verbose) flags.emplace_back("-d");
  return {xChessCCExe.str().str(), flags};
}

std::optional<std::string> dumpStrToDisk(const std::string &payload,
                                         const std::string &outputPath) {
  std::string errorMessage;
  std::unique_ptr<llvm::ToolOutputFile> outputFile =
      openOutputFile(outputPath, &errorMessage);
  if (!outputFile) return errorMessage;
  outputFile->os() << payload;
  outputFile->keep();
  return {};
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

// Returns either:
//  -- the output of running the tool, if run without failure, or
//  -- an empty optional, if the tool fails to run.
std::optional<std::string> runTool(
    StringRef program, ArrayRef<std::string> args, bool verbose,
    std::optional<ArrayRef<StringRef>> env = std::nullopt) {
  if (verbose) {
    llvm::outs() << "Run: ";
    if (env)
      for (auto &s : *env) llvm::outs() << " " << s;
    llvm::outs() << " " << program;
    for (auto &s : args) llvm::outs() << " " << s;
    llvm::outs() << "\n";
  }

  // Check that 'program' is a valid path, if not, fail immediately.
  if (!sys::fs::exists(program)) {
    llvm::errs() << "Program " << program << " does not exist\n";
    return {};
  }

  // Run the program, piping any output to a temporary file (we only want to
  // print to terminal if verbose is true).
  std::string errMsg;
  sys::ProcessStatistics stats;
  std::optional<sys::ProcessStatistics> optStats(stats);
  SmallVector<StringRef, 8> pArgs = {program};
  pArgs.append(args.begin(), args.end());
  SmallVector<char> temporaryPath;
  {
    std::string prefix{"tmpRunTool"};
    std::string suffix{"Logging"};
    auto errorCode =
        llvm::sys::fs::createTemporaryFile(prefix, suffix, temporaryPath);
    if (errorCode) {
      llvm::errs() << "Failed to create temporary file: " << errorCode.message()
                   << "\n";
      return {};
    }
  }

  std::string temporaryPathStr =
      std::string(temporaryPath.begin(), temporaryPath.size());
  StringRef temporaryPathRef(temporaryPathStr);
  auto tp = std::optional<StringRef>(temporaryPathRef);
  int result =
      sys::ExecuteAndWait(program, pArgs, env, /* redirects */ {tp, tp, tp}, 0,
                          0, &errMsg, nullptr, &optStats);

  auto maybeOutputFromFile = [&]() -> std::optional<std::string> {
    std::ifstream t(temporaryPathRef.str());
    std::stringstream buffer;
    if (t.is_open() && t.good()) {
      buffer << t.rdbuf();
      return buffer.str();
    }
    return nullptr;
  }();

  if (!maybeOutputFromFile) {
    llvm::errs() << "Failed to open temporary file " << temporaryPathRef.str()
                 << "\n";
    return {};
  }
  auto outputFromFile = maybeOutputFromFile.value();

  if (verbose) {
    auto totalTime = std::chrono::duration_cast<std::chrono::duration<float>>(
                         stats.TotalTime)
                         .count();
    std::string exitStatusStr = result == 0 ? "Succeeded" : "Failed";
    llvm::outs() << exitStatusStr << " in totalTime " << totalTime
                 << " [s]. Exit code=" << result << "\n";
    llvm::outs() << outputFromFile << "\n";
  }

  if (result != 0) {
    llvm::errs() << "Failed to run tool: " << program << ". Error: '" << errMsg
                 << "'\n"
                 << outputFromFile;
    return {};
  }

  return outputFromFile;
}

LogicalResult assembleUsingChess(const std::string &inputFile,
                                 const std::string &outputFile,
                                 const std::string &vitisDir,
                                 const std::string &tempDir, bool verbose) {
  auto [xChessCCExe, chessArgs] = makeChessArgs(vitisDir, tempDir, verbose);
  chessArgs.push_back("-c");
  chessArgs.push_back(inputFile);
  chessArgs.push_back("-o");
  chessArgs.push_back(outputFile);
  if (!runTool(xChessCCExe, chessArgs, verbose)) {
    llvm::errs() << "Failed to assemble with chess";
    return failure();
  }
  return success();
}

bool useMeBasic(const std::string &peanoDir, bool verbose) {
  if (verbose)
    llvm::outs() << "Checking if we should use me_basic, based on "
                    "the version of peano\n";
  SmallString<64> peanoOptBin(peanoDir);
  sys::path::append(peanoOptBin, "bin", "opt");
  auto maybeVersion = runTool(peanoOptBin, {"--version"}, verbose);
  // default to "yes do use"
  if (!maybeVersion) return true;
  auto version = maybeVersion.value();
  std::regex r("LLVM version 17.0.0git", std::regex_constants::multiline);
  return std::regex_search(version, r);
}

// Generate the elf files for the core
static LogicalResult generateCoreElfFiles(
    ModuleOp moduleOp, const StringRef objFile, const std::string &tempDir,
    bool useChess, std::optional<std::string> &vitisDir,
    const std::string &targetArch, bool verbose, const std::string &peanoDir) {
  auto deviceOps = moduleOp.getOps<AIE::DeviceOp>();
  if (!llvm::hasSingleElement(deviceOps))
    return moduleOp.emitOpError("expected a single device op");

  AIE::DeviceOp deviceOp = *deviceOps.begin();
  auto tileOps = deviceOp.getOps<AIE::TileOp>();

  std::string errorMessage;

  const bool doUseMeBasic = !useChess && useMeBasic(peanoDir, verbose);

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

    SmallString<64> elfFile(tempDir);
    sys::path::append(elfFile, elfFileName);

    if (useChess) {
      SmallString<64> chessIntrinsicsCPPFile(tempDir);
      sys::path::append(chessIntrinsicsCPPFile, "chess_intrinsic_wrapper.cpp");
      if (auto maybeErr = dumpStrToDisk(_CHESS_INTRINSIC_WRAPPER_CPP,
                                        chessIntrinsicsCPPFile.str().str());
          maybeErr.has_value()) {
        llvm::errs()
            << "Failed to dump to disk chess_intrinsic_wrapper.cpp because: "
            << maybeErr;
        return failure();
      }

      SmallString<64> chessIntrinsicsObjFile(tempDir);
      sys::path::append(chessIntrinsicsObjFile, "chess_intrinsic_wrapper.o");
      auto maybeVitisDir = findVitis(vitisDir);
      if (failed(maybeVitisDir)) return failure();

      if (failed(assembleUsingChess(chessIntrinsicsCPPFile.c_str(),
                                    chessIntrinsicsObjFile.c_str(), *vitisDir,
                                    tempDir, verbose))) {
        llvm::errs() << "Failed to assemble chess_intrinsic_wrapper.o";
        return failure();
      }

      // Use xbridge (to remove any peano dependency with use-chess option)
      SmallString<64> bcfPath(tempDir);
      sys::path::append(bcfPath, elfFileName + ".bcf");

      {
        auto bcfOutput = openOutputFile(bcfPath, &errorMessage);
        if (!bcfOutput) {
          llvm::errs() << "failed to open bcf file because: " << errorMessage;
          return failure();
        }

        if (failed(mlir::iree_compiler::AMDAIE::AIETranslateToBCF(
                moduleOp, bcfOutput->os(), col, row))) {
          llvm::errs() << "Failed to generate BCF";
          return failure();
        }
        bcfOutput->os() << "_include _file chess_intrinsic_wrapper.o\n";
        bcfOutput->keep();
      }

      std::vector<std::string> extractedIncludes{
          chessIntrinsicsObjFile.c_str()};
      {
        auto bcfFileIn = openInputFile(bcfPath, &errorMessage);
        if (!bcfFileIn) {
          llvm::errs() << "failed to open bcf because: " << errorMessage;
          return failure();
        }

        std::string bcfFile = std::string(bcfFileIn->getBuffer());
        std::regex r("_include _file (.*)");
        auto begin = std::sregex_iterator(bcfFile.begin(), bcfFile.end(), r);
        auto end = std::sregex_iterator();
        for (std::sregex_iterator i = begin; i != end; ++i) {
          if (i->str(1) == "chess_intrinsic_wrapper.o") continue;
          extractedIncludes.push_back(i->str(1));
        }
      }

      auto [xChessCCExe, chessArgs] =
          makeChessArgs(*vitisDir, tempDir, verbose);
      chessArgs.emplace_back(objFile);
      for (const auto &inc : extractedIncludes) chessArgs.emplace_back(inc);
      chessArgs.push_back("+l");
      chessArgs.push_back(bcfPath.c_str());
      chessArgs.push_back("-o");
      chessArgs.push_back(elfFile.c_str());
      if (!runTool(xChessCCExe, chessArgs, verbose)) {
        llvm::errs() << "Failed to link with xbridge";
        return failure();
      }
    } else {
      SmallString<64> ldscriptPath(tempDir);
      sys::path::append(ldscriptPath, elfFileName + ".ld");
      {
        auto ldscriptOutput = openOutputFile(ldscriptPath, &errorMessage);
        if (!ldscriptOutput) {
          llvm::errs() << "Failed to open ldscript file because: "
                       << errorMessage;
          return failure();
        }
        if (failed(mlir::iree_compiler::AMDAIE::AIETranslateToLdScript(
                moduleOp, ldscriptOutput->os(), col, row))) {
          llvm::errs() << "failed to generate ld script for core (" << col
                       << "," << row << ")";
          return failure();
        }
        ldscriptOutput->keep();
      }

      // We are running a clang command for now, but really this is an lld
      // command.
      {
        SmallString<64> peanoClangExe(peanoDir);
        sys::path::append(peanoClangExe, "bin", "clang");

        std::string targetLower = StringRef(targetArch).lower();
        SmallVector<std::string, 10> flags;
        flags.push_back("-O2");
        std::string targetFlag = "--target=" + targetLower + "-none-elf";
        flags.push_back(targetFlag);

        if (doUseMeBasic) {
          SmallString<64> meBasicPath(peanoDir);
          sys::path::append(meBasicPath, "lib",
                            targetLower + "-none-unknown-elf", "me_basic.o");
          flags.emplace_back(meBasicPath);
        }

        flags.emplace_back(objFile);
        SmallString<64> libcPath(peanoDir);
        sys::path::append(libcPath, "lib", targetLower + "-none-unknown-elf",
                          "libc.a");
        flags.emplace_back(libcPath);
        flags.push_back("-Wl,--gc-sections");
        std::string ldScriptFlag = "-Wl,-T," + std::string(ldscriptPath);
        flags.push_back(ldScriptFlag);
        flags.push_back("-o");
        flags.emplace_back(elfFile);
        if (!runTool(peanoClangExe, flags, verbose)) {
          llvm::errs() << "failed to link elf file for core(" << col << ","
                       << row << ")";
          return failure();
        }
      }
    }
  }
  return success();
}

static LogicalResult generateCDO(MLIRContext *context, ModuleOp moduleOp,
                                 bool printIRBeforeAll, bool printIRAfterAll,
                                 bool printIRModuleScope, bool timing,
                                 const std::string &tempDir) {
  ModuleOp copy = moduleOp.clone();
  std::string errorMessage;
  PassManager passManager(context, ModuleOp::getOperationName());
  applyConfigToPassManager(passManager, printIRBeforeAll, printIRAfterAll,
                           printIRModuleScope, timing);
  passManager.addNestedPass<AIE::DeviceOp>(
      mlir::iree_compiler::AMDAIE::createAMDAIEPathfinderPass());
  if (failed(passManager.run(copy))) {
    llvm::errs() << "failed to run passes to prepare for XCLBin generation";
    return failure();
  }

  if (failed(mlir::iree_compiler::AMDAIE::AIETranslateToCDODirect(copy,
                                                                  tempDir))) {
    llvm::errs() << "failed to emit CDO";
    return failure();
  }

  copy->erase();
  return success();
}

static json::Object makeKernelJSON(const std::string &name,
                                   const std::string &id,
                                   const std::string &instance) {
  return json::Object{
      {"name", name},
      {"type", "dpu"},
      {"extended-data",
       json::Object{
           {"subtype", "DPU"}, {"functional", "0"}, {"dpu_kernel_id", id}}},
      {"arguments", json::Array{json::Object{{"name", "opcode"},
                                             {"address-qualifier", "SCALAR"},
                                             {"type", "uint64_t"},
                                             {"offset", "0x00"}},
                                json::Object{{"name", "instr"},
                                             {"memory-connection", "SRAM"},
                                             {"address-qualifier", "GLOBAL"},
                                             {"type", "char *"},
                                             {"offset", "0x08"}},
                                json::Object{{"name", "ninstr"},
                                             {"address-qualifier", "SCALAR"},
                                             {"type", "uint32_t"},
                                             {"offset", "0x10"}},
                                json::Object{{"name", "bo0"},
                                             {"memory-connection", "HOST"},
                                             {"address-qualifier", "GLOBAL"},
                                             {"type", "void*"},
                                             {"offset", "0x14"}},
                                json::Object{{"name", "bo1"},
                                             {"memory-connection", "HOST"},
                                             {"address-qualifier", "GLOBAL"},
                                             {"type", "void*"},
                                             {"offset", "0x1c"}},
                                json::Object{{"name", "bo2"},
                                             {"memory-connection", "HOST"},
                                             {"address-qualifier", "GLOBAL"},
                                             {"type", "void*"},
                                             {"offset", "0x24"}},
                                json::Object{{"name", "bo3"},
                                             {"memory-connection", "HOST"},
                                             {"address-qualifier", "GLOBAL"},
                                             {"type", "void*"},
                                             {"offset", "0x2c"}},
                                json::Object{{"name", "bo4"},
                                             {"memory-connection", "HOST"},
                                             {"address-qualifier", "GLOBAL"},
                                             {"type", "void*"},
                                             {"offset", "0x34"}},
                                json::Object{{"name", "bo5"},
                                             {"memory-connection", "HOST"},
                                             {"address-qualifier", "GLOBAL"},
                                             {"type", "void*"},
                                             {"offset", "0x3c"}}}},
      {"instances", json::Array{json::Object{{"name", instance}}}}};
}

static LogicalResult generateXCLBin(
    ModuleOp moduleOp, const std::string &Output, const std::string &tempDir,
    const std::string &xclBinKernelID, const std::string &xclBinKernelName,
    const std::string &xclBinInstanceName, const std::string &amdAIEInstallDir,
    bool verbose, const std::string &inputXclbin = "") {
  std::string errorMessage;
  // Create mem_topology.json.
  SmallString<64> memTopologyJsonFile(tempDir);
  sys::path::append(memTopologyJsonFile, "mem_topology.json");
  {
    std::string memTopologyData = R"({
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
    if (auto maybeErr =
            dumpStrToDisk(memTopologyData, memTopologyJsonFile.str().str());
        maybeErr.has_value()) {
      llvm::errs() << "failed to dump to disk mem_topology.json because: "
                   << *maybeErr;
      return failure();
    }
  }

  // Create aie_partition.json.
  SmallString<64> aiePartitionJsonFile(tempDir);
  sys::path::append(aiePartitionJsonFile, "aie_partition.json");
  {
    std::string uuidStr = getUUIDString();
    std::string aiePartitionJsonData = R"(
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
              "uuid": ")" + uuidStr + R"(",
              "file_name": "./design.pdi",
              "cdo_groups": [
                {
                  "name": "DPU",
                  "type": "PRIMARY",
                  "pdi_id": "0x01",
                  "dpu_kernel_ids": [
                    ")" + xclBinKernelID +
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
    if (auto maybeErr = dumpStrToDisk(aiePartitionJsonData,
                                      aiePartitionJsonFile.str().str());
        maybeErr.has_value()) {
      llvm::errs() << "failed to dump to disk aie_partition.json because: "
                   << *maybeErr;
      return failure();
    }
  }

  // Create kernels.json.
  SmallString<64> kernelsJsonFile(tempDir);
  sys::path::append(kernelsJsonFile, "kernels.json");
  {
    // TODO: Support for multiple kernels
    json::Object kernels_data{
        {"ps-kernels",
         json::Object{{"kernels", json::Array{makeKernelJSON(
                                      xclBinKernelName, xclBinKernelID,
                                      xclBinInstanceName)}}}}};

    auto kernelStr =
        llvm::formatv("{0:2}", json::Value(std::move(kernels_data)));
    if (auto maybeErr = dumpStrToDisk(kernelStr, kernelsJsonFile.str().str());
        maybeErr.has_value()) {
      llvm::errs() << "failed to dump to disk kernels.json because: "
                   << *maybeErr;
      return failure();
    }
  }
  // Create design.bif.
  SmallString<64> designBifFile(tempDir);
  sys::path::append(designBifFile, "design.bif");
  {
    auto designBifOut = openOutputFile(designBifFile, &errorMessage);
    if (!designBifOut) {
      llvm::errs() << "failed to open design.bif because: " << errorMessage;
      return failure();
    }

    designBifOut->os() << "all:\n"
                       << "{\n"
                       << "  id_code = 0x14ca8093\n"
                       << "  extended_id_code = 0x01\n"
                       << "  image\n"
                       << "  {\n"
                       << "    name=aie_image, id=0x1c000000\n"
                       << "    { type=cdo\n"
                       << "      file=" << tempDir << "/aie_cdo_elfs.bin\n"
                       << "      file=" << tempDir << "/aie_cdo_init.bin\n"
                       << "      file=" << tempDir << "/aie_cdo_enable.bin\n"
                       << "    }\n"
                       << "  }\n"
                       << "}";
    designBifOut->keep();
  }

  // Execute the bootgen command.
  SmallString<64> designPdiFile(tempDir);
  sys::path::append(designPdiFile, "design.pdi");
  {
    SmallVector<std::string, 7> flags{"-arch",  "versal",
                                      "-image", std::string(designBifFile),
                                      "-o",     std::string(designPdiFile),
                                      "-w"};

    SmallString<64> bootgenBin(amdAIEInstallDir);
    sys::path::append(bootgenBin, "bin", "amdaie_bootgen");
    if (!sys::fs::exists(bootgenBin)) {
      bootgenBin = amdAIEInstallDir;
      sys::path::append(bootgenBin, "tools", "amdaie_bootgen");
    }
    if (!runTool(bootgenBin, flags, verbose)) {
      llvm::errs() << "failed to execute bootgen";
      return failure();
    }
  }
  SmallVector<std::string, 20> flags;
  // Execute the xclbinutil command.
  std::string memArg = "MEM_TOPOLOGY:JSON:" + std::string(memTopologyJsonFile);
  std::string partArg =
      "AIE_PARTITION:JSON:" + std::string(aiePartitionJsonFile);
  SmallString<64> xclbinutilBin(amdAIEInstallDir);
  sys::path::append(xclbinutilBin, "bin", "amdaie_xclbinutil");
  if (!sys::fs::exists(xclbinutilBin)) {
    xclbinutilBin = amdAIEInstallDir;
    sys::path::append(xclbinutilBin, "tools", "amdaie_xclbinutil");
  }
  {
    if (!inputXclbin.empty()) {
      // Create aie_partition.json.
      SmallString<64> aieInputPartitionJsonFile(tempDir);
      sys::path::append(aieInputPartitionJsonFile, "aie_input_partition.json");

      std::string inputPartArg =
          "AIE_PARTITION:JSON:" + std::string(aieInputPartitionJsonFile);
      SmallVector<std::string, 20> inputFlags{"--dump-section", inputPartArg,
                                              "--force", "--input",
                                              std::string(inputXclbin)};

      if (!runTool(xclbinutilBin, inputFlags, verbose)) {
        llvm::errs() << "failed to execute xclbinutil";
        return failure();
      }
      auto aieInputPartitionOut =
          openInputFile(aieInputPartitionJsonFile, &errorMessage);
      if (!aieInputPartitionOut) {
        llvm::errs() << "failed to open aie_input_partition.json because: "
                     << errorMessage;
        return failure();
      }
      Expected<json::Value> aieInputPartitionOutValue =
          llvm::json::parse(aieInputPartitionOut->getBuffer());
      json::Array *aieInputPartionPDIs;
      aieInputPartionPDIs = aieInputPartitionOutValue->getAsObject()
                                ->getObject("aie_partition")
                                ->getArray("PDIs");
      auto aiePartitionOut = openInputFile(aiePartitionJsonFile, &errorMessage);
      if (!aiePartitionOut) {
        llvm::errs() << "failed to open aie aie_input_partition.json for "
                        "output because: "
                     << errorMessage;
        return failure();
      }
      llvm::Expected<llvm::json::Value> aiePartitionOutValue =
          llvm::json::parse(aiePartitionOut->getBuffer());
      json::Array *aiePartionPDIs;
      aiePartionPDIs = aiePartitionOutValue->getAsObject()
                           ->getObject("aie_partition")
                           ->getArray("PDIs");
      aieInputPartionPDIs->insert(aieInputPartionPDIs->end(),
                                  aiePartionPDIs->begin(),
                                  aiePartionPDIs->end());
      // rewrite aie partion json file
      if (auto maybeErr =
              dumpStrToDisk(formatv("{0:2}", *aieInputPartitionOutValue),
                            aiePartitionJsonFile.str().str());
          maybeErr.has_value()) {
        llvm::errs()
            << "failed to dump to disk aie_input_partition.json because: "
            << errorMessage;
        return failure();
      }
      flags.insert(flags.end(), {"--input", std::string(inputXclbin)});
    } else {
      flags.insert(flags.end(), {"--add-replace-section", memArg});
    }
    flags.insert(flags.end(), {"--add-kernel", std::string(kernelsJsonFile),
                               "--add-replace-section", partArg, "--force",
                               "--output", std::string(Output)});

    if (!runTool(xclbinutilBin, flags, verbose)) {
      llvm::errs() << "failed to execute xclbinutil";
      return failure();
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

static LogicalResult generateUnifiedObject(
    MLIRContext *context, ModuleOp moduleOp, const std::string &outputFile,
    bool printIRBeforeAll, bool printIRAfterAll, bool printIRModuleScope,
    bool timing, bool useChess, bool verbose, const std::string &tempDir,
    std::optional<std::string> &vitisDir, const std::string &targetArch,
    const std::string &peanoDir) {
  PassManager pm(context, moduleOp.getOperationName());
  applyConfigToPassManager(pm, printIRBeforeAll, printIRAfterAll,
                           printIRModuleScope, timing);

  pm.addPass(mlir::iree_compiler::AMDAIE::createAMDAIECoreToStandardPass());
  pm.addPass(mlir::iree_compiler::AMDAIE::createAMDAIEXToStandardPass());
  // Convert specific vector dialect ops (like vector.contract) to the AIEVec
  // dialect
  mlir::iree_compiler::aievec::buildConvertVectorToAIEVec(pm);
  mlir::iree_compiler::AMDAIE::addLowerToLLVMPasses(pm);
  pm.addPass(std::make_unique<RemoveAlignment2FromLLVMLoadPass>());

  if (verbose) {
    llvm::outs() << "Running: ";
    pm.printAsTextualPipeline(llvm::outs());
    llvm::outs() << "\n";
  }

  ModuleOp copy = moduleOp.clone();
  if (failed(pm.run(copy)))
    return moduleOp.emitOpError("Failed to lower to LLVM");

  SmallString<64> LLVMIRFile(tempDir);
  sys::path::append(LLVMIRFile, "input.ll");

  llvm::LLVMContext llvmContext;
  auto llvmModule = translateModuleToLLVMIR(copy, llvmContext);
  if (!llvmModule)
    return moduleOp.emitOpError("Failed to translate module to LLVMIR");

  std::string errorMessage;
  {
    auto output = openOutputFile(LLVMIRFile, &errorMessage);
    if (!output) {
      llvm::errs() << "failed to open output LLVMIRFile because: "
                   << errorMessage;
      return failure();
    }
    llvmModule->print(output->os(), nullptr);
    output->keep();
  }

  if (useChess) {
    SmallString<64> inputLLChessHackedFile(tempDir);
    sys::path::append(inputLLChessHackedFile, "input.chesshacked.ll");
    {
      std::string inputLLStr;
      auto inputLL = openInputFile(LLVMIRFile, &errorMessage);
      if (!inputLL) {
        llvm::errs() << "Failed to open input.ll file because: "
                     << errorMessage;
        return failure();
      }
      inputLLStr = std::string(inputLL->getBuffer());
      std::string inputLLChessHackedStr = chesshack(inputLLStr);
      if (auto maybeErr = dumpStrToDisk(inputLLChessHackedStr,
                                        inputLLChessHackedFile.str().str());
          maybeErr.has_value()) {
        llvm::errs() << "Failed to dump to disk input.chesshacked.ll because: "
                     << maybeErr;
        return failure();
      }
    }

    auto maybeVitisDir = findVitis(vitisDir);
    if (failed(maybeVitisDir)) return failure();

    if (failed(assembleUsingChess(inputLLChessHackedFile.c_str(), outputFile,
                                  *vitisDir, tempDir, verbose))) {
      llvm::errs() << "Failed to assemble input.o";
      return failure();
    }
  } else {
    SmallString<64> peanoOptBin(peanoDir);
    sys::path::append(peanoOptBin, "bin", "opt");
    SmallString<64> peanoLLCBin(peanoDir);
    sys::path::append(peanoLLCBin, "bin", "llc");

    SmallString<64> OptLLVMIRFile(tempDir);
    sys::path::append(OptLLVMIRFile, "input.opt.ll");
    if (!runTool(peanoOptBin,
                 {"-O2", "--inline-threshold=10", "-S", std::string(LLVMIRFile),
                  "--disable-builtin=memset", "-o", std::string(OptLLVMIRFile)},
                 verbose)) {
      llvm::errs() << "Failed to optimize ll with peano";
      return failure();
    }

    if (!runTool(
            peanoLLCBin,
            {std::string(OptLLVMIRFile), "-O2",
             "--march=" + StringRef(targetArch).lower(), "--function-sections",
             "--filetype=obj", "-o", std::string(outputFile)},
            verbose)) {
      llvm::errs() << "Failed to assemble ll with peano";
      return failure();
    }
  }
  copy->erase();
  return success();
}

LogicalResult aie2xclbin(
    MLIRContext *ctx, ModuleOp moduleOp, const std::string &outputNPU,
    const std::string &outputXCLBin, bool printIRBeforeAll,
    bool printIRAfterAll, bool printIRModuleScope, bool timing,
    const std::string &tempDir, bool useChess, bool verbose,
    std::optional<std::string> vitisDir, const std::string &targetArch,
    const std::string &peanoDir, const std::string &xclBinKernelID,
    const std::string &xclBinKernelName, const std::string &xclBinInstanceName,
    const std::string &amdAIEInstallDir, const std::string &InputXCLBin) {
  PassManager pm(ctx, moduleOp.getOperationName());
  applyConfigToPassManager(pm, printIRBeforeAll, printIRAfterAll,
                           printIRModuleScope, timing);
  // generateNPUInstructions
  pm.addNestedPass<AIE::DeviceOp>(
      mlir::iree_compiler::AMDAIE::createAMDAIEDmaToNpuPass());
  if (failed(pm.run(moduleOp)))
    return moduleOp.emitOpError(": NPU Instruction pipeline failed");

  // TODO(max): should be using UI32 resource or something like that...
  ArrayRef<int32_t> signedNpuInstructionsAttr =
      cast<DenseI32ArrayAttr>(
          (*moduleOp.getOps<xilinx::AIE::DeviceOp>().begin())
              ->getAttr("npu_instructions"))
          .asArrayRef();
  std::vector<uint32_t> unsignedNpuInstructions(
      signedNpuInstructionsAttr.begin(), signedNpuInstructionsAttr.end());

  std::string errorMessage;
  auto output = openOutputFile(outputNPU, &errorMessage);
  if (!output) {
    llvm::errs() << "Failed to open npu_instructions.txt for writing because: "
                 << errorMessage;
    return failure();
  }
  for (auto w : unsignedNpuInstructions)
    output->os() << llvm::format("%08X\n", w);
  output->keep();

  SmallString<64> unifiedObj(tempDir);
  sys::path::append(unifiedObj, "input.o");
  if (failed(generateUnifiedObject(
          ctx, moduleOp, std::string(unifiedObj), printIRBeforeAll,
          printIRAfterAll, printIRModuleScope, timing, useChess, verbose,
          tempDir, vitisDir, targetArch, peanoDir)))
    return moduleOp.emitOpError("Failed to generate unified object");

  if (failed(generateCoreElfFiles(moduleOp, unifiedObj, tempDir, useChess,
                                  vitisDir, targetArch, verbose, peanoDir)))
    return moduleOp.emitOpError("Failed to generate core ELF file(s)");

  if (failed(generateCDO(ctx, moduleOp, printIRBeforeAll, printIRAfterAll,
                         printIRModuleScope, timing, tempDir)))
    return moduleOp.emitOpError("Failed to generate CDO");

  if (failed(generateXCLBin(moduleOp, outputXCLBin, tempDir, xclBinKernelID,
                            xclBinKernelName, xclBinInstanceName,
                            amdAIEInstallDir, verbose, InputXCLBin)))
    return moduleOp.emitOpError("Failed to generate XCLBin");

  return success();
}
