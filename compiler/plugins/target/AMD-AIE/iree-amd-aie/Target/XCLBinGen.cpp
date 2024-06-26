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
#include "aie/Dialect/AIEVec/Pipelines/Passes.h"
#include "aie/Targets/AIETargets.h"
#include "iree-amd-aie/Transforms/Passes.h"
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

void findVitis(XCLBinGenConfig &TK) {
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
    sys::path::append(aietools_bin_path, "bin");
    const char *env_path = ::getenv("PATH");
    if (env_path == nullptr) env_path = "";
    SmallString<128> new_path(env_path);
    if (new_path.size()) new_path += sys::EnvPathSeparator;
    new_path += aietools_bin_path;
    new_path += sys::EnvPathSeparator;
    new_path += vitis_bin_path;
    ::setenv("PATH", new_path.c_str(), 1);
  } else {
    errs() << "VITIS not found ...\n";
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

int runTool(StringRef program, ArrayRef<std::string> args, bool verbose,
            std::optional<ArrayRef<StringRef>> env = std::nullopt) {
  if (verbose) {
    llvm::outs() << "Run: ";
    if (env)
      for (auto &s : *env) llvm::outs() << " " << s;
    llvm::outs() << " " << program;
    for (auto &s : args) llvm::outs() << " " << s;
    llvm::outs() << "\n";
  }
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
      return -1;
    }
  }
  std::string temporaryPathStr =
      std::string(temporaryPath.begin(), temporaryPath.size());
  StringRef temporaryPathRef(temporaryPathStr);
  auto tp = std::optional<StringRef>(temporaryPathRef);
  int result =
      sys::ExecuteAndWait(program, pArgs, env, /* redirects */ {tp, tp, tp}, 0,
                          0, &errMsg, nullptr, &optStats);
  if (verbose) {
    auto totalTime = std::chrono::duration_cast<std::chrono::duration<float>>(
                         stats.TotalTime)
                         .count();
    std::string exitStatusStr = result == 0 ? "Succeeded" : "Failed";
    llvm::outs() << exitStatusStr << " in " << totalTime
                 << " [s]. Exit code=" << result << "\n";
    std::ifstream t(temporaryPathRef.str());
    if (t.is_open() && t.good()) {
      std::stringstream buffer;
      buffer << t.rdbuf();
      llvm::outs() << buffer.str();
    } else {
      llvm::outs() << "Failed to open temporary file " << temporaryPathRef.str()
                   << ", not printing output\n";
    }
  }
  return result;
}

template <unsigned N>
static void aieTargetDefines(SmallVector<std::string, N> &Args,
                             std::string aie_target) {
  Args.push_back("-D__AIEARCH__=20");
}

// Generate the elf files for the core
static LogicalResult generateCoreElfFiles(ModuleOp moduleOp,
                                          const StringRef objFile,
                                          XCLBinGenConfig &TK) {
  auto deviceOps = moduleOp.getOps<AIE::DeviceOp>();
  if (!llvm::hasSingleElement(deviceOps))
    return moduleOp.emitOpError("expected a single device op");

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

    if (TK.UseChess) {
      // Use xbridge (to remove any peano dependency with use-chess option)
      SmallString<64> bcfPath(TK.TempDir);
      sys::path::append(bcfPath, elfFileName + ".bcf");

      {
        auto bcfOutput = openOutputFile(bcfPath, &errorMessage);
        if (!bcfOutput) return coreOp.emitOpError(errorMessage);

        if (failed(mlir::iree_compiler::AMDAIE::AIETranslateToBCF(
                moduleOp, bcfOutput->os(), col, row)))
          return coreOp.emitOpError("Failed to generate BCF");
        bcfOutput->keep();
      }

      std::vector<std::string> extractedIncludes;
      {
        auto bcfFileIn = openInputFile(bcfPath, &errorMessage);
        if (!bcfFileIn) moduleOp.emitOpError(errorMessage);

        std::string bcfFile = std::string(bcfFileIn->getBuffer());
        std::regex r("_include _file (.*)");
        auto begin = std::sregex_iterator(bcfFile.begin(), bcfFile.end(), r);
        auto end = std::sregex_iterator();
        for (std::sregex_iterator i = begin; i != end; ++i)
          extractedIncludes.push_back(i->str(1));
      }

      SmallString<64> chessWrapperBin(TK.InstallDir);
      sys::path::append(chessWrapperBin, "bin", "xchesscc_wrapper");
      SmallString<64> chessworkDir(TK.TempDir);
      sys::path::append(chessworkDir, "chesswork");

      SmallVector<std::string> flags{StringRef(TK.TargetArch).lower(),
                                     "+w",
                                     std::string(chessworkDir),
                                     "-d",
                                     "+l",
                                     std::string(bcfPath),
                                     "-o",
                                     std::string(elfFile),
                                     "-f",
                                     std::string(objFile)};
      for (const auto &inc : extractedIncludes) flags.push_back(inc);

      if (runTool(chessWrapperBin, flags, TK.Verbose) != 0)
        coreOp.emitOpError("Failed to link with xbridge");
    } else {
      SmallString<64> ldscript_path(TK.TempDir);
      sys::path::append(ldscript_path, elfFileName + ".ld");
      {
        auto ldscript_output = openOutputFile(ldscript_path, &errorMessage);
        if (!ldscript_output) return coreOp.emitOpError(errorMessage);

        if (failed(mlir::iree_compiler::AMDAIE::AIETranslateToLdScript(
                moduleOp, ldscript_output->os(), col, row)))
          return coreOp.emitOpError("failed to generate ld script for core (")
                 << col << "," << row << ")";
        ldscript_output->keep();
      }

      // We are running a clang command for now, but really this is an lld
      // command.
      {
        std::string targetLower = StringRef(TK.TargetArch).lower();
        SmallVector<std::string, 10> flags;
        flags.push_back("-O2");
        std::string targetFlag = "--target=" + targetLower + "-none-elf";
        flags.push_back(targetFlag);
        flags.emplace_back(objFile);
        SmallString<64> meBasicPath(TK.InstallDir);
        sys::path::append(meBasicPath, "aie_runtime_lib",
                          StringRef(TK.TargetArch).upper(), "me_basic.o");
        flags.emplace_back(meBasicPath);
        SmallString<64> libcPath(TK.PeanoDir);
        sys::path::append(libcPath, "lib", targetLower + "-none-unknown-elf",
                          "libc.a");
        flags.emplace_back(libcPath);
        flags.push_back("-Wl,--gc-sections");
        std::string ldScriptFlag = "-Wl,-T," + std::string(ldscript_path);
        flags.push_back(ldScriptFlag);
        flags.push_back("-o");
        flags.emplace_back(elfFile);
        SmallString<64> clangBin(TK.PeanoDir);
        sys::path::append(clangBin, "bin", "clang");
        if (runTool(clangBin, flags, TK.Verbose) != 0)
          return coreOp.emitOpError("failed to link elf file for core(")
                 << col << "," << row << ")";
      }
    }
  }
  return success();
}

static LogicalResult generateCDO(MLIRContext *context, ModuleOp moduleOp,
                                 XCLBinGenConfig &TK) {
  ModuleOp copy = moduleOp.clone();
  std::string errorMessage;
  // This corresponds to `process_host_cgen`, which is listed as host
  // compilation in aiecc.py... not sure we need this.
  PassManager passManager(context, ModuleOp::getOperationName());
  applyConfigToPassManager(TK, passManager);

  passManager.addNestedPass<AIE::DeviceOp>(
      mlir::iree_compiler::AMDAIE::createAMDAIEPathfinderPass());
  if (failed(passManager.run(copy)))
    return moduleOp.emitOpError(
        "failed to run passes to prepare of XCLBin generation");

  if (failed(mlir::iree_compiler::AMDAIE::AIETranslateToCDODirect(copy,
                                                                  TK.TempDir)))
    return moduleOp.emitOpError("failed to emit CDO");

  copy->erase();
  return success();
}

static json::Object makeKernelJSON(std::string name, std::string id,
                                   std::string instance) {
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

static LogicalResult generateXCLBin(MLIRContext *context, ModuleOp moduleOp,
                                    XCLBinGenConfig &TK,
                                    const StringRef &Output,
                                    const StringRef &inputXclbin = "") {
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

    json::Object kernels_data{
        {"ps-kernels",
         json::Object{
             {"kernels",
              json::Array{// TODO: Support for multiple kernels
                          makeKernelJSON(TK.XCLBinKernelName, TK.XCLBinKernelID,
                                         TK.XCLBinInstanceName)}}}}};
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
                       << "\tid_code = 0x14ca8093\n"
                       << "\textended_id_code = 0x01\n"
                       << "\timage\n"
                       << "\t{\n"
                       << "\t\tname=aie_image, id=0x1c000000\n"
                       << "\t\t{ type=cdo\n"
                       << "\t\t  file=" << TK.TempDir << "/aie_cdo_elfs.bin\n"
                       << "\t\t  file=" << TK.TempDir << "/aie_cdo_init.bin\n"
                       << "\t\t  file=" << TK.TempDir << "/aie_cdo_enable.bin\n"
                       << "\t\t}\n"
                       << "\t}\n"
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

    SmallString<64> bootgenBin(TK.InstallDir);
    sys::path::append(bootgenBin, "bin", "bootgen");
    if (runTool(bootgenBin, flags, TK.Verbose) != 0)
      return moduleOp.emitOpError("failed to execute bootgen");
  }
  SmallVector<std::string, 20> flags;
  // Execute the xclbinutil command.
  std::string memArg = "MEM_TOPOLOGY:JSON:" + std::string(memTopologyJsonFile);
  std::string partArg =
      "AIE_PARTITION:JSON:" + std::string(aiePartitionJsonFile);
  {
    if (!inputXclbin.empty()) {
      // Create aie_partition.json.
      SmallString<64> aieInputPartitionJsonFile(TK.TempDir);
      sys::path::append(aieInputPartitionJsonFile, "aie_input_partition.json");

      std::string inputPartArg =
          "AIE_PARTITION:JSON:" + std::string(aieInputPartitionJsonFile);
      SmallVector<std::string, 20> inputFlags{"--dump-section", inputPartArg,
                                              "--force", "--input",
                                              std::string(inputXclbin)};
      if (auto xclbinutil = sys::findProgramByName("xclbinutil")) {
        if (runTool(*xclbinutil, inputFlags, TK.Verbose) != 0)
          return moduleOp.emitOpError("failed to execute xclbinutil");
      } else {
        return moduleOp.emitOpError("could not find xclbinutil");
      }
      auto aieInputPartitionOut =
          openInputFile(aieInputPartitionJsonFile, &errorMessage);
      if (!aieInputPartitionOut) return moduleOp.emitOpError(errorMessage);
      Expected<json::Value> aieInputPartitionOutValue =
          llvm::json::parse(aieInputPartitionOut->getBuffer());
      json::Array *aieInputPartionPDIs;
      aieInputPartionPDIs = aieInputPartitionOutValue->getAsObject()
                                ->getObject("aie_partition")
                                ->getArray("PDIs");
      auto aiePartitionOut = openInputFile(aiePartitionJsonFile, &errorMessage);
      if (!aiePartitionOut) return moduleOp.emitOpError(errorMessage);
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
      auto aiePartitionJsonOut =
          openOutputFile(aiePartitionJsonFile, &errorMessage);
      if (!aiePartitionJsonOut) return moduleOp.emitOpError(errorMessage);
      aiePartitionJsonOut->os() << formatv("{0:2}", *aieInputPartitionOutValue);
      aiePartitionJsonOut->keep();
      flags.insert(flags.end(), {"--input", std::string(inputXclbin)});

    } else {
      flags.insert(flags.end(), {"--add-replace-section", memArg});
    }
    flags.insert(flags.end(), {"--add-kernel", std::string(kernelsJsonFile),
                               "--add-replace-section", partArg, "--force",
                               "--output", std::string(Output)});

    if (auto xclbinutil = sys::findProgramByName("xclbinutil")) {
      if (runTool(*xclbinutil, flags, TK.Verbose) != 0)
        return moduleOp.emitOpError("failed to execute xclbinutil");
    } else {
      return moduleOp.emitOpError("could not find xclbinutil");
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

static LogicalResult generateUnifiedObject(MLIRContext *context,
                                           ModuleOp moduleOp,
                                           XCLBinGenConfig &TK,
                                           const std::string &outputFile) {
  PassManager pm(context, moduleOp.getOperationName());
  applyConfigToPassManager(TK, pm);

  pm.addPass(mlir::iree_compiler::AMDAIE::createAMDAIECoreToStandardPass());
  pm.addPass(mlir::iree_compiler::AMDAIE::createAMDAIEXToStandardPass());

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
      return moduleOp.emitOpError("Failed to parse options from '")
             << optionsString
             << "': Failed to construct ConvertVectorToAIEVecOptions.";
    }
    xilinx::aievec::buildConvertVectorToAIEVec(pm, vectorToAIEVecOptions);
  }

  mlir::iree_compiler::AMDAIE::addLowerToLLVMPasses(pm);
  pm.addPass(std::make_unique<RemoveAlignment2FromLLVMLoadPass>());

  if (TK.Verbose) {
    llvm::outs() << "Running: ";
    pm.printAsTextualPipeline(llvm::outs());
    llvm::outs() << "\n";
  }

  ModuleOp copy = moduleOp.clone();
  if (failed(pm.run(copy)))
    return moduleOp.emitOpError("Failed to lower to LLVM");

  SmallString<64> LLVMIRFile(TK.TempDir);
  sys::path::append(LLVMIRFile, "input.ll");

  llvm::LLVMContext llvmContext;
  auto llvmModule = translateModuleToLLVMIR(copy, llvmContext);
  if (!llvmModule)
    return moduleOp.emitOpError("Failed to translate module to LLVMIR");

  std::string errorMessage;
  {
    auto output = openOutputFile(LLVMIRFile, &errorMessage);
    if (!output) return moduleOp.emitOpError(errorMessage);
    llvmModule->print(output->os(), nullptr);
    output->keep();
  }

  if (TK.UseChess) {
    SmallString<64> chessWrapperBin(TK.InstallDir);
    sys::path::append(chessWrapperBin, "bin", "xchesscc_wrapper");

    SmallString<64> chessworkDir(TK.TempDir);
    sys::path::append(chessworkDir, "chesswork");

    SmallString<64> chessIntrinsicsLL(TK.InstallDir);
    sys::path::append(chessIntrinsicsLL, "aie_runtime_lib",
                      StringRef(TK.TargetArch).upper(),
                      "chess_intrinsic_wrapper.ll");

    std::string llvmirString;
    {
      raw_string_ostream llvmirStream(llvmirString);
      llvmModule->print(llvmirStream, nullptr);
    }

    SmallString<64> chesslinkedFile(TK.TempDir);
    sys::path::append(chesslinkedFile, "input.chesslinked.ll");
    SmallString<64> llvmLinkBin(TK.PeanoDir);
    sys::path::append(llvmLinkBin, "bin", "llvm-link");
    if (!sys::fs::exists(llvmLinkBin)) {
      if (auto llvmLink = sys::findProgramByName("llvm-link"))
        llvmLinkBin = *llvmLink;
      else
        moduleOp.emitOpError("Can't find llvm-link");
    }
    if (runTool(llvmLinkBin,
                {std::string(LLVMIRFile), std::string(chessIntrinsicsLL), "-S",
                 "-o", std::string(chesslinkedFile)},
                TK.Verbose) != 0)
      moduleOp.emitOpError("Couldn't link in the intrinsics");

    std::string mungedLLVMIR;
    {
      auto chesslinkedIn = openInputFile(chesslinkedFile, &errorMessage);
      if (!chesslinkedIn) moduleOp.emitOpError(errorMessage);

      mungedLLVMIR = std::string(chesslinkedIn->getBuffer());
      mungedLLVMIR = chesshack(mungedLLVMIR);
    }
    {
      auto chesslinkedOut = openOutputFile(chesslinkedFile);
      if (!chesslinkedOut) moduleOp.emitOpError(errorMessage);

      chesslinkedOut->os() << mungedLLVMIR;
      chesslinkedOut->keep();
    }

    if (runTool(chessWrapperBin,
                {StringRef(TK.TargetArch).lower(), "+w",
                 std::string(chessworkDir), "-c", "-d", "-f", "+P", "4",
                 std::string(chesslinkedFile), "-o", std::string(outputFile)},
                TK.Verbose) != 0)
      return moduleOp.emitOpError("Failed to assemble with chess");
  } else {
    SmallString<64> peanoOptBin(TK.PeanoDir);
    sys::path::append(peanoOptBin, "bin", "opt");
    SmallString<64> peanoLLCBin(TK.PeanoDir);
    sys::path::append(peanoLLCBin, "bin", "llc");

    SmallString<64> OptLLVMIRFile(TK.TempDir);
    sys::path::append(OptLLVMIRFile, "input.opt.ll");
    if (runTool(peanoOptBin,
                {"-O2", "--inline-threshold=10", "-S", std::string(LLVMIRFile),
                 "--disable-builtin=memset", "-o", std::string(OptLLVMIRFile)},
                TK.Verbose) != 0)
      return moduleOp.emitOpError("Failed to optimize");

    if (runTool(peanoLLCBin,
                {std::string(OptLLVMIRFile), "-O2",
                 "--march=" + StringRef(TK.TargetArch).lower(),
                 "--function-sections", "--filetype=obj", "-o",
                 std::string(outputFile)},
                TK.Verbose) != 0)
      return moduleOp.emitOpError("Failed to assemble");
  }
  copy->erase();
  return success();
}

LogicalResult xilinx::aie2xclbin(MLIRContext *ctx, ModuleOp moduleOp,
                                 XCLBinGenConfig &TK, StringRef OutputNPU,
                                 StringRef OutputXCLBin,
                                 StringRef InputXCLBin) {
  PassManager pm(ctx, moduleOp.getOperationName());
  applyConfigToPassManager(TK, pm);

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
  auto output = openOutputFile(OutputNPU, &errorMessage);
  if (!output) return moduleOp.emitOpError(errorMessage);
  for (auto w : unsignedNpuInstructions)
    output->os() << llvm::format("%08X\n", w);
  output->keep();

  SmallString<64> unifiedObj(TK.TempDir);
  sys::path::append(unifiedObj, "input.o");
  if (failed(generateUnifiedObject(ctx, moduleOp, TK, std::string(unifiedObj))))
    return moduleOp.emitOpError("Failed to generate unified object");

  if (failed(generateCoreElfFiles(moduleOp, unifiedObj, TK)))
    return moduleOp.emitOpError("Failed to generate core ELF file(s)");

  if (failed(generateCDO(ctx, moduleOp, TK)))
    return moduleOp.emitOpError("Failed to generate CDO");

  if (failed(generateXCLBin(ctx, moduleOp, TK, OutputXCLBin, InputXCLBin)))
    return moduleOp.emitOpError("Failed to generate XCLBin");

  return success();
}
