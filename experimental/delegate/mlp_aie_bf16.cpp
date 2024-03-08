//////////////////////////////////////////////////////////////////////////////
// bf16 implementation from Joe Melber

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

#include "mlp_aie_bf16.h"

std::vector<uint32_t> loadInstrSequence(std::string instr_path) {
    std::ifstream instrFile(instr_path);
    std::string line;
    std::vector<uint32_t> instrV;
    while (std::getline(instrFile, line)) {
        std::istringstream iss(line);
        uint32_t a;
        if (!(iss >> std::hex >> a)) {
           std::cerr << "Unable to parse instruction file" << std::endl;
           return {};
        }
        instrV.push_back(a);
    }
    return instrV;
}

struct XrtState {
    xrt::device device;
    xrt::kernel kernel;
    xrt::bo boInstr;
    xrt::bo boA;
    xrt::bo boB;
    xrt::bo boC;

    static XrtState *getInstance(bool shouldDelete = false) {
        static XrtState *instance = nullptr;
        if (shouldDelete) {
            delete instance;
            instance = nullptr;
            return nullptr;
        }
        if (instance == nullptr)
            instance = new XrtState();
        return instance;
    }
};

constexpr int M = MLP_M;
constexpr int K = MLP_K;
constexpr int N = MLP_N;

constexpr int aVolume = M * K;
constexpr int bVolume = K * N;
constexpr int cVolume = M * N;

using bfloat16_t = uint16_t;
using A_DATATYPE = bfloat16_t; // std::bfloat16_t;
using B_DATATYPE = bfloat16_t; // std::bfloat16_t;
using C_DATATYPE = float;

constexpr int aSize = (aVolume * sizeof(A_DATATYPE));
constexpr int bSize = (bVolume * sizeof(B_DATATYPE));
constexpr int cSize = (cVolume * sizeof(C_DATATYPE));

bool aie_setup = false;
int instrSize = 0;
int aie_matmuls_done = 0;
int matmuls_done = 0;

inline bfloat16_t toBfloat16(float f) {
    bfloat16_t bf = (bfloat16_t) (((*reinterpret_cast<uint32_t*>(&f))) >> 16);
    return bf;
}

inline float fromBfloat16(bfloat16_t b) {
    uint32_t tmp = uint32_t(b) << 16;
    float f = *reinterpret_cast<float*>(&tmp);
    return f;
}

extern "C" {

int setupNPUAccelerator() {
    std::vector<uint32_t> instrV = 
        // loadInstrSequence("aie_design/insts.txt");
        loadInstrSequence("/proj/gdba/dliddell/Projects/iree/iree-amd-aie/third_party/mlir-aie/reference_designs/ipu-xrt/matrix_multiplication/build/insts.txt");
    instrSize = instrV.size();
    if (instrSize == 0) {
        std::cout << "Couldn't load instructions" << std::endl;
        return 1;
    }
    std::cout << "Sequence instr count: " << instrV.size() << "\n";

    // Start the XRT test code
    // Get a device handle
    auto xrtState = XrtState::getInstance();
    unsigned int deviceIndex = 0;
    xrtState->device = xrt::device(deviceIndex);

    // Load the xclbin
    // auto xclbin = xrt::xclbin("aie_design/final.xclbin");
    auto xclbin = xrt::xclbin("/proj/gdba/dliddell/Projects/iree/iree-amd-aie/third_party/mlir-aie/reference_designs/ipu-xrt/matrix_multiplication/build/final.xclbin");

    std::string node = "MLIR_AIE";

    // Get the kernel from the xclbin
    auto xkernels = xclbin.get_kernels();
    auto xkernel = *std::find_if(xkernels.begin(), xkernels.end(),
                                 [node](xrt::xclbin::kernel &k) {
                                   auto name = k.get_name();
                                   std::cout << "Name: " << name << std::endl;
                                   return name.rfind(node, 0) == 0;
                                 });
    auto kernelName = xkernel.get_name();

    // Register the xclbin
    xrtState->device.register_xclbin(xclbin);

    // Get a hardware context
    xrt::hw_context context(xrtState->device, xclbin.get_uuid());

    // Get a kernel handle
    xrtState->kernel = xrt::kernel(context, kernelName);
 
    xrtState->boInstr = xrt::bo(xrtState->device, instrV.size() * sizeof(int),
                           XCL_BO_FLAGS_CACHEABLE, xrtState->kernel.group_id(0));
    xrtState->boA = xrt::bo(xrtState->device, aSize, XRT_BO_FLAGS_HOST_ONLY, xrtState->kernel.group_id(2));
    xrtState->boB = xrt::bo(xrtState->device, bSize, XRT_BO_FLAGS_HOST_ONLY, xrtState->kernel.group_id(3));
    xrtState->boC = xrt::bo(xrtState->device, cSize, XRT_BO_FLAGS_HOST_ONLY, xrtState->kernel.group_id(4));

    // copy instruction stream to NPU
    void *bufInstr = xrtState->boInstr.map<void *>();
    std::memcpy(bufInstr, instrV.data(), instrV.size() * sizeof(int));
    xrtState->boInstr.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    std::cout << "NPU setup done." << std::endl;
    
    aie_setup = true;
    return 0;  // TODO: check for and handle error conditions
}

int aie_matmul(const params_t *params) {
    int cnt = 0;

// TODO: handle strides

    // quantize and copy weights to XRT BO
    auto xrtState = XrtState::getInstance();
    bfloat16_t *bufA = xrtState->boA.map<bfloat16_t *>();
    // std::cout << "Input A" << std::endl;
    for (int i = 0; i < M; i++) {
        // std::cout << '[';
        for (int j = 0; j < K; j++) {
            float f = params->lhs[params->lhs_offset + i * params->lhs_stride0 + j * params->lhs_stride1];
            bfloat16_t bf = toBfloat16(f);
            // std::cout << "{f:" << f << ",bf:" << bf << "}";
            // *(bufA + i * K + j) = toBfloat16(params->lhs[params->lhs_offset + i * params->lhs_stride0 + j * params->lhs_stride1]);
            *(bufA + i * K + j) = bf;
        }
        // std::cout << "]," << std::endl;
    }

    cnt = 0;

    // quantize and copy input to XRT BO
    bfloat16_t *bufB = xrtState->boB.map<bfloat16_t *>();
    for (int i = 0; i < K; i++)
        for (int j = 0; j < N; j++)
            *(bufB + i * N + j) = toBfloat16(params->rhs[params->rhs_offset + i * params->rhs_stride0 + j * params->rhs_stride1]);

    // copy output to XRT BO
    float *bufC = xrtState->boC.map<float *>();
    std::memcpy(bufC, params->result + params->result_offset, (M * sizeof(float)));

    // sync buffers to NPU device
    xrtState->boA.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    xrtState->boB.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    xrtState->boC.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    // execute the kernel on NPU
    auto run = xrtState->kernel(xrtState->boInstr, instrSize, xrtState->boA, xrtState->boB, xrtState->boC);
    run.wait();

    // sync output to host
    xrtState->boC.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    std::memcpy(params->result + params->result_offset, bufC, (M * N * sizeof(float)));
    // std::vector<float> res(M * N, 768.0);
    // std::memcpy(params->result + params->result_offset, res.data(), (M * N * sizeof(float)));

    std::cout << "Result" << std::endl;
    for (int i = 0; i < M; i++) {
        std::cout << '[';
        for (int j = 0; j < N; j++) {
            float f = params->result[params->result_offset + i * N + j];
            std::cout << f << ",";
        }
        std::cout << "]," << std::endl;
    }

    return 0;  // TODO: check for and handle error conditions
}

} // extern "C"
