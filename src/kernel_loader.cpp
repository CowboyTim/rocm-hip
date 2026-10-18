// kernel_loader.cpp -- see kernel_loader.h.

#include "kernel_loader.h"

#include <stdlib.h>
#include <string.h>

#include <string>

extern "C" {
extern const char inference_co_matvec_q[], inference_co_matvec_q_end[];
extern const char inference_co_gpu_ops[], inference_co_gpu_ops_end[];
extern const char inference_co_deltanet[], inference_co_deltanet_end[];
extern const char inference_co_attention[], inference_co_attention_end[];
}

namespace {
struct Embedded {
    const char* name;
    const char* begin;
    const char* end;
};
const Embedded kEmbedded[] = {
    {"matvec_q.co", inference_co_matvec_q, inference_co_matvec_q_end},
    {"gpu_ops.co", inference_co_gpu_ops, inference_co_gpu_ops_end},
    {"deltanet.co", inference_co_deltanet, inference_co_deltanet_end},
    {"attention.co", inference_co_attention, inference_co_attention_end},
};
}  // namespace

hipError_t kernel_module_load(hipModule_t* module, const char* name) {
    const char* base = strrchr(name, '/') ? strrchr(name, '/') + 1 : name;
    if (const char* dir = getenv("INFERENCE_KERNEL_DIR")) {
        if (dir[0]) return hipModuleLoad(module, (std::string(dir) + "/" + base).c_str());
    }
    for (const Embedded& e : kEmbedded)
        if (strcmp(e.name, base) == 0 && e.end > e.begin) return hipModuleLoadData(module, e.begin);
    return hipModuleLoad(module, name);
}
