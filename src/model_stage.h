// model_stage.h -- copy a model's GPU tensors to a tmpfs (huge pages) and map them from there.
//
// The tensors the GPU reads are copied, each aligned to 2 MB, into one stage file <dir>/rocm-inference-stage-<hash>.bin
// and the GGUF's tensor pointers are pointed at the copies (gguf_set_tensor_staged); what only the CPU reads (the
// token embeddings when the LM head is separate, the PLE table) stays in the GGUF's own mapping on disk. The hash
// covers the source files' identity (device, inode, size, mtime) and the staged tensors, so a stage is reused by any
// later load of the same files (no copy: map and pin) and never used for other files. A stage is written under a
// temporary name and renamed when complete. On a huge=always tmpfs (/dev/shm here) the stage sits on 2 MB pages,
// which hipHostRegister pins far faster than the disk mapping's 4 KB pages; the source's page cache is dropped behind
// the copy.

#ifndef MODEL_STAGE_H
#define MODEL_STAGE_H

#include "gguf_mmap.h"

#include <functional>
#include <string>

#define GGUF_MAX_SPLITS_STAGE 64  // GGUF_MAX_SPLITS in gguf_mmap.cpp

struct ModelStage {
    void* base = nullptr;   // the stage mapping (read-only once complete)
    size_t bytes = 0;       // its size
    size_t staged = 0;      // tensor bytes staged
    std::string path;       // the stage file
    bool reused = false;    // an existing stage was mapped (nothing copied)
};

// Stage the tensors of g for which keep(name) is true into dir. Returns true with st filled and the tensors
// redirected; false (why set, nothing changed) when it cannot: dir missing, not enough space in dir or memory, I/O.
bool model_stage(gguf_model_t* g, const char* dir, const std::function<bool(const char*)>& keep, ModelStage& st,
                 std::string& why);

// Unmap a stage (the file stays for the next load). Restores no pointers: call when the model goes.
void model_stage_unmap(ModelStage& st);

#endif  // MODEL_STAGE_H
