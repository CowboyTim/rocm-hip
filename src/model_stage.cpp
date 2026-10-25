// model_stage.cpp -- see model_stage.h.

#include "model_stage.h"

#include "inference_log.h"
#include "sha256.h"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/statvfs.h>
#include <thread>
#include <unistd.h>
#include <vector>

namespace {

const size_t kAlign = 2u << 20;       // each tensor on its own 2 MB page(s)
const size_t kChunk = 64u << 20;      // copy unit (threads take chunks)
const char kMagic[8] = {'R', 'I', 'S', 'T', 'A', 'G', 'E', '1'};

struct Header {
    char magic[8];
    uint32_t n_files, n_entries;
    uint64_t total;
    uint64_t ids[GGUF_MAX_SPLITS_STAGE][4];
};
struct Entry {
    uint32_t idx, pad;
    uint64_t off, bytes;
};

size_t align_up(size_t v, size_t a) { return (v + a - 1) / a * a; }

// MemAvailable from /proc/meminfo, in bytes (0 if unknown)
size_t mem_available() {
    FILE* f = fopen("/proc/meminfo", "r");
    if (!f) return 0;
    char line[256];
    size_t kb = 0;
    while (fgets(line, sizeof line, f))
        if (sscanf(line, "MemAvailable: %zu kB", &kb) == 1) break;
    fclose(f);
    return kb << 10;
}

}  // namespace

bool model_stage(gguf_model_t* g, const char* dir, const std::function<bool(const char*)>& keep, ModelStage& st,
                 std::string& why) {
    st = ModelStage();
    const int n_files = gguf_n_maps(g);
    if (!dir || !dir[0] || n_files < 1 || n_files > GGUF_MAX_SPLITS_STAGE) { why = "no stage directory"; return false; }

    // The layout: a header (+ entries), then each kept tensor at the next 2 MB boundary
    std::vector<Entry> ents;
    const int64_t nt = gguf_n_tensors(g);
    size_t cur = align_up(sizeof(Header) + (size_t)nt * sizeof(Entry), kAlign);
    sha256_ctx h;
    sha256_init(&h);
    sha256_update(&h, kMagic, sizeof kMagic);
    for (int i = 0; i < n_files; ++i) {
        uint64_t id[4];
        gguf_file_identity(g, i, id);
        sha256_update(&h, id, sizeof id);
    }
    for (int64_t t = 0; t < nt; ++t) {
        const char* name = gguf_tensor_name(g, t);
        const size_t b = gguf_tensor_nbytes(g, t);
        if (!name || b == 0 || b == (size_t)-1 || !keep(name)) continue;
        cur = align_up(cur, kAlign);
        ents.push_back({(uint32_t)t, 0, cur, b});
        cur += b;
        sha256_update(&h, name, strlen(name) + 1);
        sha256_update(&h, &ents.back(), sizeof(Entry));
    }
    const size_t total = align_up(cur, kAlign);
    uint8_t dg[32];
    sha256_final(&h, dg);
    char hex[33];
    for (int i = 0; i < 16; ++i) snprintf(hex + 2 * i, 3, "%02x", dg[i]);
    st.path = std::string(dir) + "/rocm-inference-stage-" + hex + ".bin";
    for (const Entry& e : ents) st.staged += e.bytes;

    auto redirect = [&](const char* base) {
        for (const Entry& e : ents) gguf_set_tensor_staged(g, e.idx, base + e.off);
    };

    // An existing stage of the same files: map it (a complete one only: it was renamed into place when done)
    int fd = open(st.path.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd >= 0) {
        struct stat s;
        void* p = fstat(fd, &s) == 0 && (size_t)s.st_size == total
                      ? mmap(nullptr, total, PROT_READ, MAP_SHARED, fd, 0)
                      : MAP_FAILED;
        close(fd);
        if (p != MAP_FAILED) {
            const Header* hd = (const Header*)p;
            const Entry* he = (const Entry*)(hd + 1);
            bool ok = memcmp(hd->magic, kMagic, sizeof kMagic) == 0 && hd->total == total &&
                      hd->n_entries == ents.size() && hd->n_files == (uint32_t)n_files;
            for (size_t i = 0; ok && i < ents.size(); ++i)
                ok = he[i].idx == ents[i].idx && he[i].off == ents[i].off && he[i].bytes == ents[i].bytes;
            if (ok) {
                st.base = p;
                st.bytes = total;
                st.reused = true;
                redirect((const char*)p);
                return true;
            }
            munmap(p, total);
        }
        why = "a stage file of another layout is in the way: " + st.path;
        return false;
    }

    // Room for it: in the directory (tmpfs: its size limit) and in memory (tmpfs pages are RAM), with a margin for
    // the KV cache and scratch that come next
    struct statvfs vs;
    if (statvfs(dir, &vs) != 0) { why = std::string("cannot stat ") + dir + ": " + strerror(errno); return false; }
    const size_t room = (size_t)vs.f_bavail * vs.f_frsize, avail = mem_available();
    if (room < total + (256u << 20) || (avail && avail < total + (4ull << 30))) {
        char b[256];
        snprintf(b, sizeof b, "%zu MB to stage, %zu MB free in %s, %zu MB of memory available", total >> 20, room >> 20,
                 dir, avail >> 20);
        why = b;
        return false;
    }

    const std::string tmp = st.path + ".tmp." + std::to_string(getpid());
    fd = open(tmp.c_str(), O_RDWR | O_CREAT | O_EXCL | O_CLOEXEC, 0644);
    if (fd < 0) { why = "cannot create " + tmp + ": " + strerror(errno); return false; }
    char* dst = nullptr;
    if (ftruncate(fd, (off_t)total) != 0 ||
        (dst = (char*)mmap(nullptr, total, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0)) == (char*)MAP_FAILED) {
        why = "cannot size / map " + tmp + ": " + strerror(errno);
        close(fd);
        unlink(tmp.c_str());
        return false;
    }
    close(fd);

    // The copy: 64 MB chunks over all tensors, taken by threads in order (page faults on both sides run in
    // parallel); each chunk's source pages are dropped from the page cache behind it, so a model as big as the RAM
    // allows can be staged without holding two copies
    struct Chunk {
        const char* src;
        char* dst;
        size_t n;
        int file;
        uint64_t foff;  // file offset of src
    };
    std::vector<Chunk> chunks;
    for (const Entry& e : ents) {
        const char* src = (const char*)gguf_tensor_source(g, e.idx);
        const int f = gguf_tensor_split(g, e.idx);
        size_t mapsz = 0;
        const char* mb = (const char*)gguf_map_at(g, f, &mapsz);
        for (size_t o = 0; o < e.bytes; o += kChunk)
            chunks.push_back({src + o, dst + e.off + o, std::min(kChunk, e.bytes - o), f, (uint64_t)(src + o - mb)});
    }
    std::vector<int> fds(n_files, -1);
    for (int i = 0; i < n_files; ++i)
        if (const char* p = gguf_file_path(g, i)) fds[i] = open(p, O_RDONLY | O_CLOEXEC);
    std::atomic<size_t> next{0};
    const long page = sysconf(_SC_PAGESIZE);
    auto work = [&]() {
        for (size_t i; (i = next.fetch_add(1)) < chunks.size();) {
            const Chunk& c = chunks[i];
            memcpy(c.dst, c.src, c.n);
            // drop what was read: our mapping's pages and the file's page cache (whole pages inside the chunk)
            const uintptr_t b = ((uintptr_t)c.src + page - 1) / page * page, e = ((uintptr_t)c.src + c.n) / page * page;
            if (e > b) madvise((void*)b, e - b, MADV_DONTNEED);
            if (fds[c.file] >= 0) posix_fadvise(fds[c.file], (off_t)c.foff, (off_t)c.n, POSIX_FADV_DONTNEED);
        }
    };
    const unsigned nth = std::max(1u, std::min(8u, std::thread::hardware_concurrency()));
    std::vector<std::thread> th;
    for (unsigned i = 0; i < nth; ++i) th.emplace_back(work);
    for (std::thread& t : th) t.join();
    for (int f : fds)
        if (f >= 0) close(f);

    // The header last, then the rename that makes the stage visible
    Header* hd = (Header*)dst;
    memset(hd, 0, sizeof(Header));
    memcpy(hd->magic, kMagic, sizeof kMagic);
    hd->n_files = (uint32_t)n_files;
    hd->n_entries = (uint32_t)ents.size();
    hd->total = total;
    for (int i = 0; i < n_files; ++i) gguf_file_identity(g, i, hd->ids[i]);
    memcpy(hd + 1, ents.data(), ents.size() * sizeof(Entry));
    if (rename(tmp.c_str(), st.path.c_str()) != 0) {
        why = "cannot rename " + tmp + ": " + strerror(errno);
        munmap(dst, total);
        unlink(tmp.c_str());
        return false;
    }
    mprotect(dst, total, PROT_READ);
    st.base = dst;
    st.bytes = total;
    redirect(dst);
    return true;
}

void model_stage_unmap(ModelStage& st) {
    if (st.base) munmap(st.base, st.bytes);
    st.base = nullptr;
}
