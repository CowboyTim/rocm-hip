// attn_verify.cpp -- Phase 3 end-to-end acceptance test: one full GQA attention block over /dev/shm.
//
// Chains every verified Phase-3 component exactly as the engine dispatches them:
//   rmsnorm -> gemm_act_fp4 (Q/K/V projections from Codebook10 weights)
//          -> rope_rotate_half (per-head in-place; host cos/sin tables built from absolute position IDs)
//          -> kv_store into the managed KV cache (roped K + raw V at this position)
//          -> fp16_gemm (per-head scores = q @ K^T over GQA-expanded history columns)
//          -> softmax_masked (causal ceiling from absolute positions + valid-KV bound; 1/sqrt(hd) folded in)
//          -> fp16_gemm (per-head out = probs @ V over GQA-expanded history columns)
//          -> gemm_act_fp4 (output projection)
// Two scenarios run sequentially: prefill of 4 tokens at positions 0..3, then a decode step at position
// 4 reading the stored KV back through the cache manager. All buffers live in ONE pinned mapping plus
// the managed KV-cache file; a sequential CPU reference performing the same math is compared element-wise.
// Weights are generated as exact codebook codes x UE4M3 scales with canonical nibble encoding, and both
// sides use the SAME fp16-rounded dequanted values, so any difference is pure WMMA accumulation rounding.
// GQA head-grouping needs per-head row views that flat GEMMs cannot express directly, so small CPU-side
// reshapes happen at those boundaries inside one mapping (a production engine can instead pick per-head
// layouts when writing projections). Rope launches per head because its rows are contiguous only within
// one head's slice.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include "fp4_common.h"
#include "kv_cache.h"

using namespace rocmfp4;

#define HIP_CHECK(cmd, str) \
    do { \
        hipError_t s_ = (cmd); \
        if (s_ != hipSuccess) { \
            fprintf(stderr, "ERROR: %s at %s:%d (%s)\n", hipGetErrorString(s_), __FILE__, __LINE__, str); \
            exit(1); \
        } \
    } while(0)

struct Cfg {
    int n_embd = 32;      // hidden size (= d_q here, so the O-projection is square)
    int hd     = 8;       // per-head dim
    int hq     = 4;       // query heads   -> d_q  = 32
    int hkv    = 2;       // key/value heads (GQA group of 2) -> d_kv = 16
    float eps  = 1e-5f;
    float theta = 10000.0f;
};

static const int CTX = 8;                     // context length for this test
static const int MAXRUN = 4;                 // largest scenario batch (prefill tokens)
static int g_maxstep = 7;                    // -maxstep N: run only chain steps 1..N (bisecting aid)
static int g_attn_fd = -1;                   // main-mapping file handle / base / size, for fresh re-mmap probes
static void* g_attn_base = nullptr;
static size_t g_attn_total = 0;

static uint32_t g_rng_state = 101;
static float rndf(float lo, float hi) {       // LCG -> [lo, hi)
    g_rng_state = g_rng_state * 1664525u + 1013904223u;
    return lo + ((float)(g_rng_state >> 8) / (float)(1u << 24)) * (hi - lo);
}
static int rndi(int n) {                      // LCG -> [0, n)
    g_rng_state = g_rng_state * 1664525u + 1013904223u;
    return (int)((g_rng_state >> 8) % (uint32_t)n);
}

// Quantized weight matrix: exact codebook codes x UE4M3 scales per half-block. true_v holds the
// EFFECTIVE values the kernel multiplies with (one fp16 rounding at dequant), shared by both sides.
struct QuantMat {
    int N, K;                       // lines x values (K multiple of 16)
    std::vector<unsigned char> packed;   // N x K/2 bytes, canonical packing (even k -> low nibble)
    std::vector<unsigned char> scales;   // N x K/16 bytes
    std::vector<float> true_v;          // N x K effective float values
};

static QuantMat make_quant_mat(int N, int K, uint32_t seed) {
    QuantMat m; m.N = N; m.K = K;
    m.packed.assign((size_t)N * (K / 2), 0);
    m.scales.assign((size_t)N * (K / 16), 0);
    m.true_v.assign((size_t)N * K, 0.0f);

    std::vector<uint8_t> pool;                     // scale bytes decoding into [0.1, 1.0]
    for (int b = 0; b < 256; ++b) {
        float v = ue4m3_to_float((uint8_t)b);
        if (v >= 0.1f && v <= 1.0f) pool.push_back((uint8_t)b);
    }
    g_rng_state = seed;
    const size_t strips = K / 16;
    for (int n = 0; n < N; ++n)
        for (size_t s = 0; s < strips; ++s) {
            uint8_t sb = pool[rndi((int)pool.size())];
            float sc = ue4m3_to_float(sb);
            m.scales[(size_t)n * strips + s] = sb;
            for (int i = 0; i < 16; ++i) {
                int k = (int)s * 16 + i;
                int code = CODEBOOK[rndi(16)];     // exact entry, signed included
                uint8_t nib = (uint8_t)encode_code(code);   // canonical nibble (sign = bit 3, not two's-complement)
                m.packed[(size_t)n * (K / 2) + (k >> 1)] |= (nib << ((k & 1) * 4));      // even k -> low nibble
                m.true_v[(size_t)n * K + k] = (float)__half((float)code * sc);           // one fp16 round at dequant
            }
        }
    return m;
}

static bool bad_result(float g, float r) {         // ~1% of magnitude + tiny floor (covers chained fp16 output ulps)
    float d = fabsf(g - r);
    return d > 0.01f * fmaxf(fabsf(g), fabsf(r)) + 1e-4f;
}

// ---------------------------------------------------------------------------

int main(int argc, char** argv) {
    for (int i = 1; i < argc; ++i) if (!strcmp(argv[i], "-maxstep") && i + 1 < argc) g_maxstep = atoi(argv[++i]);
    const Cfg c;
    const int E = c.n_embd, HD = c.hd, HQ = c.hq, HKV = c.hkv;
    const int DQ = HQ * HD, DKV = HKV * HD, GRP = HQ / HKV;

    printf("=== Phase 3 Attention Block Integration Test (gfx1151 zero-copy GQA) ===\n");
    printf("  embd=%d hd=%d heads_q=%d heads_kv=%d (group %d) | prefill pos 0..3 then decode at pos 4\n",
           E, HD, HQ, HKV, GRP);

    // ---------------- weights & norm scale ----------------
    QuantMat Wq = make_quant_mat(E,   DQ,  11);
    QuantMat Wk = make_quant_mat(E,   DKV, 22);
    QuantMat Wv = make_quant_mat(E,   DKV, 33);
    QuantMat Wo = make_quant_mat(DQ,  E,   44);

    // ---------------- one pinned mapping holding every buffer ----------------
    std::vector<size_t> szs = {
        (size_t)E     * (DQ/2),            // 0  wq_p   Wq packed bytes
        (size_t)E     * (DQ/16),           // 1  wq_s   Wq scales
        (size_t)E     * (DKV/2),           // 2  wk_p
        (size_t)E     * (DKV/16),          // 3  wk_s
        (size_t)E     * (DKV/2),           // 4  wv_p
        (size_t)E     * (DKV/16),          // 5  wv_s
        (size_t)DQ    * (E/2),             // 6  wo_p
        (size_t)DQ    * (E/16),            // 7  wo_s
        (size_t)CTX   * E     * 2,         // 8  x      [pos x embd] f16 token inputs
        (size_t)CTX   * E     * 2,         // 9  hbuf   rmsnorm out by position
        (size_t)CTX   * DQ    * 2,         // 10 q      q projection by position
        (size_t)CTX   * DKV   * 2,         // 11 kbuf   k projection by position
        (size_t)CTX   * DKV   * 2,         // 12 vbuf   v projection by position
        (size_t)E              * 2,        // 13 gamma  [embd] f16
        (size_t)HQ *MAXRUN*HD*2,           // 14 qh     staged per-head q [head][run-tok][dim] f16; rope in place
        (size_t)HKV*MAXRUN*HD*2,           // 15 kh     staged per-head k; rope in place
        (size_t)MAXRUN*DKV*2,              // 16 rks    token-major roped-k for kv_store source (run offset 0..)
        (size_t)HQ *MAXRUN*(HD/2)*4,       // 17 tabqc  rope cos tables [head][run-tok][pair]
        (size_t)HQ *MAXRUN*(HD/2)*4,       // 18 tabqs
        (size_t)HKV*MAXRUN*(HD/2)*4,       // 19 tabkc
        (size_t)HKV*MAXRUN*(HD/2)*4,       // 20 tabks
        (size_t)HQ *MAXRUN*CTX*4,          // 21 sc     scores/probs [head][run-tok][pos] f32
        (size_t)HQ *MAXRUN*CTX*2,          // 22 p16    fp16 probs copy for out-gemm input
        (size_t)HD *CTX*2,                 // 23 kexpB  B operand of scores gemm: [d][pos] layout
        (size_t)CTX*HD*2,                  // 24 vexpB  B operand of out gemm:    [pos][d] layout
        (size_t)HQ *MAXRUN*HD*4,           // 25 oh     attention out [head][run-tok][dim] f32
        (size_t)CTX   * DQ    * 2,         // 26 oin    O-proj input by position f16
        (size_t)HQ *MAXRUN*4,              // 27 qpos   absolute query positions per softmax row [head][run-tok]
        (size_t)CTX   * E     * 4,         // 28 outC   final hidden rows (GPU) f32
        (size_t)CTX   * E     * 4 };       // 29 yref   CPU reference rows f32

    std::vector<size_t> cum(szs.size());
    size_t total = 0;
    for (size_t i = 0; i < szs.size(); ++i) { cum[i] = total; total += szs[i]; }

    int fd = open("/dev/shm/attn_verify.bin", O_RDWR | O_CREAT, 0666);
    if (fd < 0) { perror("open"); return 1; }
    ftruncate(fd, (off_t)total);
    void* base = mmap(NULL, total, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (base == MAP_FAILED) { perror("mmap"); close(fd); return 1; }
    g_attn_fd = fd; g_attn_base = base; g_attn_total = total;   // exposed for fresh re-mmap probes in run_scenario
    memset(base, 0, total);

    #define BP(i) ((void*)((char*)base + cum[i]))
    unsigned char *wq_p=(unsigned char*)BP(0), *wq_s=(unsigned char*)BP(1),
                  *wk_p=(unsigned char*)BP(2), *wk_s=(unsigned char*)BP(3),
                  *wv_p=(unsigned char*)BP(4), *wv_s=(unsigned char*)BP(5),
                  *wo_p=(unsigned char*)BP(6), *wo_s=(unsigned char*)BP(7);
    __half *x     = (__half*)BP(8),   *hbuf = (__half*)BP(9),
           *q     = (__half*)BP(10),  *kbuf = (__half*)BP(11),
           *vbuf  = (__half*)BP(12),  *gamma= (__half*)BP(13);
    __half *qh    = (__half*)BP(14);
    __half *kh    = (__half*)BP(15);
    __half *rks   = (__half*)BP(16);   // token-major roped-k staging [MAXRUN x DKV] at run offset 0
    float  *tabqc = (float*)BP(17),   *tabqs = (float*)BP(18),
           *tabkc = (float*)BP(19),   *tabks = (float*)BP(20);
    float  *sc    = (float*)BP(21);
    __half *p16   = (__half*)BP(22);
    __half *kexpB = (__half*)BP(23);   // [HD][CTX]: B[d][pos] of scores gemm
    __half *vexpB = (__half*)BP(24);   // [CTX][HD]: B[pos][d] of out gemm
    float  *oh    = (float*)BP(25);
    __half *oin   = (__half*)BP(26);
    int    *qpos  = (int*)BP(27);
    float  *outC  = (float*)BP(28);
    float  *yref  = (float*)BP(29);
    #undef BP

    memcpy(wq_p, Wq.packed.data(), Wq.packed.size()); memcpy(wq_s, Wq.scales.data(), Wq.scales.size());
    memcpy(wk_p, Wk.packed.data(), Wk.packed.size()); memcpy(wk_s, Wk.scales.data(), Wk.scales.size());
    memcpy(wv_p, Wv.packed.data(), Wv.packed.size()); memcpy(wv_s, Wv.scales.data(), Wv.scales.size());
    memcpy(wo_p, Wo.packed.data(), Wo.packed.size()); memcpy(wo_s, Wo.scales.data(), Wo.scales.size());

    g_rng_state = 5;
    for (int i = 0; i < CTX * E; ++i) x[i] = __half(rndf(-3.0f, 3.0f));     // token inputs by position
    g_rng_state = 77;
    for (int i = 0; i < E; ++i) gamma[i] = __half(rndf(0.5f, 1.5f));

    HIP_CHECK(hipHostRegister(base, total, hipHostRegisterDefault), "pin main mapping");

    // Device views of the main mapping (identity on UMA).
    void *dw_q=nullptr,*dw_qs=nullptr,*dkp=nullptr,*dks=nullptr,*dvp=nullptr,*dvs=nullptr,*dop=nullptr,*dops=nullptr;
    void *dx=nullptr,*dh=nullptr,*dq=nullptr,*dkb=nullptr,*dvv=nullptr,*dg=nullptr;
    void *dqh=nullptr,*dkh=nullptr,*drks=nullptr;
    void *dtqc=nullptr,*dtqs=nullptr,*dtkc=nullptr,*dtks=nullptr;
    void *dsc=nullptr,*dp16=nullptr,*dkb2=nullptr,*dvb2=nullptr,*doh=nullptr,*doin=nullptr,*dqpos=nullptr,*doc=nullptr;
    HIP_CHECK(hipHostGetDevicePointer(&dw_q, wq_p, 0), "wq_p");   HIP_CHECK(hipHostGetDevicePointer(&dw_qs, wq_s, 0), "wq_s");
    HIP_CHECK(hipHostGetDevicePointer(&dkp, wk_p, 0), "wk_p");    HIP_CHECK(hipHostGetDevicePointer(&dks, wk_s, 0), "wk_s");
    HIP_CHECK(hipHostGetDevicePointer(&dvp, wv_p, 0), "wv_p");    HIP_CHECK(hipHostGetDevicePointer(&dvs, wv_s, 0), "wv_s");
    HIP_CHECK(hipHostGetDevicePointer(&dop, wo_p, 0), "wo_p");    HIP_CHECK(hipHostGetDevicePointer(&dops, wo_s, 0), "wo_s");
    HIP_CHECK(hipHostGetDevicePointer(&dx, x, 0), "x");           HIP_CHECK(hipHostGetDevicePointer(&dh, hbuf, 0), "hbuf");
    HIP_CHECK(hipHostGetDevicePointer(&dq, q, 0), "q");           HIP_CHECK(hipHostGetDevicePointer(&dkb, kbuf, 0), "kbuf");
    HIP_CHECK(hipHostGetDevicePointer(&dvv, vbuf, 0), "vbuf");    HIP_CHECK(hipHostGetDevicePointer(&dg, gamma, 0), "gamma");
    HIP_CHECK(hipHostGetDevicePointer(&dqh, qh, 0), "qh");        HIP_CHECK(hipHostGetDevicePointer(&dkh, kh, 0), "kh");
    HIP_CHECK(hipHostGetDevicePointer(&drks, rks, 0), "rks");
    HIP_CHECK(hipHostGetDevicePointer(&dtqc, tabqc, 0), "tabqc"); HIP_CHECK(hipHostGetDevicePointer(&dtqs, tabqs, 0), "tabqs");
    HIP_CHECK(hipHostGetDevicePointer(&dtkc, tabkc, 0), "tabkc"); HIP_CHECK(hipHostGetDevicePointer(&dtks, tabks, 0), "tabks");
    HIP_CHECK(hipHostGetDevicePointer(&dsc, sc, 0), "sc");        HIP_CHECK(hipHostGetDevicePointer(&dp16, p16, 0), "p16");
    HIP_CHECK(hipHostGetDevicePointer(&dkb2, kexpB, 0), "kexpB"); HIP_CHECK(hipHostGetDevicePointer(&dvb2, vexpB, 0), "vexpB");
    HIP_CHECK(hipHostGetDevicePointer(&doh, oh, 0), "oh");        HIP_CHECK(hipHostGetDevicePointer(&doin, oin, 0), "oin");
    HIP_CHECK(hipHostGetDevicePointer(&dqpos, qpos, 0), "qpos");  HIP_CHECK(hipHostGetDevicePointer(&doc, outC, 0), "outC");

    {   // verify device==host address identity for every region of the main mapping
        struct PV { const char* n; const void* h; void* d; };
        PV pv[29] = {
            {"wq_p", wq_p, dw_q},   {"wq_s", wq_s, dw_qs},   {"wk_p", wk_p, dkp},     {"wk_s", wk_s, dks},
            {"wv_p", wv_p, dvp},    {"wv_s", wv_s, dvs},     {"wo_p", wo_p, dop},     {"wo_s", wo_s, dops},
            {"x", x, dx},           {"hbuf", hbuf, dh},      {"q", q, dq},             {"kbuf", kbuf, dkb},
            {"vbuf", vbuf, dvv},   {"gamma", gamma, dg},    {"qh", qh, dqh},          {"kh", kh, dkh},
            {"rks", rks, drks},    {"tabqc", tabqc, dtqc},   {"tabqs", tabqs, dtqs},  {"tabkc", tabkc, dtkc},
            {"tabks", tabks, dtks},{"sc", sc, dsc},         {"p16", p16, dp16},       {"kexpB", kexpB, dkb2},
            {"vexpB", vexpB, dvb2},{"oh", oh, doh},         {"oin", oin, doin},       {"qpos", qpos, dqpos},
            {"outC", outC, doc} };
        int bad = 0;
        for (auto& e : pv) if ((void*)e.h != e.d) { printf("    PTR IDENTITY FAIL: %s host=%p dev=%p\n", e.n, e.h, e.d); bad++; }
        printf("  pointer identity check: %s (%d regions)\n", bad ? "MISMATCHES FOUND" : "all identical", 29);
    }

    // KV cache manager over its own /dev/shm file (the engine's real store).
    kv_cache_config_t kcfg{1, CTX, c.n_embd, HD, HQ, HKV};
    inference_kv_cache_t* kvc = kv_cache_create("/dev/shm/attn_verify_kvc.bin", &kcfg);
    if (!kvc) { fprintf(stderr, "kv_cache_create failed\n"); return 1; }
    printf("  KV cache: %zu bytes pinned in /dev/shm (zero-copy identity=%s)\n",
           kv_cache_total_bytes(kvc), kv_cache_zero_copy_ok(kvc) ? "yes" : "NO(!)");

    // ---------------- kernel modules ----------------
    auto load_mod = [](const char* co, const char* sym, hipModule_t& m, hipFunction_t& f) -> bool {
        int kf = open(co, O_RDONLY); if (kf < 0) { perror(co); return false; }
        struct stat st; fstat(kf, &st);
        void* cd = mmap(NULL, st.st_size, PROT_READ, MAP_PRIVATE, kf, 0);
        bool ok = true;
        if (hipModuleLoadData(&m, cd) != hipSuccess || hipModuleGetFunction(&f, m, sym) != hipSuccess) ok = false;
        munmap(cd, st.st_size); close(kf); return ok;
    };
    hipModule_t m_ln, m_g4, m_rope, m_kvs, m_g16, m_sm;
    hipFunction_t f_ln, f_g4, f_rope, f_kvs, f_g16, f_sm;
    HIP_CHECK(hipInit(0), "hipInit");
    if (!load_mod("layernorm.co",      "rmsnorm",          m_ln,  f_ln)  ||
        !load_mod("gemm_act_fp4.co",   "gemm_act_fp4",     m_g4,  f_g4)  ||
        !load_mod("rope_embedding.co", "rope_rotate_half", m_rope,f_rope) ||
        !load_mod("kv_cache_store.co", "kv_store",         m_kvs, f_kvs) ||
        !load_mod("fp16_gemm.co",      "fp16_gemm",        m_g16, f_g16) ||
        !load_mod("softmax_masked.co", "softmax_masked",   m_sm,  f_sm)) { fprintf(stderr,"module load failed\n"); return 1; }

    long total_failures = 0;
    const double log_theta = log((double)c.theta);

    // ========================================================================
    // Scenario: T tokens at consecutive positions S..S+T-1. CPU reference and GPU chain both start from
    // the current cache contents (snapshot into local history arrays). Final hidden rows are compared
    // position-by-position, and the stored KV rows must match the reference's values bit-for-bit.
    // ========================================================================
    auto run_scenario = [&](const char* name, int S, int T) -> bool {
        printf("--- %s: %d token(s) at position%s %d..%d ---\n", name, T, T > 1 ? "s" : "", S, S + T - 1);

        float eps = c.eps;
        float scale = 1.0f / sqrtf((float)HD);

        std::vector<__half> hist_k((size_t)CTX * DKV), hist_v((size_t)CTX * DKV);   // roped K / raw V as stored
        for (int p = 0; p < CTX; ++p) {
            memcpy(hist_k.data() + (size_t)p * DKV, kv_cache_k_row_const(kvc, 0, p), DKV * sizeof(__half));
            memcpy(hist_v.data() + (size_t)p * DKV, kv_cache_v_row_const(kvc, 0, p), DKV * sizeof(__half));
        }

        // ============================ CPU reference ============================
        std::vector<__half> refq((size_t)CTX*DQ), refk((size_t)CTX*DKV), refv((size_t)CTX*DKV);   // per-position projection refs
        for (int t = 0; t < T; ++t) {
            int p = S + t;

            float ssq = 0.0f;                                                   // rmsnorm -> effective fp16 values
            for (int i2 = 0; i2 < E; ++i2) { float v2 = (float)x[(size_t)p*E+i2]; ssq += v2*v2; }
            float inv = 1.0f / sqrtf(ssq / (float)E + eps);
            std::vector<float> hf(E);
            for (int i2 = 0; i2 < E; ++i2) hf[i2] = (float)__half((float)x[(size_t)p*E+i2] * inv * (float)gamma[i2]);

            std::vector<__half> qv(DQ), kv_(DKV), vv(DKV);                      // projections, rounded to fp16 as stored
            for (int j = 0; j < DQ;  ++j) { float s=0; for (int k2=0;k2<E;++k2) s += hf[k2]*Wq.true_v[(size_t)j*Wq.K+k2];   qv[j]=__half(s); }
            for (int j = 0; j < DKV; ++j) { float s=0; for (int k2=0;k2<E;++k2) s += hf[k2]*Wk.true_v[(size_t)j*Wk.K+k2];  kv_[j]=__half(s); }
            for (int j = 0; j < DKV; ++j) { float s=0; for (int k2=0;k2<E;++k2) s += hf[k2]*Wv.true_v[(size_t)j*Wv.K+k2];  vv[j]=__half(s); }

            memcpy(refq.data()+(size_t)p*DQ,  qv.data(), DQ*sizeof(__half));      // keep for later GPU-vs-ref intermediate checks
            memcpy(refk.data()+(size_t)p*DKV, kv_.data(), DKV*sizeof(__half));
            memcpy(refv.data()+(size_t)p*DKV, vv.data(), DKV*sizeof(__half));

            auto rope_head = [&](const __half* src, int off, std::vector<float>& dst) {    // rotate-half over one head at position p
                for (int d = 0; d < HD/2; ++d) {
                    double fr = exp(-2.0 * ((double)d / (double)HD) * log_theta);
                    float co = cosf((float)((double)p * fr)), sn = sinf((float)((double)p * fr));
                    float a = (float)src[off + d], b = (float)src[off + HD/2 + d];
                    dst[d]          = a*co - b*sn;
                    dst[HD/2 + d]   = b*co + a*sn;
                }
            };
            std::vector<std::vector<float>> rqh(HQ, std::vector<float>(HD)), rkh(HKV, std::vector<float>(HD));
            for (int h = 0; h < HQ;  ++h) rope_head(qv.data(),  h*HD,  rqh[h]);
            for (int h = 0; h < HKV; ++h) rope_head(kv_.data(), h*HD,  rkh[h]);

            for (int kh_ = 0; kh_ < HKV; ++kh_)                                    // store roped K + raw V BEFORE attending: token sees itself
                for (int d = 0; d < HD; ++d) hist_k[(size_t)p*DKV + kh_*HD + d] = (__half)rkh[kh_][d];   // one fp16 round as on GPU
            for (int j = 0; j < DKV; ++j)                                          // vv already holds the stored fp16 values
                hist_v[(size_t)p*DKV+j] = vv[j];

            std::vector<float> attn(DQ, 0.0f);                                    // attention over [0..p], per query head
            for (int h = 0; h < HQ; ++h) {
                int kh_ = h / GRP;                                                 // GQA: kv head serving this q head
                float m = -INFINITY;
                std::vector<float> e(CTX, 0.0f);
                for (int pp = 0; pp <= p; ++pp) {
                    float s2 = 0.0f;
                    for (int d = 0; d < HD; ++d) s2 += rqh[h][d] * (float)hist_k[(size_t)pp*DKV + kh_*HD + d];
                    e[pp] = s2 * scale;                                            // fold the attention scale before max/exp
                    if (e[pp] > m) m = e[pp];
                }
                float sum = 0.0f;
                for (int pp = 0; pp <= p; ++pp) { e[pp] = expf(e[pp] - m); sum += e[pp]; }
                float invs = 1.0f / sum;
                for (int d = 0; d < HD; ++d) {
                    float acc = 0.0f;
                    for (int pp = 0; pp <= p; ++pp) acc += e[pp]*invs * (float)hist_v[(size_t)pp*DKV + kh_*HD + d];
                    attn[h*HD+d] = acc;
                }
            }

            std::vector<__half> oinv(DQ);                                          // O-proj input rounded to fp16 as stored, then final projection
            for (int j = 0; j < DQ; ++j) oinv[j] = __half(attn[j]);
            for (int j2 = 0; j2 < E; ++j2) {
                float s3 = 0.0f;
                for (int k2 = 0; k2 < DQ; ++k2) s3 += (float)oinv[k2] * Wo.true_v[(size_t)j2*Wo.K+k2];
                yref[(size_t)p*E+j2] = s3;
            }
        }

        // ============================ GPU chain ============================
        __half* h_s   = hbuf + (size_t)S * E;
        __half* q_s   = q    + (size_t)S * DQ;
        __half* k_s   = kbuf + (size_t)S * DKV;
        __half* v_s   = vbuf + (size_t)S * DKV;

        if (g_maxstep >= 1) {   // 1) RMSNorm over the batch rows
            void* src=(void*)((char*)dx + (size_t)S*E*sizeof(__half)), *dst=(void*)((char*)dh + (size_t)S*E*sizeof(__half));
            void* ar[] = { &src, &dg, &dst, (void*)&T, (void*)&E, (void*)&eps };
            HIP_CHECK(hipModuleLaunchKernel(f_ln, T, 1, 1, 32, 1, 1, 0, 0, ar, nullptr), "rmsnorm");
        } else printf("    [maxstep] skip step 1\n");

        if (g_maxstep >= 2) {   // 2) Q/K/V projections from Codebook10 weights
            int M=T, K=E;
            void* dhv=(void*)h_s;
            { int N=DQ;  void* cq=(void*)q_s; void* ar[] = { &dhv,&dw_q,&dw_qs,&cq,(void*)&M,(void*)&N,(void*)&K };
              HIP_CHECK(hipModuleLaunchKernel(f_g4, (T+15)/16, (DQ+15)/16, 1, 32, 1, 1, 0, 0, ar, nullptr), "proj-q"); }   // grid.x=M rows, grid.y=N cols
            { int N=DKV; void* ck=(void*)k_s; void* ar[] = { &dhv,&dkp,&dks,&ck,(void*)&M,(void*)&N,(void*)&K };
              HIP_CHECK(hipModuleLaunchKernel(f_g4, (T+15)/16, (DKV+15)/16, 1, 32, 1, 1, 0, 0, ar, nullptr), "proj-k"); }   // grid.x=M rows, grid.y=N cols
            { int N=DKV; void* cv=(void*)v_s; void* ar[] = { &dhv,&dvp,&dvs,&cv,(void*)&M,(void*)&N,(void*)&K };
              HIP_CHECK(hipModuleLaunchKernel(f_g4, (T+15)/16, (DKV+15)/16, 1, 32, 1, 1, 0, 0, ar, nullptr), "proj-v"); }   // grid.x=M rows, grid.y=N cols
            HIP_CHECK(hipDeviceSynchronize(), "post-proj sync");   // staging copies below read GPU outputs
        } else printf("    [maxstep] skip step 2\n");

        // Snapshot projection outputs immediately after their launches: lets us tell "kernel computed wrong"
        // apart from "something overwrote the buffers later".
        std::vector<__half> snap_q((size_t)CTX*DQ), snap_k((size_t)CTX*DKV), snap_v((size_t)CTX*DKV);
        memcpy(snap_q.data(), q, sizeof(__half)*(size_t)CTX*DQ);
        memcpy(snap_k.data(), kbuf, sizeof(__half)*(size_t)CTX*DKV);
        memcpy(snap_v.data(), vbuf, sizeof(__half)*(size_t)CTX*DKV);

        if (g_maxstep >= 3) {   // 3) Stage per-head views from the token-major projection slices (CPU reshape inside the mapping),
                                //     build rope tables from absolute positions S+t, then rope each head in place.
            for (int h = 0; h < HQ; ++h)
                for (int t = 0; t < T; ++t)
                    memcpy(qh + ((size_t)h*MAXRUN + t)*HD, q_s + (size_t)t*DQ + h*HD, HD*sizeof(__half));
            for (int kh_ = 0; kh_ < HKV; ++kh_)
                for (int t = 0; t < T; ++t)
                    memcpy(kh + ((size_t)kh_*MAXRUN + t)*HD, k_s + (size_t)t*DKV + kh_*HD, HD*sizeof(__half));

            for (int h = 0; h < HQ; ++h)
                for (int t = 0; t < T; ++t) {
                    int p = S + t;
                    float* cr = tabqc + ((size_t)h*MAXRUN + t)*(HD/2);
                    float* sr = tabqs + ((size_t)h*MAXRUN + t)*(HD/2);
                    for (int d2 = 0; d2 < HD/2; ++d2) {
                        double fr = exp(-2.0 * ((double)d2 / (double)HD) * log_theta);
                        cr[d2] = cosf((float)((double)p * fr));
                        sr[d2] = sinf((float)((double)p * fr));
                    }
                }
            for (int h = 0; h < HKV; ++h)
                for (int t = 0; t < T; ++t) {
                    int p = S + t;
                    float* cr = tabkc + ((size_t)h*MAXRUN + t)*(HD/2);
                    float* sr = tabks + ((size_t)h*MAXRUN + t)*(HD/2);
                    for (int d2 = 0; d2 < HD/2; ++d2) {
                        double fr = exp(-2.0 * ((double)d2 / (double)HD) * log_theta);
                        cr[d2] = cosf((float)((double)p * fr));
                        sr[d2] = sinf((float)((double)p * fr));
                    }
                }

            {   // per-head rope launches over contiguous head slices (in place: x == y)
                int hd2 = HD / 2, rows = T;
                for (int h = 0; h < HQ; ++h) {
                    void* px=(void*)((char*)dqh + ((size_t)h*MAXRUN)*HD*sizeof(__half));
                    void* pc=(void*)((char*)dtqc+ ((size_t)h*MAXRUN)*(HD/2)*sizeof(float));
                    void* ps=(void*)((char*)dtqs+ ((size_t)h*MAXRUN)*(HD/2)*sizeof(float));
                    void* ar[] = { &px, &pc, &ps, &px, (void*)&rows, (void*)&hd2 };
                    HIP_CHECK(hipModuleLaunchKernel(f_rope, rows, 1, 1, 32, 1, 1, 0, 0, ar, nullptr), "rope-q");
                }
                for (int h = 0; h < HKV; ++h) {
                    void* px=(void*)((char*)dkh + ((size_t)h*MAXRUN)*HD*sizeof(__half));
                    void* pc=(void*)((char*)dtkc+ ((size_t)h*MAXRUN)*(HD/2)*sizeof(float));
                    void* ps=(void*)((char*)dtks+ ((size_t)h*MAXRUN)*(HD/2)*sizeof(float));
                    void* ar[] = { &px, &pc, &ps, &px, (void*)&rows, (void*)&hd2 };
                    HIP_CHECK(hipModuleLaunchKernel(f_rope, rows, 1, 1, 32, 1, 1, 0, 0, ar, nullptr), "rope-k");
                }
            }
            HIP_CHECK(hipDeviceSynchronize(), "post-rope sync");   // rks reshape below reads rope'd kh
        } else printf("    [maxstep] skip step 3\n");

        if (g_maxstep >= 4) {   // 4) Reshape roped K into token-major staging + KV store into the managed cache at S..S+T-1
            for (int t = 0; t < T; ++t)
                for (int kh_ = 0; kh_ < HKV; ++kh_)
                    memcpy(rks + (size_t)t*DKV + kh_*HD, kh + ((size_t)kh_*MAXRUN + t)*HD, HD*sizeof(__half));
            {
                __half* kdst_base = kv_cache_k_row(kvc, 0, 0);      // region base; kernel adds pos_start inside
                __half* vdst_base = kv_cache_v_row(kvc, 0, 0);
                void* pk=(void*)drks;                                // run rows start at offset 0 in rks
                void* pv=(void*)((char*)dvv + (size_t)S*DKV*sizeof(__half));   // raw V slice is already token-major at S
                void* pdk=(void*)kdst_base;                         // identity on UMA for registered pages
                void* pdv=(void*)vdst_base;
                int toks=T, dkv=DKV, pos_start=S, n_ctx=CTX;
                void* ar[] = { &pk, &pv, &pdk, &pdv, (void*)&toks, (void*)&dkv, (void*)&pos_start, (void*)&n_ctx };
                HIP_CHECK(hipModuleLaunchKernel(f_kvs, 8, 1, 1, 32, 1, 1, 0, 0, ar, nullptr), "kv-store");
                HIP_CHECK(hipDeviceSynchronize(), "post-store sync");
            }
        } else printf("    [maxstep] skip step 4\n");

        if (g_maxstep >= 5) {   // 5) Per-head attention over GQA-expanded history columns read from the FRESH cache contents.
            const __half* kc_now = kv_cache_k_row_const(kvc, 0, 0);   // [CTX x DKV] region rows
            const __half* vc_now = kv_cache_v_row_const(kvc, 0, 0);

            for (int h = 0; h < HQ; ++h) {
                int kh_ = h / GRP;

                for (int d = 0; d < HD; ++d)                            // rebuild both GEMM-B operands for this head
                    for (int pp = 0; pp < CTX; ++pp) {
                        kexpB[(size_t)d*CTX+pp] = kc_now[(size_t)pp*DKV + kh_*HD + d];   // B[d][pos] layout
                        vexpB[(size_t)pp*HD+d]   = vc_now[(size_t)pp*DKV + kh_*HD + d];   // B[pos][d] layout
                    }

                float* sc_h    = sc     + (size_t)h*(MAXRUN*CTX);       // rows t=0..T-1 at offset t*CTX
                __half* p16_h  = p16    + (size_t)h*(MAXRUN*CTX);
                float* oh_h    = oh     + (size_t)h*(MAXRUN*HD);
                void* qpos_h   = (void*)(qpos + h*MAXRUN);              // absolute query positions of this head's rows

                for (int t = 0; t < T; ++t) qpos[h*MAXRUN + t] = S + t;

                {   // scores [T x CTX] = roped q head @ K_exp^T  (A=[T x HD] contiguous per-head slice)
                    void* pa=(void*)((char*)dqh + (size_t)h*(MAXRUN*HD)*sizeof(__half));
                    int Mq=T, Nc=CTX, Ki=HD;
                    void* ps=(void*)sc_h;
                    void* ar[] = { &pa, &dkb2, &ps, (void*)&Mq, (void*)&Nc, (void*)&Ki };
                    HIP_CHECK(hipModuleLaunchKernel(f_g16, 1, (Nc+15)/16, 1, 32, 1, 1, 0, 0, ar, nullptr), "scores-gemm");
                }

                {   // softmax over this head's T rows: causal ceiling from absolute positions, valid-KV bound S+T
                    int nr=T, nc=CTX, nvalid=S+T;
                    float sc_sm = scale;
                    void* ps=(void*)sc_h;
                    void* ar[] = { &ps, (void*)&nr, (void*)&nc, (void*)&sc_sm, &qpos_h, (void*)&nvalid };
                    HIP_CHECK(hipModuleLaunchKernel(f_sm, nr, 1, 1, 32, 1, 1, 0, 0, ar, nullptr), "softmax");
                }
                HIP_CHECK(hipDeviceSynchronize(), "post-softmax sync");   // p16 copy below reads scores

                for (int t = 0; t < T; ++t)                             // probs -> fp16 copy for the out-gemm A operand
                    for (int pp = 0; pp < CTX; ++pp) p16_h[t*CTX+pp] = __half(sc_h[t*CTX+pp]);

                {   // out [T x HD] = probs @ V_exp
                    void* pa=(void*)p16_h;
                    int Mo=T, No=HD, Ki2=CTX;
                    void* po=(void*)oh_h;
                    void* ar[] = { &pa, &dvb2, &po, (void*)&Mo, (void*)&No, (void*)&Ki2 };
                    HIP_CHECK(hipModuleLaunchKernel(f_g16, (Mo+15)/16, (No+15)/16, 1, 32, 1, 1, 0, 0, ar, nullptr), "out-gemm");
                }
            }
        } else printf("    [maxstep] skip step 5\n");

        if (g_maxstep >= 6) {   // 6) Flatten per-head attention outputs into the token-major O-proj input (fp16 as stored).
            for (int t = 0; t < T; ++t) {
                int p = S + t;
                for (int h = 0; h < HQ; ++h)
                    for (int d = 0; d < HD; ++d) oin[(size_t)p*DQ + h*HD + d] = __half(oh[((size_t)h*MAXRUN+t)*HD + d]);
            }
            HIP_CHECK(hipDeviceSynchronize(), "post-outgemm sync");   // oin flatten above reads oh of every head
        } else printf("    [maxstep] skip step 6\n");

        if (g_maxstep >= 7) {   // 7) Output projection -> final hidden rows.
            int M=T, K=DQ;
            void* ca=(void*)(oin + (size_t)S*DQ);
            int N=E;
            void* cc=(void*)(outC + (size_t)S*E);
            void* ar[] = { &ca, &dop, &dops, &cc, (void*)&M, (void*)&N, (void*)&K };
            HIP_CHECK(hipModuleLaunchKernel(f_g4, 1, (N+15)/16, 1, 32, 1, 1, 0, 0, ar, nullptr), "proj-o");
        } else printf("    [maxstep] skip step 7\n");
        HIP_CHECK(hipDeviceSynchronize(), "scenario sync");

        // ---- intermediate sanity: GPU projection buffers vs reference (diagnostic only) ----
        long iq=0, ik=0, iv=0; float imax=0.0f; long shown=0;
        auto dump = [&](const char* tag, int p, int j, const void* gbuf, size_t stride_elems, float r) {
            if (shown >= 8) return; shown++;
            printf("    SAMPLE %s pos%d[%d]: gpu=%.6g ref=%.6g\n", tag, p, j, (float)((const __half*)gbuf)[(size_t)p*stride_elems+j], r);
        };
        for (int t = 0; t < T; ++t) { int p = S + t;
            for (int j = 0; j < DQ;  ++j) { float g=(float)q[(size_t)p*DQ+j],   r=(float)refq[(size_t)p*DQ+j];   float d=fabsf(g-r); if (d>imax) imax=d; if (bad_result(g,r)) { iq++; dump("q",p,j,q,DQ,r); } }
            for (int j = 0; j < DKV; ++j) { float g=(float)kbuf[(size_t)p*DKV+j],r=(float)refk[(size_t)p*DKV+j]; float d=fabsf(g-r); if (d>imax) imax=d; if (bad_result(g,r)) { ik++; dump("k",p,j,kbuf,DKV,r); } }
            for (int j = 0; j < DKV; ++j) { float g=(float)vbuf[(size_t)p*DKV+j],r=(float)refv[(size_t)p*DKV+j]; float d=fabsf(g-r); if (d>imax) imax=d; if (bad_result(g,r)) { iv++; dump("v",p,j,vbuf,DKV,r); } }
        }
        printf("    projections: q_bad=%ld k_bad=%ld v_bad=%ld max_mid_diff=%.4f\n", iq, ik, iv, imax);

        // Were the projection buffers overwritten AFTER their launches? Compare live vs as-computed snapshot.
        {
            long dq_=0, dk_=0, dv_=0; float ex_q=0, ex_k=0;
            for (size_t i = 0; i < snap_q.size(); ++i) if (q[i] != snap_q[i]) dq_++;
            for (size_t i = 0; i < snap_k.size(); ++i) if (kbuf[i] != snap_k[i]) dk_++;
            for (size_t i = 0; i < snap_v.size(); ++i) if (vbuf[i] != snap_v[i]) dv_++;
            for (int p2 = S; p2 < S+T && ex_q==0; ++p2)
                for (int j2 = 0; j2 < DQ && ex_q==0; ++j2)
                    if (q[(size_t)p2*DQ+j2] != snap_q[(size_t)p2*DQ+j2]) { ex_q=(float)q[(size_t)p2*DQ+j2]; printf("    OVERWRITE q pos%d[%d]: as-computed=%.6g now=%.6g\n", p2, j2, (float)snap_q[(size_t)p2*DQ+j2], ex_q); }
            for (int p2 = S; p2 < S+T && ex_k==0; ++p2)
                for (int j2 = 0; j2 < DKV && ex_k==0; ++j2)
                    if (kbuf[(size_t)p2*DKV+j2] != snap_k[(size_t)p2*DKV+j2]) { ex_k=(float)kbuf[(size_t)p2*DKV+j2]; printf("    OVERWRITE k pos%d[%d]: as-computed=%.6g now=%.6g\n", p2, j2, (float)snap_k[(size_t)p2*DKV+j2], ex_k); }
            printf("    post-launch drift: q=%ld/%zu k=%ld/%zu v=%ld/%zu cells changed after projection\n",
                   dq_, snap_q.size(), dk_, snap_k.size(), dv_, snap_v.size());

            // Fresh re-mmap of the same file at this instant: does a brand-new mapping see different bytes?
            void* base2 = mmap(NULL, g_attn_total, PROT_READ, MAP_SHARED, g_attn_fd, 0);
            if (base2 != MAP_FAILED) {
                const __half* q2  = (const __half*)((char*)base2 + ((const char*)q    - (const char*)g_attn_base));
                const __half* kb2 = (const __half*)((char*)base2 + ((const char*)kbuf - (const char*)g_attn_base));
                printf("    FRESH-MAP probe: q[pos S][0..3]= %.6g %.6g %.6g %.6g | k[pos S][0..3]= %.6g %.6g %.6g %.6g\n",
                       (float)q2[0],(float)q2[1],(float)q2[2],(float)q2[3],
                       (float)kb2[0],(float)kb2[1],(float)kb2[2],(float)kb2[3]);
                long diffcells = 0;
                for (size_t i = 0; i < snap_q.size(); ++i) if (q[i] != q2[i]) diffcells++;
                printf("    fresh-map vs old-view q-region: %ld/%zu cells differ\n", diffcells, snap_q.size());
            }
        }

        // PROBE-A: recompute the q-projection on CPU at pos S in BOTH orientations:
        //   correct: C[m][j] = sum_k A[m][k] * W_line[j][k], effective values stored as true_v[j*K + k]
        //   (my earlier "ref" indexed true_v[k*DQ + j] -- check which one the kernel actually produced)
        {
            int p = S;
            float cx[4] = {0,0,0,0}, ch[4] = {0,0,0,0};
            for (int k2 = 0; k2 < E; ++k2) {
                float xv = (float)x[(size_t)p*E+k2], hv = (float)hbuf[(size_t)p*E+k2];
                for (int j = 0; j < 4; ++j) { cx[j] += xv * Wq.true_v[(size_t)j*E+k2]; ch[j] += hv * Wq.true_v[(size_t)j*E+k2]; }
            }
            printf("    PROBE-A pos%d gpu-q[0..3]= %.6g %.6g %.6g %.6g\n", p,
                   (float)q[p*DQ+0], (float)q[p*DQ+1], (float)q[p*DQ+2], (float)q[p*DQ+3]);
            printf("                 x[0..3]=   %.6g %.6g %.6g %.6g\n",
                   (float)x[p*E+0], (float)x[p*E+1], (float)x[p*E+2], (float)x[p*E+3]);
            printf("                 from-x-corr  = %.6g %.6g %.6g %.6g\n", cx[0],cx[1],cx[2],cx[3]);
            printf("                 from-hub-corr= %.6g %.6g %.6g %.6g\n", ch[0],ch[1],ch[2],ch[3]);
        }

        // also dump a few rmsnorm outputs vs the CPU-computed effective values for position S
        {
            int p = S;
            float ssq=0; for (int i2=0;i2<E;++i2){ float v2=(float)x[(size_t)p*E+i2]; ssq+=v2*v2; }
            float inv = 1.0f / sqrtf(ssq/(float)E + c.eps);
            for (int j = 0; j < 8; ++j)
                printf("    hbuf pos%d[%d]: gpu=%.6g ref=%.6g\n", p, j, (float)hbuf[(size_t)p*E+j],
                       (float)__half((float)x[(size_t)p*E+j]*inv*(float)gamma[j]));
        }

        // ---------------- comparison: final hidden rows ----------------
        long fails_here = 0; float maxd = 0.0f;
        for (int t = 0; t < T; ++t) {
            int p = S + t;
            for (int j = 0; j < E; ++j) {
                float g = outC[(size_t)p*E+j], r = yref[(size_t)p*E+j];
                float d = fabsf(g-r); if (d > maxd) maxd = d;
                if (bad_result(g,r)) {
                    if (fails_here < 6) printf("    MISMATCH pos %d [%d]: gpu=%.4f ref=%.4f\n", p, j, g, r);
                    fails_here++;
                }
            }
        }
        total_failures += fails_here;
        printf("    final rows: elements=%d failures=%ld max_abs_diff=%.5f\n", T*E, fails_here, maxd);

        // KV store correctness: re-read the stored rows and require bit-identical halves vs reference.
        long store_bad = 0;
        for (int t = 0; t < T; ++t) {
            int p = S + t;
            const __half* ck_ = kv_cache_k_row_const(kvc, 0, p);
            const __half* cv_ = kv_cache_v_row_const(kvc, 0, p);
            for (int j = 0; j < DKV; ++j) {
                if ((float)ck_[j] != (float)hist_k[(size_t)p*DKV+j]) store_bad++;
                if ((float)cv_[j] != (float)hist_v[(size_t)p*DKV+j]) store_bad++;
            }
        }
        if (store_bad > 0) { total_failures += store_bad; printf("    KV STORE MISMATCH: %ld values differ from reference\n", store_bad); }
        else               { printf("    KV store: bit-exact vs reference (%d values)\n", 2*T*DKV); }

        // PROBE-B: re-launch proj-q ALONE now (same A bytes, same weights) into outC-as-scratch.
        // Correct => the earlier wrong result was state/order-dependent; wrong again => data or module issue.
        {
            int M=T, N=DQ, K=E;
            void* ca=(void*)((char*)dh + (size_t)S*E*sizeof(__half));
            void* cc=(void*)(outC + (size_t)S*E);   // scratch reuse: final-row comparison already done above
            void* ar[] = { &ca, &dw_q, &dw_qs, &cc, (void*)&M, (void*)&N, (void*)&K };
            HIP_CHECK(hipModuleLaunchKernel(f_g4, (T+15)/16, (N+15)/16, 1, 32, 1, 1, 0, 0, ar, nullptr), "probeB");
            HIP_CHECK(hipDeviceSynchronize(), "probeB sync");
            long bb=0; float mx=0;
            for (int t = 0; t < T; ++t)
                for (int j = 0; j < DQ; ++j) {
                    float g=outC[(size_t)(S+t)*E+j], r=(float)refq[(size_t)(S+t)*DQ+j];
                    float d=fabsf(g-r); if (d>mx) mx=d; if (bad_result(g,r)) bb++;
                }
            printf("    PROBE-B proj-q re-launched alone: bad=%ld/%d maxdiff=%.4f\n", bb, T*DQ, mx);
        }

        return true;   // per-scenario status already folded into total_failures above
    };

    // ============================ scenarios & summary ============================
    run_scenario("prefill", 0, 4);      // positions 0..3
    run_scenario("decode ", 4, 1);      // position 4, reads stored history through the cache manager

    kv_cache_free(kvc);
    HIP_CHECK(hipHostUnregister(base), "unpin main mapping");
    munmap(base, total); close(fd);

    if (total_failures == 0) {
        printf("\n  ATTENTION BLOCK INTEGRATION PASSED (full zero-copy GQA chain on gfx1151)\n\n");
        return 0;
    }
    printf("\n  ATTENTION BLOCK INTEGRATION FAILED (%ld element mismatches)\n\n", total_failures);
    return 1;
}

