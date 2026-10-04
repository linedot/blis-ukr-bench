// prefetch_test -- does a software prefetch reach this core's L1, and how far
// ahead must it go?
//
// Default: walks the 64-byte lines of a buffer in a random order, which no
// hardware prefetcher follows; each line holds the index of the next, so the
// walk is a pointer chase: no two misses overlap and each costs its full
// latency, on in-order and out-of-order cores alike. Then the same walk with a
// software prefetch of the line D steps ahead: once D steps take longer than
// the latency, every load should find its line in L1 -- if the prefetch does
// anything. "chase alone" is the latency of the level the buffer lives in.
//
// --copy: packing as BLIS packs A instead -- 8-row panels of a column-major
// matrix, one 64-byte line per column, four columns per iteration, loaded then
// stored -- with vector loads (vle64 at LMUL 2, VLEN >= 256; like the X60's
// dpackm) and with scalar loads and stores, each with the prefetch D columns
// ahead (D = 0: none). The chase has one miss in flight and scalar loads;
// packing has four, vector loads and stores.
//
// Prefetch kinds (--pf):
//            RISC-V                  x86            AArch64
//   l1       prefetch.r              prefetcht0     prfm pldl1keep
//   l2       ntl.p1 + prefetch.r     prefetcht1     prfm pldl2keep
//   l3       ntl.pall + prefetch.r   prefetcht2     prfm pldl3keep
//   nta      ntl.all + prefetch.r    prefetchnta    prfm pldl1strm
//   w        prefetch.w              prefetchw      prfm pstl1keep
//   load     ld x0 (value dropped)   mov            ldr
// RISC-V encodings are raw (Zicbop prefetch.r 0(rs1) = ori x0, rs1, 1;
// Zihintntl ntl.* = add x0, x0, x2/x3/x5): no -march needed, hints are no-ops
// on cores without the extension.
//
// usage: prefetch_test [options] [SIZE_KB...]      (default sizes 16 256 4096)
//   --pf KINDS    prefetches to compare, comma-separated (default l1,l2,load;
//                 --copy: l1)
//   --dist LIST   distances D, comma-separated numbers and ranges a-b or
//                 a-b:step (default 1,2,4,8,16,32,64; --copy: 0,4,8,16,32,
//                 where 0 is no prefetch)
//   --copy        pack like BLIS packs A instead of the pointer chase
//   --cols N      --copy: columns per panel, a k_c (240); in groups of four,
//                 any left over one at a time
//   --flavor F    --copy: vector, scalar or both (both)
//   --no-wrap     --copy: prefetch D columns on in the same rows, past the
//                 panel's (and the matrix's) last column, as a kernel that
//                 knows only its own panel does (default: packing order)
//   --pf-pair     --copy: each prefetch twice, at +0 and +32 bytes, as BLIS's
//                 x60 A copy at LMUL 2 issued them
//   --copy-b      pack like BLIS packs B instead: n_r-wide panels of a
//                 column-major matrix, a block of VL rows at a time, each
//                 column loaded along k (vle64), then stored into the panel a
//                 row apart (vsse64); --cols is k_c, the prefetch D lines on
//                 along each column, once per line
//   --nr N        --copy-b: panel width, 4, 8, 14 or 16 (14)
//   --pf-every-block  --copy-b: prefetch in every block -- twice per line
//                 when a block is shorter than one, as BLIS's x60 B packing did
//   --reps N      fixed repetitions instead of --min-time, so that perf stat
//                 around one configuration counts known work; prints how many
//                 lines (chase) or columns (copy) each configuration touches
//   --huge        transparent huge pages for the buffer (and reports how much
//                 of it got them), keeping TLB misses out of larger sizes
//   --cpu N, --ghz G (print cycles), --min-time S (0.1)

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include <sched.h>
#include <sys/mman.h>

namespace
{
constexpr std::size_t line = 64;
constexpr std::size_t words = line / sizeof(std::size_t);

enum kind { none, pf_l1, pf_l2, pf_l3, pf_nta, pf_w, load };
const char* const kind_name[] = {"none", "l1", "l2", "l3", "nta", "w", "load"};

// ---- the prefetches --------------------------------------------------------
#if defined(__riscv)
#define PFS_none(r)
#define PFS_l1(r)   "ori x0, " r ", 1\n\t"
#define PFS_l2(r)   "add x0, x0, x2\n\tori x0, " r ", 1\n\t"
#define PFS_l3(r)   "add x0, x0, x3\n\tori x0, " r ", 1\n\t"
#define PFS_nta(r)  "add x0, x0, x5\n\tori x0, " r ", 1\n\t"
#define PFS_w(r)    "ori x0, " r ", 3\n\t"
#define PFS_load(r) "ld x0, 0(" r ")\n\t"
// a prefetch and its twin 32 bytes on, as the BLIS x60 A copy issues them
// (prefetch.r 32(rs1) = ori x0, rs1, 33)
#define PFP_none(r)
#define PFP_l1(r)   PFS_l1(r) "ori x0, " r ", 33\n\t"
#define PFP_l2(r)   PFS_l2(r) "add x0, x0, x2\n\tori x0, " r ", 33\n\t"
#define PFP_l3(r)   PFS_l3(r) "add x0, x0, x3\n\tori x0, " r ", 33\n\t"
#define PFP_nta(r)  PFS_nta(r) "add x0, x0, x5\n\tori x0, " r ", 33\n\t"
#define PFP_w(r)    PFS_w(r) "ori x0, " r ", 35\n\t"
#define PFP_load(r) PFS_load(r) "ld x0, 32(" r ")\n\t"
#endif


template <kind K>
inline void touch(const void* p)
{
#if defined(__riscv)
    if constexpr (K == pf_l1) __asm__ volatile(PFS_l1("%0") ::"r"(p));
    if constexpr (K == pf_l2) __asm__ volatile(PFS_l2("%0") ::"r"(p));
    if constexpr (K == pf_l3) __asm__ volatile(PFS_l3("%0") ::"r"(p));
    if constexpr (K == pf_nta) __asm__ volatile(PFS_nta("%0") ::"r"(p));
    if constexpr (K == pf_w) __asm__ volatile(PFS_w("%0") ::"r"(p));
    if constexpr (K == load) __asm__ volatile(PFS_load("%0") ::"r"(p));
#elif defined(__x86_64__)
    if constexpr (K == pf_l1) __asm__ volatile("prefetcht0 (%0)" ::"r"(p));
    if constexpr (K == pf_l2) __asm__ volatile("prefetcht1 (%0)" ::"r"(p));
    if constexpr (K == pf_l3) __asm__ volatile("prefetcht2 (%0)" ::"r"(p));
    if constexpr (K == pf_nta) __asm__ volatile("prefetchnta (%0)" ::"r"(p));
    if constexpr (K == pf_w) __asm__ volatile("prefetchw (%0)" ::"r"(p));
    if constexpr (K == load) { long t; __asm__ volatile("mov (%1), %0" : "=r"(t) : "r"(p)); }
#elif defined(__aarch64__)
    if constexpr (K == pf_l1) __asm__ volatile("prfm pldl1keep, [%0]" ::"r"(p));
    if constexpr (K == pf_l2) __asm__ volatile("prfm pldl2keep, [%0]" ::"r"(p));
    if constexpr (K == pf_l3) __asm__ volatile("prfm pldl3keep, [%0]" ::"r"(p));
    if constexpr (K == pf_nta) __asm__ volatile("prfm pldl1strm, [%0]" ::"r"(p));
    if constexpr (K == pf_w) __asm__ volatile("prfm pstl1keep, [%0]" ::"r"(p));
    if constexpr (K == load) { long t; __asm__ volatile("ldr %0, [%1]" : "=r"(t) : "r"(p)); }
#else
    (void)p;
#endif
}

// touch<K>, twice when `pair`: at p and 32 bytes on
template <kind K, bool pair>
inline void touch_pf(const char* p)
{
    touch<K>(p);
    if constexpr (pair) touch<K>(p + 32);
}

// one function per kind, chosen at run time
#define DISPATCH(K, F, ...)                                   \
    switch (K)                                                \
    {                                                         \
    case none:   F<none>(__VA_ARGS__); break;                 \
    case pf_l1:  F<pf_l1>(__VA_ARGS__); break;                \
    case pf_l2:  F<pf_l2>(__VA_ARGS__); break;                \
    case pf_l3:  F<pf_l3>(__VA_ARGS__); break;                \
    case pf_nta: F<pf_nta>(__VA_ARGS__); break;               \
    case pf_w:   F<pf_w>(__VA_ARGS__); break;                 \
    case load:   F<load>(__VA_ARGS__); break;                 \
    }

volatile std::size_t sink;

// best of three timed runs, `items` lines or columns each (or one run of a
// fixed `fixed_reps`), after a warm-up: ns per item
template <typename Run>
double ns_per_item(Run run, double items, double min_time, long fixed_reps)
{
    auto timed = [&](long reps) {
        const auto t0 = std::chrono::steady_clock::now();
        run(reps);
        const auto t1 = std::chrono::steady_clock::now();
        return std::chrono::duration<double, std::nano>(t1 - t0).count();
    };
    timed(1);
    if (fixed_reps > 0) return timed(fixed_reps) / (static_cast<double>(fixed_reps) * items);
    const long reps = std::max(1L, static_cast<long>(min_time * 1e9 / std::max(1.0, timed(1))));
    double best = 1e300;
    for (int t = 0; t < 3; t++) best = std::min(best, timed(reps));
    return best / (static_cast<double>(reps) * items);
}

// ---- buffers ---------------------------------------------------------------
void* alloc_buffer(std::size_t bytes, bool huge)
{
    const std::size_t align = huge ? (2u << 20) : line;
    const std::size_t size = (bytes + align - 1) / align * align;
    void* mem = nullptr;
    if (posix_memalign(&mem, align, size) != 0)
    {
        std::perror("posix_memalign");
        std::exit(1);
    }
    if (huge) madvise(mem, size, MADV_HUGEPAGE);
    return mem;
}

// kB of this process's anonymous memory in transparent huge pages
long anon_huge_kb()
{
    std::ifstream f("/proc/self/smaps_rollup");
    std::string key;
    long kb;
    while (f >> key)
    {
        if (key == "AnonHugePages:" && (f >> kb)) return kb;
        f.ignore(1 << 20, '\n');
    }
    return -1;
}

std::string huge_note(bool huge, std::size_t bytes)
{
    if (!huge) return "";
    const long kb = anon_huge_kb();
    if (kb < 0) return " (huge pages asked for; /proc/self/smaps_rollup unreadable)";
    char buf[96];
    std::snprintf(buf, sizeof buf, " (huge pages: %ld KB in this process, buffer %zu KB)", kb,
                  bytes / 1024);
    return buf;
}

// ---- pointer chase -----------------------------------------------------------
// line order[i] holds order[i+1]: the chase visits them in that order, and the
// prefetch in step i aims at the line the chase reaches D steps later
template <kind K>
void walk(const std::size_t* buf, const std::size_t* order, std::size_t n, std::size_t d,
          long reps)
{
    std::size_t p = order[0];
    for (long r = 0; r < reps; r++)
        for (std::size_t i = 0; i < n; i++)
        {
            if constexpr (K != none) touch<K>(buf + order[i + d] * words);
            p = buf[p * words];
        }
    sink = p;
}

// ---- --copy: A-panel packing ------------------------------------------------
#if defined(__riscv_vector)
#define VCLOBBER , "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15"
#else
#define VCLOBBER
#endif

#if defined(__riscv)
#define VEC_COPY4(PFS)                                                                    \
    __asm__ volatile(".option push\n\t.option arch, +v\n\t"                               \
                     "vsetivli zero, 8, e64, m2, ta, ma\n\t"                              \
                     "vle64.v v8, (%[s0])\n\t" PFS("%[p0]")                               \
                     "vle64.v v10, (%[s1])\n\t" PFS("%[p1]")                              \
                     "vle64.v v12, (%[s2])\n\t" PFS("%[p2]")                              \
                     "vle64.v v14, (%[s3])\n\t" PFS("%[p3]")                              \
                     "vse64.v v8, (%[d0])\n\t" "vse64.v v10, (%[d1])\n\t"                 \
                     "vse64.v v12, (%[d2])\n\t" "vse64.v v14, (%[d3])\n\t"                \
                     ".option pop"                                                        \
                     :: [s0] "r"(s), [s1] "r"(s1), [s2] "r"(s2), [s3] "r"(s3),            \
                        [p0] "r"(p0), [p1] "r"(p1), [p2] "r"(p2), [p3] "r"(p3),           \
                        [d0] "r"(d), [d1] "r"(d + 8), [d2] "r"(d + 16), [d3] "r"(d + 24)  \
                     : "memory" VCLOBBER)
#endif

#if defined(__riscv)
#define VEC_COPY1(PFS)                                                                    \
    __asm__ volatile(".option push\n\t.option arch, +v\n\t"                               \
                     "vsetivli zero, 8, e64, m2, ta, ma\n\t"                              \
                     "vle64.v v8, (%[s0])\n\t" PFS("%[p0]")                               \
                     "vse64.v v8, (%[d0])\n\t"                                            \
                     ".option pop"                                                        \
                     :: [s0] "r"(s), [p0] "r"(p), [d0] "r"(d)                             \
                     : "memory" VCLOBBER)
#endif

// one column of an 8-row panel (a 64-byte line) to the packed panel, the load
// followed by kind K's prefetch of p: for a panel's columns left over after
// the groups of four
#define VEC_KINDS(COPY, PFX)                         \
    if constexpr (K == none) COPY(PFX##none);        \
    if constexpr (K == pf_l1) COPY(PFX##l1);         \
    if constexpr (K == pf_l2) COPY(PFX##l2);         \
    if constexpr (K == pf_l3) COPY(PFX##l3);         \
    if constexpr (K == pf_nta) COPY(PFX##nta);       \
    if constexpr (K == pf_w) COPY(PFX##w);           \
    if constexpr (K == load) COPY(PFX##load);

template <kind K, bool vec, bool pair>
inline void copy1(const char* s, const char* p, double* d)
{
#if defined(__riscv)
    if constexpr (vec)
    {
        if constexpr (pair) { VEC_KINDS(VEC_COPY1, PFP_) }
        else { VEC_KINDS(VEC_COPY1, PFS_) }
    }
    else
    {
        touch_pf<K, pair>(p);
        __asm__ volatile("ld t0, 0(%[s])\n\t"  "ld t1, 8(%[s])\n\t"  "ld t2, 16(%[s])\n\t"
                         "ld t3, 24(%[s])\n\t" "ld t4, 32(%[s])\n\t" "ld t5, 40(%[s])\n\t"
                         "ld t6, 48(%[s])\n\t" "ld a7, 56(%[s])\n\t"
                         "sd t0, 0(%[d])\n\t"  "sd t1, 8(%[d])\n\t"  "sd t2, 16(%[d])\n\t"
                         "sd t3, 24(%[d])\n\t" "sd t4, 32(%[d])\n\t" "sd t5, 40(%[d])\n\t"
                         "sd t6, 48(%[d])\n\t" "sd a7, 56(%[d])\n\t"
                         :: [s] "r"(s), [d] "r"(d)
                         : "t0", "t1", "t2", "t3", "t4", "t5", "t6", "a7", "memory");
    }
#else
    touch_pf<K, pair>(p);
    if constexpr (vec) std::memcpy(d, s, 64);
    else
        for (int w = 0; w < 8; w++) d[w] = reinterpret_cast<const volatile double*>(s)[w];
#endif
}

// columns c..c+3 of one 8-row panel (a 64-byte line each, lda bytes apart) to
// the packed panel, loads first, each followed by kind K's prefetch of pf[g]
template <kind K, bool vec, bool pair>
inline void copy4(const char* s, std::size_t lda, const char* const* pf, double* d)
{
    const char *s1 = s + lda, *s2 = s1 + lda, *s3 = s2 + lda;
    const char *p0 = pf[0], *p1 = pf[1], *p2 = pf[2], *p3 = pf[3];
#if defined(__riscv)
    if constexpr (vec)
    {
        if constexpr (pair) { VEC_KINDS(VEC_COPY4, PFP_) }
        else { VEC_KINDS(VEC_COPY4, PFS_) }
    }
    else
    {
        // a column at a time: its prefetch, eight loads, then eight stores
        const char* const cs[4] = {s, s1, s2, s3};
        const char* const ps[4] = {p0, p1, p2, p3};
        for (int c = 0; c < 4; c++)
        {
            touch_pf<K, pair>(ps[c]);
            __asm__ volatile("ld t0, 0(%[s])\n\t"  "ld t1, 8(%[s])\n\t"  "ld t2, 16(%[s])\n\t"
                             "ld t3, 24(%[s])\n\t" "ld t4, 32(%[s])\n\t" "ld t5, 40(%[s])\n\t"
                             "ld t6, 48(%[s])\n\t" "ld a7, 56(%[s])\n\t"
                             "sd t0, 0(%[d])\n\t"  "sd t1, 8(%[d])\n\t"  "sd t2, 16(%[d])\n\t"
                             "sd t3, 24(%[d])\n\t" "sd t4, 32(%[d])\n\t" "sd t5, 40(%[d])\n\t"
                             "sd t6, 48(%[d])\n\t" "sd a7, 56(%[d])\n\t"
                             :: [s] "r"(cs[c]), [d] "r"(d + 8 * c)
                             : "t0", "t1", "t2", "t3", "t4", "t5", "t6", "a7", "memory");
        }
    }
#else
    // elsewhere both flavors are plain copies: a reference, not the RVV question
    const char* const cs[4] = {s, s1, s2, s3};
    const char* const ps[4] = {p0, p1, p2, p3};
    for (int c = 0; c < 4; c++)
    {
        touch_pf<K, pair>(ps[c]);
        if constexpr (vec) std::memcpy(d + 8 * c, cs[c], 64);
        else
            for (int w = 0; w < 8; w++)
                d[8 * c + w] = reinterpret_cast<const volatile double*>(cs[c])[w];
    }
#endif
}

// every panel of a rows x cols matrix (lda = rows doubles), into one panel
// buffer. The prefetch aims `dist` columns further in packing order: past a
// panel's last column it continues with the next panel's (8 rows down) --
// tracked by a cursor, since a division per column would cost more than the
// copy.
// Without `wrap`, as a kernel that knows only its own panel: the same rows,
// D columns on -- past the panel's last column, and the matrix's.
template <kind K, bool vec, bool pair>
void pack_all(const double* a, std::size_t rows, std::size_t cols, std::size_t dist,
              double* panel, long reps, bool wrap)
{
    const std::size_t lda = rows * sizeof(double);
    const char* const base = reinterpret_cast<const char*>(a);
    const std::size_t panels_ahead = wrap ? dist / cols : 0, cols_ahead = wrap ? dist % cols : dist;
    for (long r = 0; r < reps; r++)
        for (std::size_t i = 0; i < rows; i += 8)
        {
            std::size_t prow = i + 8 * panels_ahead, pcol = cols_ahead;
            const char* pf = base + prow * sizeof(double) + pcol * lda;
            auto next = [&]() -> const char* {
                const char* at = pf;
                if (!wrap) pf += lda;
                else if (++pcol == cols)
                {
                    pcol = 0;
                    prow += 8;
                    pf = base + prow * sizeof(double);
                }
                else pf += lda;
                return at;
            };
            const char* col = base + i * sizeof(double);
            std::size_t c = 0;
            for (; c + 4 <= cols; c += 4, col += 4 * lda)
            {
                const char* pfs[4] = {nullptr, nullptr, nullptr, nullptr};
                if constexpr (K != none)
                    for (auto& p : pfs) p = next();
                copy4<K, vec, pair>(col, lda, pfs, panel + 8 * c);
            }
            for (; c < cols; c++, col += lda)
                copy1<K, vec, pair>(col, K != none ? next() : nullptr, panel + 8 * c);
        }
}

template <kind K>
void pack_vec(const double* a, std::size_t rows, std::size_t cols, std::size_t dist,
              double* panel, long reps, bool wrap, bool pair)
{
    if (pair) pack_all<K, true, true>(a, rows, cols, dist, panel, reps, wrap);
    else pack_all<K, true, false>(a, rows, cols, dist, panel, reps, wrap);
}

template <kind K>
void pack_scalar(const double* a, std::size_t rows, std::size_t cols, std::size_t dist,
                 double* panel, long reps, bool wrap, bool pair)
{
    if (pair) pack_all<K, false, true>(a, rows, cols, dist, panel, reps, wrap);
    else pack_all<K, false, false>(a, rows, cols, dist, panel, reps, wrap);
}

// the vector copy needs 8 doubles in an LMUL-2 group: VLEN >= 256
bool vector_copy_fits()
{
#if defined(__riscv)
    std::size_t vl = 0;
    __asm__ volatile(".option push\n\t.option arch, +v\n\t"
                     "vsetivli %0, 8, e64, m2, ta, ma\n\t"
                     ".option pop" : "=r"(vl));
    return vl == 8;
#else
    return true;
#endif
}

// ---- --copy-b: B-panel packing ---------------------------------------------
// One block: VL rows (k) of n_r source columns, ldb bytes apart, as BLIS's
// x60 B packing does it -- each column's VL elements with one vle64 (LMUL 1),
// each load followed by the prefetch, then every column with a strided vsse64
// into the panel (row stride n_r doubles). v8.. hold the columns.
#if defined(__riscv_vector)
#define VCLOBBER_B , "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", \
                     "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23"
#else
#define VCLOBBER_B
#endif
#if defined(__riscv)
#define BLD(n, PFM) "vle64.v v" #n ", (%[t])\n\t" PFM "add %[t], %[t], %[ldb]\n\t"
#define BST(n)      "vsse64.v v" #n ", (%[u]), %[ds]\n\t" "addi %[u], %[u], 8\n\t"
// the prefetch of the column being loaded: pfoff bytes on along it
#define BPF_none
#define BPF_l1   "add %[t3], %[t], %[pfoff]\n\t" PFS_l1("%[t3]")
#define BPF_l2   "add %[t3], %[t], %[pfoff]\n\t" PFS_l2("%[t3]")
#define BPF_l3   "add %[t3], %[t], %[pfoff]\n\t" PFS_l3("%[t3]")
#define BPF_nta  "add %[t3], %[t], %[pfoff]\n\t" PFS_nta("%[t3]")
#define BPF_w    "add %[t3], %[t], %[pfoff]\n\t" PFS_w("%[t3]")
#define BPF_load "add %[t3], %[t], %[pfoff]\n\t" PFS_load("%[t3]")
#define BLOADS4(PFM) BLD(8, PFM) BLD(9, PFM) BLD(10, PFM) BLD(11, PFM)
#define BSTORES4 BST(8) BST(9) BST(10) BST(11)
#define BLOADS8(PFM) BLD(8, PFM) BLD(9, PFM) BLD(10, PFM) BLD(11, PFM) BLD(12, PFM) BLD(13, PFM) BLD(14, PFM) BLD(15, PFM)
#define BSTORES8 BST(8) BST(9) BST(10) BST(11) BST(12) BST(13) BST(14) BST(15)
#define BLOADS14(PFM) BLD(8, PFM) BLD(9, PFM) BLD(10, PFM) BLD(11, PFM) BLD(12, PFM) BLD(13, PFM) BLD(14, PFM) BLD(15, PFM) BLD(16, PFM) BLD(17, PFM) BLD(18, PFM) BLD(19, PFM) BLD(20, PFM) BLD(21, PFM)
#define BSTORES14 BST(8) BST(9) BST(10) BST(11) BST(12) BST(13) BST(14) BST(15) BST(16) BST(17) BST(18) BST(19) BST(20) BST(21)
#define BLOADS16(PFM) BLD(8, PFM) BLD(9, PFM) BLD(10, PFM) BLD(11, PFM) BLD(12, PFM) BLD(13, PFM) BLD(14, PFM) BLD(15, PFM) BLD(16, PFM) BLD(17, PFM) BLD(18, PFM) BLD(19, PFM) BLD(20, PFM) BLD(21, PFM) BLD(22, PFM) BLD(23, PFM)
#define BSTORES16 BST(8) BST(9) BST(10) BST(11) BST(12) BST(13) BST(14) BST(15) BST(16) BST(17) BST(18) BST(19) BST(20) BST(21) BST(22) BST(23)
#define B_BLOCK(LOADS, STORES)                                                        \
    __asm__ volatile(".option push\n\t.option arch, +v\n\t"                              \
                     "vsetvli zero, %[vl], e64, m1, ta, ma\n\t"                            \
                     "mv %[t], %[src]\n\t" LOADS                                            \
                     "mv %[u], %[dst]\n\t" STORES                                           \
                     ".option pop"                                                         \
                     : [t] "=&r"(t), [u] "=&r"(u), [t3] "=&r"(t3)                          \
                     : [src] "r"(src), [dst] "r"(dst), [ldb] "r"(ldb), [ds] "r"(ds),       \
                       [vl] "r"(vl), [pfoff] "r"(pfoff)                                    \
                     : "memory" VCLOBBER_B)
#define B_WIDTHS(PFM)                                                     \
    if constexpr (NR == 4) B_BLOCK(BLOADS4(PFM), BSTORES4);               \
    if constexpr (NR == 8) B_BLOCK(BLOADS8(PFM), BSTORES8);               \
    if constexpr (NR == 14) B_BLOCK(BLOADS14(PFM), BSTORES14);            \
    if constexpr (NR == 16) B_BLOCK(BLOADS16(PFM), BSTORES16);
#endif

// one block of an n_r-wide B panel: rows src..src+vl of each column to the
// packed panel at dst; the prefetch of each column `pfoff` bytes on
template <kind K, int NR, bool vec>
inline void b_block(const char* src, std::size_t ldb, std::size_t vl, long pfoff, double* dst)
{
    const std::size_t ds = NR * sizeof(double);
#if defined(__riscv)
    if constexpr (vec)
    {
        const char *t, *t3;
        double* u;
        if constexpr (K == none) { B_WIDTHS(BPF_none) }
        if constexpr (K == pf_l1) { B_WIDTHS(BPF_l1) }
        if constexpr (K == pf_l2) { B_WIDTHS(BPF_l2) }
        if constexpr (K == pf_l3) { B_WIDTHS(BPF_l3) }
        if constexpr (K == pf_nta) { B_WIDTHS(BPF_nta) }
        if constexpr (K == pf_w) { B_WIDTHS(BPF_w) }
        if constexpr (K == load) { B_WIDTHS(BPF_load) }
        (void)t; (void)t3; (void)u;
        return;
    }
#endif
    // scalar (and the reference elsewhere): a column at a time, its elements
    // loaded, the prefetch, then stored a panel row apart
    (void)ds;
    for (int j = 0; j < NR; j++)
    {
        const double* col = reinterpret_cast<const double*>(src + j * ldb);
        double e[16];
        for (std::size_t r = 0; r < vl; r++) e[r] = reinterpret_cast<const volatile double*>(col)[r];
        touch<K>(reinterpret_cast<const char*>(col) + pfoff);
        for (std::size_t r = 0; r < vl; r++) dst[r * NR + j] = e[r];
    }
}

// every n_r-wide panel of a kc x n matrix (ldb = kc doubles) into one panel
// buffer, block by block. The prefetch aims `dist` lines on along each
// column, once per line: with blocks shorter than a line it alternates every
// block with -64, the line just read, as the BLIS x60 kernel does -- unless
// `every_block`, the scheme that prefetched each line twice.
template <kind K, int NR, bool vec>
void pack_b_all(const double* b, std::size_t kc, std::size_t panels, std::size_t vl,
                std::size_t dist, bool every_block, double* panel, long reps)
{
    const std::size_t ldb = kc * sizeof(double), block_bytes = vl * sizeof(double);
    const long ahead = static_cast<long>(dist * line);
    const bool alternate = !every_block && block_bytes < line;
    for (long r = 0; r < reps; r++)
        for (std::size_t q = 0; q < panels; q++)
        {
            const char* col0 = reinterpret_cast<const char*>(b + q * NR * kc);
            long pfoff = ahead;
            for (std::size_t p = 0; p < kc; p += vl)
            {
                b_block<K, NR, vec>(col0 + p * sizeof(double), ldb, vl, pfoff, panel + p * NR);
                if (alternate) pfoff = pfoff == ahead ? -static_cast<long>(line) : ahead;
            }
        }
}

template <kind K>
void pack_b(bool vec, std::size_t nr, const double* b, std::size_t kc, std::size_t panels,
            std::size_t vl, std::size_t dist, bool every_block, double* panel, long reps)
{
#define PACK_B_NR(N)                                                                          \
    if (nr == N)                                                                              \
    {                                                                                         \
        if (vec) pack_b_all<K, N, true>(b, kc, panels, vl, dist, every_block, panel, reps);   \
        else pack_b_all<K, N, false>(b, kc, panels, vl, dist, every_block, panel, reps);      \
        return;                                                                               \
    }
    PACK_B_NR(4) PACK_B_NR(8) PACK_B_NR(14) PACK_B_NR(16)
#undef PACK_B_NR
}

// the vector block height: VLMAX at e64, m1 (4 doubles at VLEN 256)
std::size_t vector_rows()
{
#if defined(__riscv)
    std::size_t vl = 0;
    __asm__ volatile(".option push\n\t.option arch, +v\n\t"
                     "vsetvli %0, zero, e64, m1, ta, ma\n\t"
                     ".option pop" : "=r"(vl));
    return vl;
#else
    return 4;
#endif
}

// ---- options -----------------------------------------------------------------
struct options
{
    int cpu = -1;
    double ghz = 0, min_time = 0.1;
    long reps = 0;
    bool huge = false, copy = false, wrap = true, pair = false;
    bool copy_b = false, every_block = false;
    std::size_t nr = 14;
    std::size_t cols = 240;
    bool vector = true, scalar = true;
    std::vector<kind> kinds;
    std::vector<std::size_t> dists, sizes_kb;
};

[[noreturn]] void usage(const char* argv0, const std::string& why = "")
{
    if (!why.empty()) std::fprintf(stderr, "error: %s\n", why.c_str());
    std::fprintf(stderr,
                 "usage: %s [--pf l1,l2,l3,nta,w,load] [--dist LIST] [--copy] [--cols N]\n"
                 "       [--flavor vector|scalar|both] [--no-wrap] [--pf-pair] [--copy-b]\n"
                 "       [--nr N] [--pf-every-block] [--reps N]\n"
                 "       [--huge] [--cpu N] [--ghz G] [--min-time S] [SIZE_KB...]\n", argv0);
    std::exit(1);
}

std::vector<std::string> split(const std::string& s, char sep)
{
    std::vector<std::string> out;
    std::size_t start = 0;
    for (std::size_t i = 0; i <= s.size(); i++)
        if (i == s.size() || s[i] == sep)
        {
            if (i > start) out.push_back(s.substr(start, i - start));
            start = i + 1;
        }
    return out;
}

std::size_t number(const std::string& s, const char* argv0)
{
    std::size_t used = 0;
    unsigned long long v = 0;
    try { v = std::stoull(s, &used); } catch (const std::exception&) { used = 0; }
    if (used != s.size() || s.empty()) usage(argv0, "not a number: " + s);
    return static_cast<std::size_t>(v);
}

// "0,4-16:4,32" -> 0 4 8 12 16 32
std::vector<std::size_t> parse_dists(const std::string& arg, const char* argv0)
{
    std::vector<std::size_t> out;
    for (const std::string& item : split(arg, ','))
    {
        const std::size_t dash = item.find('-');
        if (dash == std::string::npos)
        {
            out.push_back(number(item, argv0));
            continue;
        }
        const std::size_t colon = item.find(':', dash);
        const std::size_t a = number(item.substr(0, dash), argv0);
        const std::size_t b = number(item.substr(dash + 1, colon == std::string::npos
                                                               ? std::string::npos
                                                               : colon - dash - 1), argv0);
        const std::size_t step = colon == std::string::npos ? 1 : number(item.substr(colon + 1), argv0);
        if (step == 0 || b < a) usage(argv0, "bad range: " + item);
        for (std::size_t d = a; d <= b; d += step) out.push_back(d);
    }
    if (out.empty()) usage(argv0, "no distances in --dist " + arg);
    return out;
}

std::vector<kind> parse_kinds(const std::string& arg, const char* argv0)
{
    std::vector<kind> out;
    for (const std::string& name : split(arg, ','))
    {
        bool found = false;
        for (int k = pf_l1; k <= load; k++)
            if (name == kind_name[k])
            {
                out.push_back(static_cast<kind>(k));
                found = true;
            }
        if (!found) usage(argv0, "unknown prefetch kind: " + name + " (l1 l2 l3 nta w load)");
    }
    if (out.empty()) usage(argv0, "no kinds in --pf " + arg);
    return out;
}

options parse(int argc, char** argv)
{
    options o;
    std::string pf, dist;
    for (int i = 1; i < argc; i++)
    {
        const std::string a = argv[i];
        auto value = [&]() -> std::string {
            if (i + 1 >= argc) usage(argv[0], a + " needs a value");
            return argv[++i];
        };
        if (a == "--cpu") o.cpu = std::atoi(value().c_str());
        else if (a == "--ghz") o.ghz = std::atof(value().c_str());
        else if (a == "--min-time") o.min_time = std::atof(value().c_str());
        else if (a == "--reps") o.reps = static_cast<long>(number(value(), argv[0]));
        else if (a == "--huge") o.huge = true;
        else if (a == "--copy") o.copy = true;
        else if (a == "--no-wrap") o.wrap = false;
        else if (a == "--copy-b") o.copy_b = true;
        else if (a == "--nr") o.nr = number(value(), argv[0]);
        else if (a == "--pf-every-block") o.every_block = true;
        else if (a == "--pf-pair") o.pair = true;
        else if (a == "--cols") o.cols = number(value(), argv[0]);
        else if (a == "--pf") pf = value();
        else if (a == "--dist") dist = value();
        else if (a == "--flavor")
        {
            const std::string f = value();
            if (f != "vector" && f != "scalar" && f != "both")
                usage(argv[0], "--flavor is vector, scalar or both");
            o.vector = f != "scalar";
            o.scalar = f != "vector";
        }
        else if (!a.empty() && a[0] != '-') o.sizes_kb.push_back(number(a, argv[0]));
        else usage(argv[0], a == "--help" || a == "-h" ? "" : "unknown option " + a);
    }
    if (o.sizes_kb.empty()) o.sizes_kb = {16, 256, 4096};
    if (o.copy && o.copy_b) usage(argv[0], "--copy or --copy-b, not both");
    if (o.copy_b && o.nr != 4 && o.nr != 8 && o.nr != 14 && o.nr != 16)
        usage(argv[0], "--nr is 4, 8, 14 or 16");
    const bool packing = o.copy || o.copy_b;
    o.kinds = parse_kinds(pf.empty() ? (packing ? "l1" : "l1,l2,load") : pf, argv[0]);
    o.dists = parse_dists(dist.empty() ? (packing ? "0,4,8,16,32" : "1,2,4,8,16,32,64") : dist,
                          argv[0]);
    if (o.cols == 0) usage(argv[0], "--cols must be positive");
    return o;
}

// ---- the two modes -------------------------------------------------------------
int chase_mode(const options& o, double scale, const char* unit)
{
    const std::size_t max_d = *std::max_element(o.dists.begin(), o.dists.end());
    std::printf("# a random-order chase over 64-byte lines, %s per line; prefetches\n"
                "# issued D lines ahead of the chase\n", unit);
    std::mt19937_64 rng(42);
    for (const std::size_t kb : o.sizes_kb)
    {
        const std::size_t n = std::max<std::size_t>(2, kb * 1024 / line);
        auto* buf = static_cast<std::size_t*>(alloc_buffer(n * line, o.huge));
        std::vector<std::size_t> order(n + max_d + 1);
        std::iota(order.begin(), order.begin() + n, std::size_t{0});
        std::shuffle(order.begin(), order.begin() + n, rng);
        for (std::size_t j = 0; j <= max_d; j++) order[n + j] = order[j % n];
        for (std::size_t i = 0; i < n; i++)
            for (std::size_t w = 0; w < words; w++)
                buf[order[i] * words + w] = order[(i + 1) % n];

        auto cost = [&](kind k, std::size_t d) {
            return ns_per_item([&](long reps) { DISPATCH(k, walk, buf, order.data(), n, d, reps) },
                               static_cast<double>(n), o.min_time, o.reps) * scale;
        };
        std::printf("\n# %zu KB%s: chase alone %.1f %s per line\n", kb,
                    huge_note(o.huge, n * line).c_str(), cost(none, 0), unit);
        std::printf("  %4s", "D");
        for (kind k : o.kinds) std::printf(" %9s", kind_name[k]);
        std::printf("\n");
        for (const std::size_t d : o.dists)
        {
            if (d == 0) continue;                       // the chase alone, above
            std::printf("  %4zu", d);
            for (kind k : o.kinds) std::printf(" %9.1f", cost(k, d));
            std::printf("\n");
        }
        if (o.reps > 0)
            std::printf("# --reps %ld: each configuration walks %ld lines\n", o.reps,
                        (o.reps + 1) * static_cast<long>(n));
        std::fflush(stdout);
        std::free(buf);
    }
    return 0;
}

int copy_mode(const options& o, double scale, const char* unit)
{
    const std::size_t cols = o.cols;
    const bool vec = o.vector && vector_copy_fits();
    if (o.vector && !vec) std::printf("# (no vector copy: VLEN < 256)\n");
    if (!vec && !o.scalar) return 0;
    std::printf("# --copy: 8-row panels of a column-major matrix, %zu columns, one 64-byte\n"
                "# line per column, 4 columns per iteration; %s per column, prefetch D\n"
                "# columns ahead %s (0: none)%s\n", cols, unit,
                o.wrap ? "in packing order" : "in the same rows, past the panel's end",
                o.pair ? ", each twice (+0, +32)" : "");
    for (const std::size_t kb : o.sizes_kb)
    {
        std::size_t rows = std::max<std::size_t>(8, kb * 1024 / sizeof(double) / cols / 8 * 8);
        if ((rows & (rows - 1)) == 0) rows += 8;        // no power-of-two lda
        const std::size_t bytes = rows * cols * sizeof(double);
        auto* a = static_cast<double*>(alloc_buffer(bytes, o.huge));
        std::vector<double> panel(8 * cols);
        for (std::size_t i = 0; i < rows * cols; i++) a[i] = static_cast<double>(i);
        const double items = static_cast<double>(rows / 8 * cols);

        auto cost = [&](bool v, kind k, std::size_t d) {
            if (d == 0) k = none;
            return ns_per_item([&](long reps) {
                if (v) { DISPATCH(k, pack_vec, a, rows, cols, d, panel.data(), reps, o.wrap, o.pair) }
                else   { DISPATCH(k, pack_scalar, a, rows, cols, d, panel.data(), reps, o.wrap, o.pair) }
            }, items, o.min_time, o.reps) * scale;
        };
        std::printf("\n# %zu KB: %zu x %zu, lda %zu bytes%s\n  %4s", kb, rows, cols,
                    rows * sizeof(double), huge_note(o.huge, bytes).c_str(), "D");
        for (int f = 0; f < 2; f++)
            if (f == 0 ? vec : o.scalar)
                for (kind k : o.kinds)
                    std::printf(" %10s", (std::string(f == 0 ? "vector." : "scalar.") +
                                          kind_name[k]).c_str());
        std::printf("\n");
        for (const std::size_t d : o.dists)
        {
            std::printf("  %4zu", d);
            for (int f = 0; f < 2; f++)
            {
                if (!(f == 0 ? vec : o.scalar)) continue;
                const double base = d == 0 ? cost(f == 0, none, 0) : 0;
                for (kind k : o.kinds)
                    std::printf(" %10.1f", d == 0 ? base : cost(f == 0, k, d));
            }
            std::printf("%s\n", d == 0 ? "   (no prefetch)" : "");
        }
        if (o.reps > 0)
            std::printf("# --reps %ld: each configuration copies %ld columns\n", o.reps,
                        (o.reps + 1) * static_cast<long>(items));
        // the last panel's columns must have landed
        for (std::size_t c = 0; c < cols; c++)
            for (std::size_t r = 0; r < 8; r++)
                if (panel[8 * c + r] != a[(rows - 8 + r) + c * rows])
                {
                    std::fprintf(stderr, "copy check failed at column %zu row %zu\n", c, r);
                    return 1;
                }
        std::fflush(stdout);
        std::free(a);
    }
    return 0;
}
int copy_b_mode(const options& o, double scale, const char* unit)
{
    const std::size_t kc = o.cols, nr = o.nr, vl = vector_rows();
    if (kc % vl) usage("prefetch_test", "--cols must be a multiple of the vector length here");
    std::printf("# --copy-b: %zu-wide panels of a column-major %zu-row matrix (ldb %zu bytes),\n"
                "# %zu rows a block, each column loaded then stored into the panel a row of\n"
                "# %zu doubles apart; %s per source line, prefetch D lines on along each column\n"
                "# (0: none), %s\n", nr, kc, kc * sizeof(double), vl, nr, unit,
                o.every_block || vl * sizeof(double) >= line ? "every block"
                                                             : "every other block (one per line)");
    for (const std::size_t kb : o.sizes_kb)
    {
        const std::size_t panels = std::max<std::size_t>(1, kb * 1024 / (kc * nr * sizeof(double)));
        const std::size_t bytes = panels * nr * kc * sizeof(double);
        auto* b = static_cast<double*>(alloc_buffer(bytes, o.huge));
        std::vector<double> panel(kc * nr);
        for (std::size_t i = 0; i < panels * nr * kc; i++) b[i] = static_cast<double>(i);
        const double items = static_cast<double>(panels * nr * kc) / (line / sizeof(double));
        auto cost = [&](bool v, kind k, std::size_t d) {
            if (d == 0) k = none;
            return ns_per_item([&](long reps) {
                DISPATCH(k, pack_b, v, nr, b, kc, panels, vl, d, o.every_block, panel.data(), reps)
            }, items, o.min_time, o.reps) * scale;
        };
        const bool vec = o.vector;
        std::printf("\n# %zu KB: %zu panels%s\n  %4s", kb, panels, huge_note(o.huge, bytes).c_str(),
                    "D");
        for (int f = 0; f < 2; f++)
            if (f == 0 ? vec : o.scalar)
                for (kind k : o.kinds)
                    std::printf(" %10s", (std::string(f == 0 ? "vector." : "scalar.") +
                                          kind_name[k]).c_str());
        std::printf("\n");
        for (const std::size_t d : o.dists)
        {
            std::printf("  %4zu", d);
            for (int f = 0; f < 2; f++)
            {
                if (!(f == 0 ? vec : o.scalar)) continue;
                const double base = d == 0 ? cost(f == 0, none, 0) : 0;
                for (kind k : o.kinds) std::printf(" %10.1f", d == 0 ? base : cost(f == 0, k, d));
            }
            std::printf("%s\n", d == 0 ? "   (no prefetch)" : "");
        }
        if (o.reps > 0)
            std::printf("# --reps %ld: each configuration packs %ld source lines\n", o.reps,
                        (o.reps + 1) * static_cast<long>(items));
        // the last panel must have landed
        const double* last = b + (panels - 1) * nr * kc;
        for (std::size_t r = 0; r < kc; r++)
            for (std::size_t j = 0; j < nr; j++)
                if (panel[r * nr + j] != last[r + j * kc])
                {
                    std::fprintf(stderr, "copy-b check failed at row %zu column %zu\n", r, j);
                    return 1;
                }
        std::fflush(stdout);
        std::free(b);
    }
    return 0;
}
}  // namespace

int main(int argc, char** argv)
{
    const options o = parse(argc, argv);
    if (o.cpu >= 0)
    {
        cpu_set_t set;
        CPU_ZERO(&set);
        CPU_SET(o.cpu, &set);
        if (sched_setaffinity(0, sizeof(set), &set) != 0) std::perror("sched_setaffinity");
    }
    const char* unit = o.ghz > 0 ? "cycles" : "ns";
    const double scale = o.ghz > 0 ? o.ghz : 1.0;
    std::printf("# prefetch_test:");
    for (int i = 1; i < argc; i++) std::printf(" %s", argv[i]);
    std::printf("\n");
    if (o.copy_b) return copy_b_mode(o, scale, unit);
    return o.copy ? copy_mode(o, scale, unit) : chase_mode(o, scale, unit);
}
