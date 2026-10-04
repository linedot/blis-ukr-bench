// prefetch_test -- does a software prefetch reach this core's L1, and how far
// ahead must it go?
//
// Walks the 64-byte lines of a buffer in a random order, which no hardware
// prefetcher follows; each line holds the index of the next, so the walk is a
// pointer chase: no two misses overlap and each costs its full latency, on
// in-order and out-of-order cores alike.  Then the same walk with a software
// prefetch of the line D steps ahead: once D steps take longer than the
// latency, every load should find its line in L1 -- if the prefetch does
// anything.
//
//   chase    the walk alone: the latency of the level the buffer lives in
//   pf.L1    the prefetch the kernels use (RISC-V: prefetch.r; x86:
//            prefetcht0; AArch64: prfm pldl1keep)
//   pf.L2    aimed at L2 (RISC-V: ntl.p1 + prefetch.r; x86: prefetcht1;
//            AArch64: prfm pldl2keep)
//   load     a plain load of the line, its value discarded (RISC-V: ld x0):
//            what a prefetch that certainly fetches achieves
//
// pf.L1 no better than chase at any D: the prefetch is ignored (Zicbop allows
// that) or lands where it does not help. load better than pf.L1: the core
// drops prefetches but not loads. A buffer in L1 shows what each costs when
// everything hits.
//
// --copy: packing as BLIS packs A instead -- 8-row panels of a column-major
// matrix, one 64-byte line per column, four columns per iteration, loaded
// then stored -- with vector loads (vle64 at LMUL 2, like the X60's dpackm)
// and with scalar ld/sd, each with prefetch.r D columns ahead (D = 0: none).
// Whether a prefetch helps a vector load, or survives four outstanding
// misses, shows here: the pointer chase has one at a time and scalar loads.
//
// usage: prefetch_test [--cpu N] [--ghz G] [--min-time S] [--huge] [--copy]
//                      [SIZE_KB...]
//        (default sizes 16 256 4096; --huge asks for transparent huge pages,
//        which keeps TLB misses out of the larger sizes)

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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
constexpr std::size_t distances[] = {1, 2, 4, 8, 16, 32, 64};
constexpr std::size_t max_distance = 64;

enum kind { chase_only, pf_l1, pf_l2, load };
const char* const kind_name[] = {"chase", "pf.L1", "pf.L2", "load"};

template <kind K>
inline void touch(const void* p)
{
#if defined(__riscv)
    // raw encodings, no -march needed: prefetch.r 0(rs1) = ori x0, rs1, 1;
    // ntl.p1 = add x0, x0, x2
    if constexpr (K == pf_l1) __asm__ volatile("ori x0, %0, 1" ::"r"(p));
    if constexpr (K == pf_l2) __asm__ volatile("add x0, x0, x2\n\tori x0, %0, 1" ::"r"(p));
    if constexpr (K == load) __asm__ volatile("ld x0, 0(%0)" ::"r"(p));
#elif defined(__x86_64__)
    if constexpr (K == pf_l1) __asm__ volatile("prefetcht0 (%0)" ::"r"(p));
    if constexpr (K == pf_l2) __asm__ volatile("prefetcht1 (%0)" ::"r"(p));
    if constexpr (K == load) { long t; __asm__ volatile("mov (%1), %0" : "=r"(t) : "r"(p)); }
#elif defined(__aarch64__)
    if constexpr (K == pf_l1) __asm__ volatile("prfm pldl1keep, [%0]" ::"r"(p));
    if constexpr (K == pf_l2) __asm__ volatile("prfm pldl2keep, [%0]" ::"r"(p));
    if constexpr (K == load) { long t; __asm__ volatile("ldr %0, [%1]" : "=r"(t) : "r"(p)); }
#else
    (void)p;
#endif
}

// line order[i] holds order[i+1]: the chase visits them in that order, and the
// prefetch in step i aims at the line the chase reaches D steps later
template <kind K>
std::size_t walk(const std::size_t* buf, const std::size_t* order, std::size_t n,
                 std::size_t d, long reps)
{
    std::size_t p = order[0];
    for (long r = 0; r < reps; r++)
        for (std::size_t i = 0; i < n; i++)
        {
            if constexpr (K != chase_only) touch<K>(buf + order[i + d] * words);
            p = buf[p * words];
        }
    return p;
}

volatile std::size_t sink;

// ---- --copy: A-panel packing ---------------------------------------------
#if defined(__riscv_vector)
#define VCLOBBER , "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15"
#else
#define VCLOBBER
#endif

// columns c..c+3 of one 8-row panel (one 64-byte line each, lda bytes apart)
// to the packed panel, loads first; pf: prefetch.r each column D ahead
template <bool vec, bool pf>
inline void copy4(const char* s, std::size_t lda, std::size_t ahead, double* d)
{
#if defined(__riscv)
    const char *s1 = s + lda, *s2 = s1 + lda, *s3 = s2 + lda;
    const char *p0 = s + ahead, *p1 = s1 + ahead, *p2 = s2 + ahead, *p3 = s3 + ahead;
    if constexpr (vec)
    {
        if constexpr (pf)
            __asm__ volatile(".option push\n\t.option arch, +v\n\t"
                             "vsetivli zero, 8, e64, m2, ta, ma\n\t"
                             "vle64.v v8, (%[s0])\n\t"  "ori x0, %[p0], 1\n\t"
                             "vle64.v v10, (%[s1])\n\t" "ori x0, %[p1], 1\n\t"
                             "vle64.v v12, (%[s2])\n\t" "ori x0, %[p2], 1\n\t"
                             "vle64.v v14, (%[s3])\n\t" "ori x0, %[p3], 1\n\t"
                             "vse64.v v8, (%[d0])\n\t"  "vse64.v v10, (%[d1])\n\t"
                             "vse64.v v12, (%[d2])\n\t" "vse64.v v14, (%[d3])\n\t"
                             ".option pop"
                             :: [s0] "r"(s), [s1] "r"(s1), [s2] "r"(s2), [s3] "r"(s3),
                                [p0] "r"(p0), [p1] "r"(p1), [p2] "r"(p2), [p3] "r"(p3),
                                [d0] "r"(d), [d1] "r"(d + 8), [d2] "r"(d + 16), [d3] "r"(d + 24)
                             : "memory" VCLOBBER);
        else
            __asm__ volatile(".option push\n\t.option arch, +v\n\t"
                             "vsetivli zero, 8, e64, m2, ta, ma\n\t"
                             "vle64.v v8, (%[s0])\n\t"  "vle64.v v10, (%[s1])\n\t"
                             "vle64.v v12, (%[s2])\n\t" "vle64.v v14, (%[s3])\n\t"
                             "vse64.v v8, (%[d0])\n\t"  "vse64.v v10, (%[d1])\n\t"
                             "vse64.v v12, (%[d2])\n\t" "vse64.v v14, (%[d3])\n\t"
                             ".option pop"
                             :: [s0] "r"(s), [s1] "r"(s1), [s2] "r"(s2), [s3] "r"(s3),
                                [d0] "r"(d), [d1] "r"(d + 8), [d2] "r"(d + 16), [d3] "r"(d + 24)
                             : "memory" VCLOBBER);
    }
    else
    {
        // a column at a time: eight loads, then eight stores
        const char* const cs[4] = {s, s1, s2, s3};
        const char* const ps[4] = {p0, p1, p2, p3};
        for (int c = 0; c < 4; c++)
        {
            if constexpr (pf) __asm__ volatile("ori x0, %0, 1" ::"r"(ps[c]));
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
    for (int c = 0; c < 4; c++)
    {
        if constexpr (pf) touch<pf_l1>(s + c * lda + ahead);
        std::memcpy(d + 8 * c, s + c * lda, 64);
    }
    (void)vec;
#endif
}

// every panel of a rows x cols matrix (lda = rows doubles), into one panel buffer
template <bool vec, bool pf>
void pack_all(const double* a, std::size_t rows, std::size_t cols, std::size_t dist,
              double* panel, long reps)
{
    const std::size_t lda = rows * sizeof(double), ahead = dist * lda;
    for (long r = 0; r < reps; r++)
        for (std::size_t i = 0; i < rows; i += 8)
        {
            const char* col = reinterpret_cast<const char*>(a + i);
            for (std::size_t c = 0; c < cols; c += 4, col += 4 * lda)
                copy4<vec, pf>(col, lda, ahead, panel + 8 * c);
        }
}

double copy_cost(bool vec, std::size_t dist, const double* a, std::size_t rows,
                 std::size_t cols, double* panel, double min_time)
{
    auto run = [&](long reps) {
        const auto t0 = std::chrono::steady_clock::now();
        if (vec && dist)  pack_all<true, true>(a, rows, cols, dist, panel, reps);
        if (vec && !dist) pack_all<true, false>(a, rows, cols, 0, panel, reps);
        if (!vec && dist) pack_all<false, true>(a, rows, cols, dist, panel, reps);
        if (!vec && !dist) pack_all<false, false>(a, rows, cols, 0, panel, reps);
        const auto t1 = std::chrono::steady_clock::now();
        return std::chrono::duration<double, std::nano>(t1 - t0).count();
    };
    run(1);
    const long reps = std::max(1L, static_cast<long>(min_time * 1e9 / std::max(1.0, run(1))));
    double best = 1e300;
    for (int t = 0; t < 3; t++) best = std::min(best, run(reps));
    return best / (static_cast<double>(reps) * static_cast<double>(rows / 8 * cols));
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

int copy_mode(const std::vector<std::size_t>& sizes_kb, double scale, const char* unit,
              double min_time)
{
    const std::size_t cols = 240;                       // a k_c
    const bool vec_ok = vector_copy_fits();
    std::printf("# --copy: 8-row panels of a column-major matrix, %zu columns, one 64-byte\n"
                "# line per column, 4 columns per iteration; %s per column, prefetch.r D\n"
                "# columns ahead (0: none)%s\n", cols, unit,
                vec_ok ? "" : "\n# (no vector column: VLEN < 256)");
    for (const std::size_t kb : sizes_kb)
    {
        std::size_t rows = std::max<std::size_t>(8, kb * 1024 / sizeof(double) / cols / 8 * 8);
        if ((rows & (rows - 1)) == 0) rows += 8;        // no power-of-two lda
        std::vector<double> a(rows * cols), panel(8 * cols);
        for (std::size_t i = 0; i < a.size(); i++) a[i] = static_cast<double>(i);
        std::printf("\n# %zu KB: %zu x %zu, lda %zu bytes\n  %4s %9s %9s\n", kb, rows, cols,
                    rows * sizeof(double), "D", vec_ok ? "vector" : "-", "scalar");
        for (const std::size_t d : {0, 4, 8, 16, 32})
        {
            std::printf("  %4zu", d);
            if (vec_ok) std::printf(" %9.1f", copy_cost(true, d, a.data(), rows, cols, panel.data(), min_time) * scale);
            else std::printf(" %9s", "-");
            std::printf(" %9.1f\n", copy_cost(false, d, a.data(), rows, cols, panel.data(), min_time) * scale);
        }
        // the last panel's columns must have landed
        for (std::size_t c = 0; c < cols; c++)
            for (std::size_t r = 0; r < 8; r++)
                if (panel[8 * c + r] != a[(rows - 8 + r) + c * rows])
                {
                    std::fprintf(stderr, "copy check failed at column %zu row %zu\n", c, r);
                    return 1;
                }
        std::fflush(stdout);
    }
    return 0;
}

double ns_per_line(kind k, const std::size_t* buf, const std::size_t* order, std::size_t n,
                   std::size_t d, double min_time)
{
    auto run = [&](long reps) {
        const auto t0 = std::chrono::steady_clock::now();
        switch (k)
        {
        case chase_only: sink = walk<chase_only>(buf, order, n, d, reps); break;
        case pf_l1:      sink = walk<pf_l1>(buf, order, n, d, reps); break;
        case pf_l2:      sink = walk<pf_l2>(buf, order, n, d, reps); break;
        case load:       sink = walk<load>(buf, order, n, d, reps); break;
        }
        const auto t1 = std::chrono::steady_clock::now();
        return std::chrono::duration<double, std::nano>(t1 - t0).count();
    };
    run(1);                                                      // warm up
    const long reps = std::max(1L, static_cast<long>(min_time * 1e9 / std::max(1.0, run(1))));
    double best = 1e300;
    for (int t = 0; t < 3; t++) best = std::min(best, run(reps));
    return best / (static_cast<double>(reps) * static_cast<double>(n));
}

[[noreturn]] void usage(const char* argv0)
{
    std::fprintf(stderr, "usage: %s [--cpu N] [--ghz G] [--min-time S] [--huge] [--copy] "
                         "[SIZE_KB...]\n",
                 argv0);
    std::exit(1);
}
}  // namespace

int main(int argc, char** argv)
{
    int cpu = -1;
    double ghz = 0, min_time = 0.1;
    bool huge = false, copy = false;
    std::vector<std::size_t> sizes_kb;
    for (int i = 1; i < argc; i++)
    {
        const std::string a = argv[i];
        if (a == "--cpu" && i + 1 < argc) cpu = std::atoi(argv[++i]);
        else if (a == "--ghz" && i + 1 < argc) ghz = std::atof(argv[++i]);
        else if (a == "--min-time" && i + 1 < argc) min_time = std::atof(argv[++i]);
        else if (a == "--huge") huge = true;
        else if (a == "--copy") copy = true;
        else if (!a.empty() && a[0] != '-') sizes_kb.push_back(std::strtoull(a.c_str(), nullptr, 10));
        else usage(argv[0]);
    }
    if (sizes_kb.empty()) sizes_kb = {16, 256, 4096};
    if (cpu >= 0)
    {
        cpu_set_t set;
        CPU_ZERO(&set);
        CPU_SET(cpu, &set);
        if (sched_setaffinity(0, sizeof(set), &set) != 0) std::perror("sched_setaffinity");
    }
    const char* unit = ghz > 0 ? "cycles" : "ns";
    const double scale = ghz > 0 ? ghz : 1.0;
    if (copy) return copy_mode(sizes_kb, scale, unit, min_time);
    std::printf("# prefetch_test: a random-order chase over 64-byte lines, %s per line;\n"
                "# pf.L1, pf.L2 and load issued D lines ahead of the chase\n", unit);
    std::mt19937_64 rng(42);
    for (const std::size_t kb : sizes_kb)
    {
        const std::size_t n = std::max<std::size_t>(2, kb * 1024 / line);
        const std::size_t bytes = n * line, align = huge ? (2u << 20) : line;
        void* mem = nullptr;
        if (posix_memalign(&mem, align, (bytes + align - 1) / align * align) != 0)
        {
            std::perror("posix_memalign");
            return 1;
        }
        if (huge) madvise(mem, (bytes + align - 1) / align * align, MADV_HUGEPAGE);
        auto* buf = static_cast<std::size_t*>(mem);
        std::vector<std::size_t> order(n + max_distance);
        std::iota(order.begin(), order.begin() + n, std::size_t{0});
        std::shuffle(order.begin(), order.begin() + n, rng);
        for (std::size_t j = 0; j < max_distance; j++) order[n + j] = order[j % n];
        for (std::size_t i = 0; i < n; i++)
            for (std::size_t w = 0; w < words; w++)
                buf[order[i] * words + w] = order[(i + 1) % n];

        const double base = ns_per_line(chase_only, buf, order.data(), n, 0, min_time) * scale;
        std::printf("\n# %zu KB%s: chase alone %.1f %s per line\n", kb,
                    huge ? " (huge pages asked for)" : "", base, unit);
        std::printf("  %4s %9s %9s %9s\n", "D", kind_name[pf_l1], kind_name[pf_l2], kind_name[load]);
        for (const std::size_t d : distances)
        {
            std::printf("  %4zu", d);
            for (kind k : {pf_l1, pf_l2, load})
                std::printf(" %9.1f", ns_per_line(k, buf, order.data(), n, d, min_time) * scale);
            std::printf("\n");
        }
        std::fflush(stdout);
        std::free(mem);
    }
    return 0;
}
