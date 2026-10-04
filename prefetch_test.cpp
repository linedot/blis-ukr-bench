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
// usage: prefetch_test [--cpu N] [--ghz G] [--min-time S] [--huge] [SIZE_KB...]
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
    std::fprintf(stderr, "usage: %s [--cpu N] [--ghz G] [--min-time S] [--huge] [SIZE_KB...]\n",
                 argv0);
    std::exit(1);
}
}  // namespace

int main(int argc, char** argv)
{
    int cpu = -1;
    double ghz = 0, min_time = 0.1;
    bool huge = false;
    std::vector<std::size_t> sizes_kb;
    for (int i = 1; i < argc; i++)
    {
        const std::string a = argv[i];
        if (a == "--cpu" && i + 1 < argc) cpu = std::atoi(argv[++i]);
        else if (a == "--ghz" && i + 1 < argc) ghz = std::atof(argv[++i]);
        else if (a == "--min-time" && i + 1 < argc) min_time = std::atof(argv[++i]);
        else if (a == "--huge") huge = true;
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
