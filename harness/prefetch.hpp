#pragma once
// Software prefetch, per ISA.  What each one can express differs, so
// describe() says what a request actually turns into on this build -- the
// model's target level and hint are only as real as that.
//
//   x86-64    prefetcht0/t1/t2 choose L1/L2/L3; 'strm' becomes prefetchnta
//             (all levels, minimal pollution); write intent uses prefetchw.
//   AArch64   PRFM {PLD,PST}{L1,L2,L3}{KEEP,STRM}: every combination exists.
//   RISC-V    Zicbop prefetch.r / prefetch.w, with a Zihintntl hint in front
//             to pick the level: ntl.p1 (not temporal in the innermost
//             private level) lands in L2, ntl.pall (in any private level) in
//             the innermost shared one -- L3 where L2 is private, L2 where it
//             is shared, as on K1.  No strm/keep distinction.  Both are base-
//             ISA HINT encodings, so they are emitted raw: no -march flag
//             needed, and a core without the extensions runs them as no-ops.

#include <string>

namespace gotobench {

enum class pf_level { L1 = 1, L2 = 2, L3 = 3 };

struct pf_kind
{
    pf_level level = pf_level::L2;
    bool stream = false;   // 'strm': insert near LRU
    bool write = false;    // the line will be written (C)
};

#if defined(__aarch64__)
#define GOTOBENCH_PRFM(op, p) __asm__ volatile("prfm " op ", [%0]" ::"r"(p))
#endif

// Every ISA uses asm volatile, which no optimisation may remove.  GCC's tree
// DCE drops __builtin_prefetch here once prefetch() is inlined -- silently,
// as its documentation allows -- so the builtin is only the last resort.
inline void prefetch(const void* p, pf_kind k)
{
#if defined(__x86_64__) || defined(__i386__)
    // memory operands, so this assembles under -masm=intel as well
    const char& m = *static_cast<const char*>(p);
    if (k.write)
        __asm__ volatile("prefetchw %0" ::"m"(m));
    else if (k.stream)
        __asm__ volatile("prefetchnta %0" ::"m"(m));
    else if (k.level == pf_level::L1)
        __asm__ volatile("prefetcht0 %0" ::"m"(m));
    else if (k.level == pf_level::L2)
        __asm__ volatile("prefetcht1 %0" ::"m"(m));
    else
        __asm__ volatile("prefetcht2 %0" ::"m"(m));
#elif defined(__aarch64__)
    const int sel = (k.write ? 6 : 0) + (static_cast<int>(k.level) - 1) * 2
                    + (k.stream ? 1 : 0);
    switch (sel)
    {
        case 0:  GOTOBENCH_PRFM("pldl1keep", p); break;
        case 1:  GOTOBENCH_PRFM("pldl1strm", p); break;
        case 2:  GOTOBENCH_PRFM("pldl2keep", p); break;
        case 3:  GOTOBENCH_PRFM("pldl2strm", p); break;
        case 4:  GOTOBENCH_PRFM("pldl3keep", p); break;
        case 5:  GOTOBENCH_PRFM("pldl3strm", p); break;
        case 6:  GOTOBENCH_PRFM("pstl1keep", p); break;
        case 7:  GOTOBENCH_PRFM("pstl1strm", p); break;
        case 8:  GOTOBENCH_PRFM("pstl2keep", p); break;
        case 9:  GOTOBENCH_PRFM("pstl2strm", p); break;
        case 10: GOTOBENCH_PRFM("pstl3keep", p); break;
        default: GOTOBENCH_PRFM("pstl3strm", p); break;
    }
#elif defined(__riscv)
    // raw encodings, so no assembler support is needed:
    //   prefetch.r 0(rs1) = ori x0, rs1, 1     prefetch.w 0(rs1) = ori x0, rs1, 3
    //   ntl.p1 = add x0, x0, x2                ntl.pall = add x0, x0, x3
    // An NTL hint applies to the access right after it: one asm statement
    // keeps the compiler from separating them.
    if (k.level == pf_level::L1)
    {
        if (k.write) __asm__ volatile("ori x0, %0, 3" ::"r"(p));
        else         __asm__ volatile("ori x0, %0, 1" ::"r"(p));
    }
    else if (k.level == pf_level::L2)
    {
        if (k.write) __asm__ volatile("add x0, x0, x2\n\tori x0, %0, 3" ::"r"(p));
        else         __asm__ volatile("add x0, x0, x2\n\tori x0, %0, 1" ::"r"(p));
    }
    else
    {
        if (k.write) __asm__ volatile("add x0, x0, x3\n\tori x0, %0, 3" ::"r"(p));
        else         __asm__ volatile("add x0, x0, x3\n\tori x0, %0, 1" ::"r"(p));
    }
#else
    __builtin_prefetch(p, k.write ? 1 : 0, 3);   // may be optimised away
#endif
}

// What prefetch(k) emits on this build, for the run's header.
inline std::string describe(pf_kind k)
{
    const char lv = static_cast<char>('0' + static_cast<int>(k.level));
#if defined(__x86_64__) || defined(__i386__)
    (void)lv;
    if (k.write) return "prefetchw (level not selectable)";
    if (k.stream) return "prefetchnta (level not selectable)";
    return std::string("prefetcht") + static_cast<char>('0' + static_cast<int>(k.level) - 1);
#elif defined(__aarch64__)
    return std::string("prfm ") + (k.write ? "pst" : "pld") + "l" + lv
           + (k.stream ? "strm" : "keep");
#elif defined(__riscv)
    (void)lv;
    const std::string op = k.write ? "prefetch.w" : "prefetch.r";
    const std::string strm = k.stream ? "; strm not expressible, kept" : "";
    if (k.level == pf_level::L1) return op + " (Zicbop" + strm + ")";
    if (k.level == pf_level::L2)
        return "ntl.p1 + " + op + " (Zihintntl + Zicbop: skips the innermost private level"
               + strm + ")";
    return "ntl.pall + " + op + " (Zihintntl + Zicbop: skips every private level, lands in"
           " the innermost shared one" + strm + ")";
#else
    (void)lv;
    return "__builtin_prefetch (no ISA prefetch known for this build; the compiler may drop it)";
#endif
}

} // namespace gotobench
