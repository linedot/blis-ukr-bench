#pragma once
// The GOTO loop nest, built from BLIS's own kernels:
//
//   jc: for n   step n_c                         [pack B, if placed here]
//     pc: for k   step k_c                       [pack B_c / A, if here]
//       ic: for m   step m_c                     [pack A_c, if here]
//         jr: for n_c step n_r
//           ir: for m_c step m_r
//             micro-kernel, k_c steps
//
// The region from `top` down is executed: rungs above it run one iteration,
// and packing placed above it happens once, outside the timed region.
// Packing can sit at any rung from the one that first needs the block
// outward -- A at ic, pc or jc, B at pc or jc -- or 'pre': once, ever.
//
// Prefetches are issued between micro-kernel calls, for what the NEXT
// iteration of their loop uses, in execution order.  If the operand is
// packed above that loop, its packed block is prefetched; otherwise the
// source that its next pack will read.  The lead is counted in calls.

#include "blis_kernels.hpp"
#include "prefetch.hpp"
#include "sme_hoist.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(__linux__)
#include <sys/mman.h>
#endif

namespace gotobench {

// rungs, innermost first -- the same names and order as model.py's LOOPS
enum rung : int { R_UKR = 0, R_IR = 1, R_JR = 2, R_IC = 3, R_PC = 4, R_JC = 5, R_PRE = 6 };

inline const char* rung_name(int r)
{
    static const char* names[] = {"ukr", "ir", "jr", "ic", "pc", "jc", "pre"};
    return (r >= 0 && r <= R_PRE) ? names[r] : "?";
}

inline int parse_rung(const std::string& s)
{
    for (int r = R_UKR; r <= R_PRE; r++)
        if (s == rung_name(r)) return r;
    throw std::invalid_argument("unknown rung '" + s + "'");
}

// operands that change across each loop: the only ones there is anything new
// to prefetch for (model.py's VARYING, without the k-loop)
inline bool varies(int loop, char op)
{
    switch (loop)
    {
        case R_IR: return op == 'A' || op == 'C';
        case R_JR: return op == 'B' || op == 'C';
        case R_IC: return op == 'A' || op == 'C';
        case R_PC: return op == 'A' || op == 'B';
        case R_JC: return op == 'B' || op == 'C';
        default: return false;
    }
}

struct prefetch_spec
{
    int loop = R_IR;
    char operand = 'C';
    pf_kind kind{};
    long lead = 0;           // calls left in the loop's iteration; 0: its start
    std::string text;        // as given
};

struct nest_config
{
    dim_t m = 0, n = 0, k = 0;
    dim_t mc = 0, nc = 0, kc = 0;
    int top = R_JC;
    int pack_a = R_IC, pack_b = R_PC;
    bool c_row_major = true;
    double alpha = 2.1, beta = 1.3;
    std::vector<prefetch_spec> prefetches;
    std::size_t line = 64;
    bool hugepages = false;
};

// One cache line per call of f, over a strided 2-D region of doubles.
template <class F>
void for_each_line(const double* base, dim_t rows, dim_t cols, inc_t rs,
                   inc_t cs, std::size_t line, F f)
{
    if (rows <= 0 || cols <= 0) return;
    const bool rows_inner = (rs == 1) || (cols == 1);
    const dim_t outer = rows_inner ? cols : rows, inner = rows_inner ? rows : cols;
    const inc_t so = rows_inner ? cs : rs, si = rows_inner ? rs : cs;
    const std::uintptr_t mask = ~static_cast<std::uintptr_t>(line - 1);
    for (dim_t o = 0; o < outer; o++)
    {
        const double* run = base + o * so;
        if (si == 1)
        {
            auto a = reinterpret_cast<std::uintptr_t>(run) & mask;
            const auto end = reinterpret_cast<std::uintptr_t>(run + inner);
            for (; a < end; a += line) f(reinterpret_cast<const void*>(a));
        }
        else
        {
            std::uintptr_t last = 0;
            for (dim_t i = 0; i < inner; i++)
            {
                auto a = reinterpret_cast<std::uintptr_t>(run + i * si) & mask;
                if (a != last) f(reinterpret_cast<const void*>(a));
                last = a;
            }
        }
    }
}

class goto_nest
{
public:
    goto_nest(const blis_kernels& ker, const nest_config& cfg) : ker_(ker), c_(cfg)
    {
        validate();
        mr = ker.mr; nr = ker.nr;
        n_ic = ceil_div(c_.m, c_.mc); n_pc = ceil_div(c_.k, c_.kc);
        n_jc = ceil_div(c_.n, c_.nc);
        lda = c_.m; ldb = c_.k;
        rs_c = c_.c_row_major ? c_.n : 1;
        cs_c = c_.c_row_major ? 1 : c_.m;
        ps_a = ker.ldp_a() * c_.kc;
        ps_b = ker.ldp_b() * c_.kc;
        block_a = ceil_div(c_.mc, mr) * ps_a;
        block_b = ceil_div(c_.nc, nr) * ps_b;
        jc_hi = c_.top >= R_JC ? n_jc : 1;
        pc_hi = c_.top >= R_PC ? n_pc : 1;
        ic_hi = c_.top >= R_IC ? n_ic : 1;

        A = alloc(static_cast<std::size_t>(c_.m) * c_.k);
        B = alloc(static_cast<std::size_t>(c_.k) * c_.n);
        C = alloc(static_cast<std::size_t>(c_.m) * c_.n);
        Ap = alloc(static_cast<std::size_t>(slots_a()) * block_a);
        Bp = alloc(static_cast<std::size_t>(slots_b()) * block_b);
        fill(A, static_cast<std::size_t>(c_.m) * c_.k, 1);
        fill(B, static_cast<std::size_t>(c_.k) * c_.n, 2);
        fill(C, static_cast<std::size_t>(c_.m) * c_.n, 3);

        bli_auxinfo_set_is_a(1, &aux);
        bli_auxinfo_set_is_b(1, &aux);
#if defined(AOCL_BLIS)
        bli_auxinfo_set_schema_a(BLIS_PACKED_ROW_PANELS, &aux);
        bli_auxinfo_set_schema_b(BLIS_PACKED_COL_PANELS, &aux);
#else
        bli_auxinfo_set_schema_a(BLIS_PACKED_PANELS, &aux);
        bli_auxinfo_set_schema_b(BLIS_PACKED_PANELS, &aux);
#endif
        bli_auxinfo_set_ps_a(ps_a, &aux);
        bli_auxinfo_set_ps_b(ps_b, &aux);
        prepack();
    }

    ~goto_nest()
    {
        for (double* p : {A, B, C, Ap, Bp}) std::free(p);
    }
    goto_nest(const goto_nest&) = delete;
    goto_nest& operator=(const goto_nest&) = delete;

    // One execution of the region from `top` down.
    void run()
    {
        for (dim_t jc = 0; jc < jc_hi; jc++)
        {
            if (packs_here(c_.pack_b, R_JC))
                for (dim_t pc = 0; pc < n_pc; pc++) pack_b_block(jc, pc);
            if (packs_here(c_.pack_a, R_JC))
                for (dim_t pc = 0; pc < n_pc; pc++)
                    for (dim_t ic = 0; ic < n_ic; ic++) pack_a_block(pc, ic);
            enter(R_JC, calls_in_jc(jc));
            for (dim_t pc = 0; pc < pc_hi; pc++)
            {
                if (packs_here(c_.pack_b, R_PC)) pack_b_block(jc, pc);
                if (packs_here(c_.pack_a, R_PC))
                    for (dim_t ic = 0; ic < n_ic; ic++) pack_a_block(pc, ic);
                enter(R_PC, calls_in_pc(jc));
                for (dim_t ic = 0; ic < ic_hi; ic++)
                {
                    if (packs_here(c_.pack_a, R_IC)) pack_a_block(pc, ic);
                    enter(R_IC, jr_hi(jc) * ir_hi(ic));
                    macro(jc, pc, ic);
                }
            }
        }
    }

    // Pack one A_c, or one B_c, on its own -- for measuring the pack rate.
    void run_pack_a() { pack_a_block(0, 0); }
    void run_pack_b() { pack_b_block(0, 0); }
    dim_t pack_a_elems() const { return std::min(c_.mc, c_.m) * std::min(c_.kc, c_.k); }
    dim_t pack_b_elems() const { return std::min(c_.nc, c_.n) * std::min(c_.kc, c_.k); }

    // Extent of one run, and its work.
    dim_t m_exec() const { return c_.top >= R_IC ? c_.m : c_.top >= R_IR ? mcur(0) : std::min(mr, mcur(0)); }
    dim_t n_exec() const { return c_.top >= R_JC ? c_.n : c_.top >= R_JR ? ncur(0) : std::min(nr, ncur(0)); }
    dim_t k_exec() const { return c_.top >= R_PC ? c_.k : kcur(0); }
    double flops() const { return 2.0 * m_exec() * n_exec() * k_exec(); }
    double calls() const
    {
        double s = 0;
        for (dim_t jc = 0; jc < jc_hi; jc++) s += static_cast<double>(calls_in_jc(jc));
        return s;
    }
    std::size_t footprint_bytes() const
    {
        return 8 * (static_cast<std::size_t>(c_.m) * c_.k + static_cast<std::size_t>(c_.k) * c_.n
                    + static_cast<std::size_t>(c_.m) * c_.n
                    + static_cast<std::size_t>(slots_a()) * block_a
                    + static_cast<std::size_t>(slots_b()) * block_b);
    }

    // Verification: run once from a known C and compare the executed region
    // with a reference, beta*C0 + alpha*A*B over the same m_exec x n_exec x
    // k_exec, computed either by a plain loop or by the linked library's own
    // GEMM (fast enough for full-size problems).  Nothing outside the region
    // may change, and C is restored.  Returns the largest relative error.
    enum class reference { loop, library };

    static const char* library_gemm()
    {
#if defined(BLIS_ENABLE_CBLAS)
        return "cblas_dgemm";
#else
        return "bli_dgemm: the library has no CBLAS layer";
#endif
    }

    double verify(reference how = reference::loop)
    {
        const std::size_t mn = static_cast<std::size_t>(c_.m) * c_.n;
        std::vector<double> c0(C, C + mn);
        run();
        const dim_t me = m_exec(), ne = n_exec(), ke = k_exec();
        double worst = 0.0;
        auto check = [&](dim_t i, dim_t j, double ref) {
            const std::size_t ij = i * rs_c + j * cs_c;
            worst = std::max(worst, std::fabs(C[ij] - ref) / (std::fabs(ref) + 1.0));
        };
        if (how == reference::library)
        {
            std::vector<double> ref(c0);
            library_gemm(me, ne, ke, ref.data());
            for (dim_t j = 0; j < ne; j++)
                for (dim_t i = 0; i < me; i++) check(i, j, ref[i * rs_c + j * cs_c]);
        }
        else
        {
            std::vector<double> acc(static_cast<std::size_t>(me));
            for (dim_t j = 0; j < ne; j++)
            {
                std::fill(acc.begin(), acc.end(), 0.0);
                for (dim_t p = 0; p < ke; p++)
                {
                    const double bpj = B[p + j * ldb];
                    const double* a = A + p * lda;
                    for (dim_t i = 0; i < me; i++) acc[i] += a[i] * bpj;
                }
                for (dim_t i = 0; i < me; i++)
                    check(i, j, c_.beta * c0[i * rs_c + j * cs_c] + c_.alpha * acc[i]);
            }
        }
        // nothing outside the region may have been touched
        for (dim_t i = 0; i < c_.m; i++)
            for (dim_t j = 0; j < c_.n; j++)
                if ((i >= me || j >= ne) && C[i * rs_c + j * cs_c] != c0[i * rs_c + j * cs_c])
                    return 1.0;
        std::copy(c0.begin(), c0.end(), C);
        return worst;
    }

private:
    // The library's GEMM on the region, into c (laid out as C).  A and B are
    // column-major; CBLAS wants one order for all three, so a row-major C
    // gets them as transposed row-major matrices.
    void library_gemm(dim_t me, dim_t ne, dim_t ke, double* c) const
    {
#if defined(BLIS_ENABLE_CBLAS)
        if (c_.c_row_major)
            cblas_dgemm(CblasRowMajor, CblasTrans, CblasTrans, me, ne, ke, c_.alpha,
                        A, lda, B, ldb, c_.beta, c, rs_c);
        else
            cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, me, ne, ke, c_.alpha,
                        A, lda, B, ldb, c_.beta, c, cs_c);
#else
        double alpha = c_.alpha, beta = c_.beta;
        bli_dgemm(BLIS_NO_TRANSPOSE, BLIS_NO_TRANSPOSE, me, ne, ke, &alpha,
                  A, 1, lda, B, 1, ldb, &beta, c, rs_c, cs_c);
#endif
    }

    const blis_kernels& ker_;
    nest_config c_;
    dim_t mr = 0, nr = 0, n_ic = 0, n_pc = 0, n_jc = 0;
    dim_t jc_hi = 1, pc_hi = 1, ic_hi = 1;
    inc_t lda = 0, ldb = 0, rs_c = 0, cs_c = 0, ps_a = 0, ps_b = 0;
    dim_t block_a = 0, block_b = 0;
    double *A = nullptr, *B = nullptr, *C = nullptr, *Ap = nullptr, *Bp = nullptr;
    auxinfo_t aux{};
    // calls made so far in the current iteration of each loop, and its total
    long done[R_PRE + 1] = {};
    long total[R_PRE + 1] = {};
public:
    long prefetched = 0;        // lines prefetched (test builds only)
private:

    static dim_t ceil_div(dim_t a, dim_t b) { return (a + b - 1) / b; }
    dim_t mcur(dim_t ic) const { return std::min(c_.mc, c_.m - ic * c_.mc); }
    dim_t ncur(dim_t jc) const { return std::min(c_.nc, c_.n - jc * c_.nc); }
    dim_t kcur(dim_t pc) const { return std::min(c_.kc, c_.k - pc * c_.kc); }
    dim_t ir_hi(dim_t ic) const { return c_.top >= R_IR ? ceil_div(mcur(ic), mr) : 1; }
    dim_t jr_hi(dim_t jc) const { return c_.top >= R_JR ? ceil_div(ncur(jc), nr) : 1; }
    long calls_in_pc(dim_t jc) const
    {
        long s = 0;
        for (dim_t ic = 0; ic < ic_hi; ic++) s += ir_hi(ic);
        return s * jr_hi(jc);
    }
    long calls_in_jc(dim_t jc) const { return pc_hi * calls_in_pc(jc); }
    bool packs_here(int at, int r) const { return at == r && c_.top >= r; }

    dim_t slots_a() const
    {
        return c_.pack_a == R_IC ? 1 : c_.pack_a == R_PC ? n_ic : n_pc * n_ic;
    }
    dim_t slots_b() const
    {
        return c_.pack_b == R_PC ? 1 : c_.pack_b == R_JC ? n_pc : n_jc * n_pc;
    }
    dim_t slot_a(dim_t pc, dim_t ic) const
    {
        return c_.pack_a == R_IC ? 0 : c_.pack_a == R_PC ? ic : pc * n_ic + ic;
    }
    dim_t slot_b(dim_t jc, dim_t pc) const
    {
        return c_.pack_b == R_PC ? 0 : c_.pack_b == R_JC ? pc : jc * n_pc + pc;
    }

    void validate() const
    {
        if (c_.m <= 0 || c_.n <= 0 || c_.k <= 0 || c_.mc <= 0 || c_.nc <= 0 || c_.kc <= 0)
            throw std::invalid_argument("sizes and blocks must be positive");
        if (c_.pack_a < R_IC || c_.pack_a > R_PRE || c_.pack_a == R_JR)
            throw std::invalid_argument("A can be packed at ic, pc, jc or pre");
        if (c_.pack_b < R_PC || c_.pack_b > R_PRE)
            throw std::invalid_argument("B can be packed at pc, jc or pre");
        for (const auto& s : c_.prefetches)
        {
            if (!varies(s.loop, s.operand))
                throw std::invalid_argument(
                    "prefetch " + s.text + ": " + s.operand + " is invariant across "
                    + rung_name(s.loop) + ", so there is nothing new to fetch");
            if (s.loop > c_.top)
                throw std::invalid_argument(
                    "prefetch " + s.text + ": " + rung_name(s.loop)
                    + " does not iterate below --top " + rung_name(c_.top));
        }
    }

    double* alloc(std::size_t elems) const
    {
        const std::size_t align = c_.hugepages ? (2u << 20) : 4096;
        std::size_t bytes = std::max<std::size_t>(elems * sizeof(double), align);
        bytes = (bytes + align - 1) / align * align;
        void* p = nullptr;
        if (posix_memalign(&p, align, bytes) != 0) throw std::bad_alloc();
#if defined(__linux__) && defined(MADV_HUGEPAGE)
        if (c_.hugepages) madvise(p, bytes, MADV_HUGEPAGE);
#endif
        std::memset(p, 0, bytes);
        return static_cast<double*>(p);
    }

    static void fill(double* x, std::size_t n, unsigned seed)
    {
        // deterministic, no denormals, mean near zero
        for (std::size_t i = 0; i < n; i++)
            x[i] = 0.25 * static_cast<double>(static_cast<int>((i * 2654435761u + seed * 97u) % 9u) - 4);
    }

    void pack_a_block(dim_t pc, dim_t ic)
    {
        const dim_t i0 = ic * c_.mc, m_c = mcur(ic), p0 = pc * c_.kc, k_c = kcur(pc);
        double* dst = Ap + slot_a(pc, ic) * block_a;
        for (dim_t i = 0; i < m_c; i += mr)
            ker_.pack_a(std::min(mr, m_c - i), k_c, A + (i0 + i) + p0 * lda, 1, lda,
                        dst + (i / mr) * ps_a);
    }

    void pack_b_block(dim_t jc, dim_t pc)
    {
        const dim_t j0 = jc * c_.nc, n_c = ncur(jc), p0 = pc * c_.kc, k_c = kcur(pc);
        double* dst = Bp + slot_b(jc, pc) * block_b;
        for (dim_t j = 0; j < n_c; j += nr)
            ker_.pack_b(std::min(nr, n_c - j), k_c, B + p0 + (j0 + j) * ldb, ldb, 1,
                        dst + (j / nr) * ps_b);
    }

    // Packing placed above the region happens once, for what the region reads.
    void prepack()
    {
        if (!packs_inside(c_.pack_a))
            for (dim_t pc = 0; pc < pc_hi; pc++)
                for (dim_t ic = 0; ic < ic_hi; ic++) pack_a_block(pc, ic);
        if (!packs_inside(c_.pack_b))
            for (dim_t jc = 0; jc < jc_hi; jc++)
                for (dim_t pc = 0; pc < pc_hi; pc++) pack_b_block(jc, pc);
    }
    bool packs_inside(int at) const { return at <= c_.top; }

    void enter(int r, long calls)
    {
        done[r] = 0;
        total[r] = calls;
    }

    void macro(dim_t jc, dim_t pc, dim_t ic)
    {
        const dim_t i0 = ic * c_.mc, m_c = mcur(ic), j0 = jc * c_.nc, n_c = ncur(jc);
        const dim_t k_c = kcur(pc);
        const double* a_blk = Ap + slot_a(pc, ic) * block_a;
        const double* b_blk = Bp + slot_b(jc, pc) * block_b;
        const double beta = pc == 0 ? c_.beta : 1.0;
        const dim_t jr_n = jr_hi(jc), ir_n = ir_hi(ic);
        const bool hooks = !c_.prefetches.empty();
        BENCH_SME_ENTER();
        for (dim_t jr = 0; jr < jr_n; jr++)
        {
            const double* b1 = b_blk + jr * ps_b;
            const dim_t n_r = std::min(nr, n_c - jr * nr);
            enter(R_JR, ir_n);
            for (dim_t ir = 0; ir < ir_n; ir++)
            {
                const double* a1 = a_blk + ir * ps_a;
                const dim_t m_r = std::min(mr, m_c - ir * mr);
                // next panels exactly as BLIS's macro-kernel sets them
                const bool last_ir = ir == ir_n - 1;
                bli_auxinfo_set_next_a(last_ir ? a_blk : a1 + ps_a, &aux);
                bli_auxinfo_set_next_b(last_ir ? (jr == jr_n - 1 ? b_blk : b1 + ps_b) : b1, &aux);
                enter(R_IR, 1);
                if (hooks) prefetch_hook(jc, pc, ic, jr, ir);
                double* c11 = C + (i0 + ir * mr) * rs_c + (j0 + jr * nr) * cs_c;
                ker_.gemm(m_r, n_r, k_c, &c_.alpha, a1, b1, &beta, c11, rs_c, cs_c, &aux);
                for (int r = R_IR; r <= R_JC; r++) done[r]++;
            }
        }
        BENCH_SME_EXIT();
    }

    // -- prefetching --------------------------------------------------------

    struct position { dim_t jc, pc, ic, jr, ir; };

    // The next iteration of loop `r` after `p`, in execution order: an
    // odometer that carries into the enclosing loops and wraps at the top --
    // the region is repeated, so after the last iteration comes the first.
    position next_at(int r, position p) const
    {
        dim_t* idx[] = {nullptr, &p.ir, &p.jr, &p.ic, &p.pc, &p.jc};
        for (int i = R_IR; i < r; i++) *idx[i] = 0;
        for (int i = r; i <= R_JC; i++)
        {
            // bounds of the loops above i are still those of the current
            // iteration, so the edge-block trip counts are right
            const dim_t hi = i == R_IR ? ir_hi(p.ic) : i == R_JR ? jr_hi(p.jc)
                           : i == R_IC ? ic_hi : i == R_PC ? pc_hi : jc_hi;
            if (++*idx[i] < hi) return p;
            *idx[i] = 0;
        }
        return position{0, 0, 0, 0, 0};
    }

    void prefetch_hook(dim_t jc, dim_t pc, dim_t ic, dim_t jr, dim_t ir)
    {
        const position now{jc, pc, ic, jr, ir};
        for (const auto& s : c_.prefetches)
        {
            // a lead of 0, or longer than the iteration, means its start --
            // capped at one iteration, as model.py caps it
            const long left = total[s.loop] - done[s.loop];
            const long want = (s.lead <= 0 || s.lead > total[s.loop]) ? total[s.loop] : s.lead;
            const bool due = left == want;
            if (!due) continue;
            const position nx = next_at(s.loop, now);
            issue(s, nx);
        }
    }

    // Which buffer a prefetch must land in -- checked in test builds, since a
    // wrong prefetch address never faults and so is otherwise invisible.
    const double* expected_buffer(const prefetch_spec& s) const
    {
        if (s.operand == 'C') return C;
        if (s.operand == 'A') return (s.loop == R_IR || c_.pack_a > s.loop) ? Ap : A;
        return (s.loop == R_JR || c_.pack_b > s.loop) ? Bp : B;
    }
    std::size_t buffer_bytes(const double* b) const
    {
        const std::size_t e = b == A ? static_cast<std::size_t>(c_.m) * c_.k
                            : b == B ? static_cast<std::size_t>(c_.k) * c_.n
                            : b == C ? static_cast<std::size_t>(c_.m) * c_.n
                            : b == Ap ? static_cast<std::size_t>(slots_a()) * block_a
                            : static_cast<std::size_t>(slots_b()) * block_b;
        return e * sizeof(double);
    }

    void issue(const prefetch_spec& s, const position& q)
    {
#if defined(GOTOBENCH_CHECK_PREFETCH)
        const double* want = expected_buffer(s);
        const auto lo = reinterpret_cast<std::uintptr_t>(want);
        const auto hi = lo + buffer_bytes(want);
        auto pf = [&, lo, hi](const void* p) {
            const auto a = reinterpret_cast<std::uintptr_t>(p);
            if (a < (lo & ~static_cast<std::uintptr_t>(c_.line - 1)) || a >= hi)
            {
                std::fprintf(stderr, "prefetch %s: line %p outside its buffer [%p, %p)\n",
                             s.text.c_str(), p, reinterpret_cast<void*>(lo),
                             reinterpret_cast<void*>(hi));
                std::abort();
            }
            prefetch(p, s.kind);
            prefetched++;
        };
#else
        auto pf = [&](const void* p) { prefetch(p, s.kind); };
#endif
        const std::size_t L = c_.line;
        const dim_t i0 = q.ic * c_.mc, j0 = q.jc * c_.nc, p0 = q.pc * c_.kc;
        const dim_t k_c = kcur(q.pc);
        switch (s.operand)
        {
            case 'C':
            {
                dim_t r0 = i0, rows = mcur(q.ic), c0 = j0, cols = ncur(q.jc);
                if (s.loop <= R_JR) { c0 += q.jr * nr; cols = std::min(nr, cols - q.jr * nr); }
                if (s.loop == R_IR) { r0 += q.ir * mr; rows = std::min(mr, rows - q.ir * mr); }
                if (s.loop == R_JC) { r0 = 0; rows = c_.top >= R_IC ? c_.m : mcur(0); }
                for_each_line(C + r0 * rs_c + c0 * cs_c, rows, cols, rs_c, cs_c, L, pf);
                break;
            }
            case 'A':
            {
                if (s.loop == R_IR)
                {
                    const double* a = Ap + slot_a(q.pc, q.ic) * block_a + q.ir * ps_a;
                    for_each_line(a, ker_.ldp_a() * k_c, 1, 1, 1, L, pf);
                }
                else if (c_.pack_a > s.loop)   // already packed: the packed block(s)
                {
                    const dim_t n = s.loop == R_IC ? 1 : n_ic;
                    const double* a = Ap + slot_a(q.pc, s.loop == R_IC ? q.ic : 0) * block_a;
                    for_each_line(a, n * block_a, 1, 1, 1, L, pf);
                }
                else                            // the source the next pack reads
                {
                    const dim_t r0 = s.loop == R_IC ? i0 : 0;
                    const dim_t rows = s.loop == R_IC ? mcur(q.ic) : c_.m;
                    for_each_line(A + r0 + p0 * lda, rows, k_c, 1, lda, L, pf);
                }
                break;
            }
            case 'B':
            {
                if (s.loop == R_JR)
                {
                    const double* b = Bp + slot_b(q.jc, q.pc) * block_b + q.jr * ps_b;
                    for_each_line(b, ker_.ldp_b() * k_c, 1, 1, 1, L, pf);
                }
                else if (c_.pack_b > s.loop)
                {
                    const dim_t n = s.loop == R_PC ? 1 : n_pc;
                    const double* b = Bp + slot_b(q.jc, s.loop == R_PC ? q.pc : 0) * block_b;
                    for_each_line(b, n * block_b, 1, 1, 1, L, pf);
                }
                else
                {
                    const dim_t rows = s.loop == R_PC ? k_c : c_.k;
                    const dim_t pr0 = s.loop == R_PC ? p0 : 0;
                    for_each_line(B + pr0 + j0 * ldb, rows, ncur(q.jc), 1, ldb, L, pf);
                }
                break;
            }
            default: break;
        }
    }
};

} // namespace gotobench
