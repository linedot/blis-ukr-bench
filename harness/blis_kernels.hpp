#pragma once
// Uniform access to the two BLIS kernels the GOTO harness drives -- the gemm
// micro-kernel and the pack kernel -- for BLIS >= 1.0 and for AOCL-BLIS,
// which keeps the older 0.x interface.
//
// Function-pointer types are declared here from the documented signatures
// rather than taken from BLIS's own typedefs, whose names changed between
// versions.

extern "C" {
#include <blis.h>
}

#include <cstdio>
#include <stdexcept>

namespace gotobench {

class blis_kernels
{
public:
    dim_t mr = 0, nr = 0;          // register blocking of the micro-kernel
    dim_t bbm = 1, bbn = 1;        // broadcast factors of the packed panels
    dim_t mc = 0, nc = 0, kc = 0;  // BLIS's own default cache blocking
    const char* impl = "";         // micro-kernel implementation string
    bool fallback_pack_a = false;  // no BLIS pack kernel: plain loop instead
    bool fallback_pack_b = false;
    bool prefers_rows = false;     // C storage the micro-kernel is fastest with;
                                   // BLIS transposes the problem to match it

    blis_kernels()
    {
        bli_init();
        impl = bli_info_get_gemm_ukr_impl_string(BLIS_NAT, BLIS_DOUBLE);
#if defined(AOCL_BLIS)
        cntx = bli_gks_query_nat_cntx();
        gemm_old = reinterpret_cast<gemm_old_ft>(
            bli_cntx_get_l3_nat_ukr_dt(BLIS_DOUBLE, BLIS_GEMM_UKR, cntx));
#else
        cntx = bli_gks_query_cntx();
        gemm_new = reinterpret_cast<gemm_new_ft>(
            bli_cntx_get_ukr_dt(BLIS_DOUBLE, BLIS_GEMM_UKR, cntx));
#endif
#if defined(AOCL_BLIS)
        prefers_rows = !bli_cntx_l3_nat_ukr_prefers_cols_dt(BLIS_DOUBLE, BLIS_GEMM_UKR, cntx);
#else
        prefers_rows = bli_cntx_ukr_prefers_rows_dt(BLIS_DOUBLE, BLIS_GEMM_UKR, cntx);
#endif
        mr = bli_cntx_get_blksz_def_dt(BLIS_DOUBLE, BLIS_MR, cntx);
        nr = bli_cntx_get_blksz_def_dt(BLIS_DOUBLE, BLIS_NR, cntx);
        mc = bli_cntx_get_blksz_def_dt(BLIS_DOUBLE, BLIS_MC, cntx);
        nc = bli_cntx_get_blksz_def_dt(BLIS_DOUBLE, BLIS_NC, cntx);
        kc = bli_cntx_get_blksz_def_dt(BLIS_DOUBLE, BLIS_KC, cntx);
#if defined(AOCL_BLIS)
        // the 0.x pack kernels are looked up by panel width
        pack_old_a = reinterpret_cast<pack_old_ft>(bli_cntx_get_packm_ker_dt(
            BLIS_DOUBLE, static_cast<l1mkr_t>(mr), cntx));
        pack_old_b = reinterpret_cast<pack_old_ft>(bli_cntx_get_packm_ker_dt(
            BLIS_DOUBLE, static_cast<l1mkr_t>(nr), cntx));
        fallback_pack_a = pack_old_a == nullptr;
        fallback_pack_b = pack_old_b == nullptr;
#else
        bbm = bli_cntx_get_blksz_def_dt(BLIS_DOUBLE, BLIS_BBM, cntx);
        bbn = bli_cntx_get_blksz_def_dt(BLIS_DOUBLE, BLIS_BBN, cntx);
        if (bbm < 1) bbm = 1;
        if (bbn < 1) bbn = 1;
        // one kernel for every panel width; it dispatches to the
        // configuration's optimised mr/nr pack kernels for full panels
        pack_new = reinterpret_cast<pack_new_ft>(
            bli_cntx_get_ukr_dt(BLIS_DOUBLE, BLIS_PACKM_KER, cntx));
        fallback_pack_a = fallback_pack_b = pack_new == nullptr;
#endif
        if (fallback_pack_a || fallback_pack_b)
            std::fprintf(stderr, "# warning: no BLIS pack kernel for %s; using a "
                         "plain loop, so pack timings are not BLIS's\n",
                         fallback_pack_a && fallback_pack_b ? "A or B"
                         : fallback_pack_a ? "A" : "B");
    }

    // Leading dimension of a packed panel: one k-column of the panel.
    dim_t ldp_a() const { return mr * bbm; }
    dim_t ldp_b() const { return nr * bbn; }

    // C(m x n) = beta*C + alpha*A_r*B_r over k steps; m <= mr, n <= nr.
    void gemm(dim_t m, dim_t n, dim_t k, const double* alpha, const double* a,
              const double* b, const double* beta, double* c, inc_t rs_c,
              inc_t cs_c, auxinfo_t* aux) const
    {
#if defined(AOCL_BLIS)
        if (m == mr && n == nr)
        {
            gemm_old(k, const_cast<double*>(alpha), const_cast<double*>(a),
                     const_cast<double*>(b), const_cast<double*>(beta), c,
                     rs_c, cs_c, aux, cntx);
            return;
        }
        // the 0.x micro-kernel has no edge-case support: run it on a
        // full temporary tile, then merge the valid part
        double ct[64 * 64];
        const double zero = 0.0;
        gemm_old(k, const_cast<double*>(alpha), const_cast<double*>(a),
                 const_cast<double*>(b), const_cast<double*>(&zero), ct, nr, 1,
                 aux, cntx);
        for (dim_t i = 0; i < m; i++)
            for (dim_t j = 0; j < n; j++)
                c[i * rs_c + j * cs_c] = *beta * c[i * rs_c + j * cs_c] + ct[i * nr + j];
#else
        // BLIS >= 1.0 kernels handle m < mr and n < nr themselves
        gemm_new(m, n, k, alpha, a, b, beta, c, rs_c, cs_c, aux, cntx);
#endif
    }

    // Pack one micro-panel of A: dim (<= mr) rows by len columns, read with
    // stride inc along the rows and ld along the columns, into p.
    void pack_a(dim_t dim, dim_t len, const double* src, inc_t inc, inc_t ld,
                double* p) const
    {
        pack(fallback_pack_a, true, dim, mr, bbm, len, src, inc, ld, p, ldp_a());
    }

    // Pack one micro-panel of B: dim (<= nr) columns by len rows.
    void pack_b(dim_t dim, dim_t len, const double* src, inc_t inc, inc_t ld,
                double* p) const
    {
        pack(fallback_pack_b, false, dim, nr, bbn, len, src, inc, ld, p, ldp_b());
    }

private:
#if defined(AOCL_BLIS)
    cntx_t* cntx = nullptr;          // the 0.x kernels take a mutable context
#else
    const cntx_t* cntx = nullptr;    // BLIS >= 1.0 hands out a const one
#endif
#if defined(AOCL_BLIS)
    using gemm_old_ft = void (*)(dim_t, double*, double*, double*, double*,
                                 double*, inc_t, inc_t, auxinfo_t*, cntx_t*);
    using pack_old_ft = void (*)(conj_t, pack_t, dim_t, dim_t, dim_t, double*,
                                 double*, inc_t, inc_t, double*, inc_t, cntx_t*);
    gemm_old_ft gemm_old = nullptr;
    pack_old_ft pack_old_a = nullptr, pack_old_b = nullptr;
#else
    using gemm_new_ft = void (*)(dim_t, dim_t, dim_t, const void*, const void*,
                                 const void*, const void*, void*, inc_t, inc_t,
                                 auxinfo_t*, const cntx_t*);
    using pack_new_ft = void (*)(conj_t, pack_t, dim_t, dim_t, dim_t, dim_t,
                                 dim_t, const void*, const void*, inc_t, inc_t,
                                 void*, inc_t, const void*, const cntx_t*);
    gemm_new_ft gemm_new = nullptr;
    pack_new_ft pack_new = nullptr;
#endif

    void pack(bool fallback, bool is_a, dim_t dim, dim_t dim_max, dim_t bcast,
              dim_t len, const double* src, inc_t inc, inc_t ld, double* p,
              dim_t ldp) const
    {
        static const double one = 1.0;
        if (fallback)
        {
            // what the reference kernel does: copy, broadcast, zero-pad
            for (dim_t l = 0; l < len; l++, p += ldp)
                for (dim_t d = 0; d < dim_max; d++)
                    for (dim_t b = 0; b < bcast; b++)
                        p[d * bcast + b] = d < dim ? src[d * inc + l * ld] : 0.0;
            return;
        }
#if defined(AOCL_BLIS)
        (void)bcast; (void)dim_max;
        (is_a ? pack_old_a : pack_old_b)(
            BLIS_NO_CONJUGATE, is_a ? BLIS_PACKED_ROW_PANELS : BLIS_PACKED_COL_PANELS,
            dim, len, len, const_cast<double*>(&one), const_cast<double*>(src),
            inc, ld, p, ldp, cntx);
#else
        (void)is_a;
        pack_new(BLIS_NO_CONJUGATE, BLIS_PACKED_PANELS, dim, dim_max, bcast, len,
                 len, &one, src, inc, ld, p, ldp, nullptr, cntx);
#endif
    }
};

} // namespace gotobench
