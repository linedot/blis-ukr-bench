#include "dynamic_aligned_allocator.hpp"
#include "performance_counters/performance_counters.hpp"

#include <omp.h> // Include this first
extern "C" {
    #include <blis.h>
}

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <print>
#include <string>
#include <vector>
#include <algorithm>

int main(int argc, char *argv[])
{
    if (argc != 8)
    {
        std::print("usage: {} <alignment> <start_offset> <offset_step> <byte_factor> <measurements> <max_k_power> <add_k>\n", argv[0]);
        return -1;
    }
    using hrc = std::chrono::high_resolution_clock;

    std::vector<std::string> counters = {{"CYCLES"}, {"L1D_READ_MISS"}};
    bli_init();

    auto c = bli_info_get_gemm_ukr_impl_string(BLIS_NAT, BLIS_DOUBLE);
    std::print("# ukr implementation: {}\n", c);

    #if defined(AOCL_BLIS)
    auto* cntx = bli_gks_query_nat_cntx();
    auto ukr = reinterpret_cast<gemm_ukr_vft>(bli_cntx_get_l3_nat_ukr_dt(BLIS_DOUBLE, BLIS_GEMM_UKR, cntx));
    #else
    auto* cntx = bli_gks_query_cntx();
    auto ukr = reinterpret_cast<gemm_ukr_ft>(bli_cntx_get_ukr_dt(BLIS_DOUBLE, BLIS_GEMM_UKR, cntx));
    #endif

    auto mr = bli_cntx_get_blksz_def_dt(BLIS_DOUBLE, BLIS_MR, cntx);
    auto nr = bli_cntx_get_blksz_def_dt(BLIS_DOUBLE, BLIS_NR, cntx);

    #if defined(ROWSTRIDE)
    inc_t rs_c = nr;
    inc_t cs_c = 1;
    #elif defined(COLSTRIDE)
    inc_t rs_c = 1;
    inc_t cs_c = mr;
    #else
    inc_t rs_c = 1;
    inc_t cs_c = 1;
    #endif

    std::print("# ukr size:            {}x{}\n", mr, nr);

    double alpha = 2.1;
    double beta = 1.3;
    const std::uint64_t byte_factor = std::strtoull(argv[4], nullptr, 10);
    const std::uint64_t measurements = std::strtoull(argv[5], nullptr, 10);
    const std::uint64_t max_k_power = std::strtoull(argv[6], nullptr, 10);
    const std::uint64_t add_k = std::strtoull(argv[7], nullptr, 10);
    const std::uint64_t alignment = std::strtoull(argv[1], nullptr, 10);
    const std::uint64_t start_offset = std::strtoull(argv[2], nullptr, 10);
    const std::uint64_t offset_step = std::strtoull(argv[3], nullptr, 10);

    int max_threads = omp_get_max_threads();
    std::print("# threads:             {}\n", max_threads);
    std::print("Size[Byte],[GFLOP/s],FLOPS,cycles/iter,l1d_read_miss/iter\n");

    std::vector<std::uint64_t> t_min_ns(max_threads);
    std::vector<std::uint64_t> t_min_cycles(max_threads);
    std::vector<std::uint64_t> t_min_l1d(max_threads);

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int num_threads = omp_get_num_threads();

        performance_counters pc(counters);

        auto alloca = archcomp::dynamic_aligned_allocator<double>(alignment, start_offset + (tid * 3 + 0) * offset_step);
        auto allocb = archcomp::dynamic_aligned_allocator<double>(alignment, start_offset + (tid * 3 + 1) * offset_step);
        auto allocc = archcomp::dynamic_aligned_allocator<double>(alignment, start_offset + (tid * 3 + 2) * offset_step);

        std::vector<double, archcomp::dynamic_aligned_allocator<double>> A(alloca);
        std::vector<double, archcomp::dynamic_aligned_allocator<double>> B(allocb);
        std::vector<double, archcomp::dynamic_aligned_allocator<double>> C(allocc);

        for (std::uint64_t kpow = 1; kpow < max_k_power; kpow++)
        {
            std::uint64_t k = std::max(1UL, (1UL << kpow) + add_k);
            double flops = 2.0 * mr * nr * k + 3.0 * mr * nr;

            A.resize(mr * k);
            B.resize(nr * k);
            C.resize(mr * nr);

            std::uint64_t size = A.size() * sizeof(double) +
                                 B.size() * sizeof(double) +
                                 C.size() * sizeof(double);

            std::uint64_t iterations = std::max(byte_factor / size, 10UL);

            auxinfo_t aux;
            bli_auxinfo_set_next_a(A.data(), &aux);
            bli_auxinfo_set_next_b(B.data(), &aux);

            // Warmup
            for (std::uint64_t m = 0; m < iterations / 4; m++)
            {
                #if defined(AOCL_BLIS)
                ukr(k, &alpha, A.data(), B.data(), &beta, C.data(), rs_c, cs_c, &aux, cntx);
                #else
                ukr(mr, nr, k, &alpha, A.data(), B.data(), &beta, C.data(), rs_c, cs_c, &aux, cntx);
                #endif
            }

            std::uint64_t local_min_ns = std::numeric_limits<std::uint64_t>::max();

            for (std::uint64_t i = 0; i < measurements; i++)
            {
                // sync up before starting
                #pragma omp barrier

                auto start = hrc::now();
                pc.tic();
                for (std::uint64_t m = 0; m < iterations; m++)
                {
                    #if defined(AOCL_BLIS)
                    ukr(k, &alpha, A.data(), B.data(), &beta, C.data(), rs_c, cs_c, &aux, cntx);
                    #else
                    ukr(mr, nr, k, &alpha, A.data(), B.data(), &beta, C.data(), rs_c, cs_c, &aux, cntx);
                    #endif
                }
                pc.toc_stat();
                auto stop = hrc::now();

                auto dur = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start);
                local_min_ns = std::min(local_min_ns, static_cast<std::uint64_t>(dur.count()) / iterations);
            }

            std::uint64_t local_min_cycles = 0;
            std::uint64_t local_min_l1d_miss = 0;
            auto results = pc.get_counter_statistics();

            for (const auto& res : results)
            {
                auto [name, min_val, avg_val, max_val] = res;
                if ("CYCLES" == name) local_min_cycles = min_val;
                if ("L1D_READ_MISS" == name) local_min_l1d_miss = min_val;
            }

            t_min_ns[tid] = local_min_ns;
            t_min_cycles[tid] = local_min_cycles;
            t_min_l1d[tid] = local_min_l1d_miss;
            pc.reset_counter_storage();

            #pragma omp barrier

            // Reduce stats
            #pragma omp single
            {
                std::uint64_t max_ns_across_threads = 0;
                std::uint64_t sum_cycles = 0;
                std::uint64_t sum_l1d = 0;

                for (int t = 0; t < num_threads; t++)
                {
                    max_ns_across_threads = std::max(max_ns_across_threads, t_min_ns[t]);
                    sum_cycles += t_min_cycles[t];
                    sum_l1d += t_min_l1d[t];
                }

                double total_flops = flops * num_threads;
                std::uint64_t avg_cycles_per_thread = sum_cycles / num_threads;
                std::uint64_t avg_l1d_per_thread = sum_l1d / num_threads;

                std::print("{},{},{},{},{}\n",
                    size,
                    total_flops / max_ns_across_threads,
                    total_flops,
                    avg_cycles_per_thread / iterations,
                    avg_l1d_per_thread / iterations
                );
            }
        }
    }

    return 0;
}
